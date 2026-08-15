"""Train embedding CNN (ref-centric, rotation-augmented).

Core idea: the discriminator must be ROTATION-INVARIANT. The reference icons
are the canonical form at 0deg; queries are arbitrary rotations. So we train
a classifier on reference sprites with heavy random rotation augmentation
(sample /class ~ infinite), plus the rendered query sprites (realistic noise).
The embedding layer then maps rotated sprites near their canonical ref.

  python pynet/train.py --mode ref+query --src-subset 300 --epochs 30 --tag exp
"""
import argparse
import json
import os
import time
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

import common
import model as model_mod
from preprocess import query_sprite, ref_sprite
from dataset import build_query_cache_canon


def load_ref_dataset(ref_dir, src_subset=0):
    names = sorted(f for f in os.listdir(ref_dir) if f.endswith(".png"))
    if src_subset and src_subset < len(names):
        random.Random(1).shuffle(names)
        names = names[:src_subset]
    sprites = np.stack([ref_sprite(os.path.join(ref_dir, n), common.SPRITE_N) for n in names])
    return sprites, names


def load_query_dataset(manifest_path, img_dir, max_q=0):
    entries = json.load(open(manifest_path, encoding="utf-8"))
    if max_q and max_q < len(entries):
        entries = entries[:max_q]
    X, y, names = [], [], []
    src_list = sorted(set(e["src"] for e in entries))
    sid = {s: i for i, s in enumerate(src_list)}
    ok = 0
    for e in entries:
        sp = query_sprite(os.path.join(img_dir, e["image"]), common.SPRITE_N)
        if sp is None:
            continue
        X.append(sp)
        y.append(sid[e["src"]])
        names.append(e["src"])
        ok += 1
    return np.stack(X), np.array(y), names

class SpriteAug:
    """Random rotation + scale + shift jitter + background compositing."""

    def __init__(self, rot_p=0.95, scale_p=0.6, shift_p=0.4, bg_p=0.8):
        self.rot_p, self.scale_p, self.shift_p, self.bg_p = rot_p, scale_p, shift_p, bg_p

    def __call__(self, x):
        # x: B x 4 x S x S in [0,1]
        B = x.shape[0]
        S = x.shape[-1]
        theta = torch.eye(2, 3, device=x.device).repeat(B, 1, 1)
        for i in range(B):
            if random.random() < self.rot_p:
                ang = random.random() * 360
                r = np.deg2rad(ang)
                c, s = np.cos(r), np.sin(r)
                theta[i, 0, 0], theta[i, 0, 1] = c, -s
                theta[i, 1, 0], theta[i, 1, 1] = s, c
            scale = 1.0
            if random.random() < self.scale_p:
                scale = np.random.uniform(0.8, 1.2)
            theta[i, :2, :2] *= scale
            if random.random() < self.shift_p:
                sh = np.random.uniform(-0.08, 0.08)
                theta[i, 0, 2] = sh
                theta[i, 1, 2] = np.random.uniform(-0.08, 0.08)
        grid = F.affine_grid(theta, x.shape, align_corners=False)
        out = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        if self.bg_p > 0:
            # composite onto a random light/dark background: simulates the
            # render pipeline's bg-blended edges (closes ref/query domain gap)
            for i in range(B):
                if random.random() < self.bg_p:
                    bg = torch.rand(3, device=x.device) * 0.9 + 0.05
                    alpha = out[i, 3:4]
                    out[i, :3] = out[i, :3] * alpha + bg.view(3, 1, 1) * (1 - alpha)
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="train_set_vit")
    ap.add_argument("--src-subset", type=int, default=0, help="limit train refs")
    ap.add_argument("--max-q", type=int, default=0, help="limit query samples")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--emb", type=int, default=64)
    ap.add_argument("--tag", default="exp")
    ap.add_argument("--ref-weight", type=float, default=0.5,
                    help="fraction of each batch from refs (rest = queries)")
    ap.add_argument("--no-query", action="store_true")
    ap.add_argument("--ch", type=str, default="16,32,48,64")
    ap.add_argument("--iters", type=int, default=200, help="optimizer steps per epoch")
    ap.add_argument("--eval-every", type=int, default=0, help="quick-eval every N epochs (0=off)")
    ap.add_argument("--rot-cons", type=float, default=0.0, help="rotation-consistency loss weight")
    ap.add_argument("--rot-p", type=float, default=0.5, help="rotation aug probability")
    ap.add_argument("--res", type=int, default=48, help="sprite input resolution")
    ap.add_argument("--arcface", action="store_true", help="use ArcFace margin head")
    ap.add_argument("--s", type=float, default=20.0, help="arcface scale")
    ap.add_argument("--m", type=float, default=0.30, help="arcface margin")
    ap.add_argument("--cosine-lr", action="store_true", help="cosine LR schedule")
    args = ap.parse_args()

    os.makedirs(common.OUT_DIR, exist_ok=True)

    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
    else:
        print("CPU mode (no CUDA)")

    side = args.res
    common.SPRITE_N = side

    ref_sprites, ref_names = load_ref_dataset(common.SCRAPED_TRAIN_DIR, args.src_subset)
    # full src2id across both refs and queries
    src2id = {s: i for i, s in enumerate(ref_names)}
    X_ref = torch.from_numpy(ref_sprites).float() / 255.0
    X_ref = X_ref.permute(0, 3, 1, 2).contiguous().to(dev)
    y_ref = torch.tensor([src2id[s] for s in ref_names], dtype=torch.long).to(dev)

    X_q = y_q = None
    if not args.no_query:
        cache = os.path.join(common.CACHE_DIR, "qcache_canon_%s_%d.npz" % (args.data, side))
        q_sprites, q_names, q_rots, _ = build_query_cache_canon(
            json.load(open(os.path.join(common.ROOT, args.data, "manifest.json"), encoding="utf-8")),
            os.path.join(common.ROOT, args.data), side, cache, args.max_q)
        q_ids = []
        for s in q_names:
            q_ids.append(src2id[s] if s in src2id else -1)
        q_ids = np.array(q_ids)
        keep = q_ids >= 0
        q_ids = q_ids[keep]
        q_sprites = q_sprites[keep]
        X_q = torch.from_numpy(q_sprites).float() / 255.0
        X_q = X_q.permute(0, 3, 1, 2).contiguous().to(dev)
        y_q = torch.tensor(q_ids, dtype=torch.long).to(dev)
        print("canonical query samples", X_q.shape[0])

    n_classes = len(src2id)
    ch = tuple(int(c) for c in args.ch.split(","))
    net = model_mod.EmbedCNN(in_ch=4, emb_dim=args.emb, n_classes=0 if args.arcface else n_classes, ch=ch).to(dev)
    if args.arcface:
        head = model_mod.ArcMarginHead(args.emb, n_classes, s=args.s, m=args.m).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    if args.arcface:
        opt = torch.optim.Adam(list(net.parameters()) + list(head.parameters()), lr=args.lr)
    crit = nn.CrossEntropyLoss()
    aug = SpriteAug(rot_p=args.rot_p)
    if args.cosine_lr:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * 0.02)
    print("refs", X_ref.shape[0], "classes", n_classes, "params", net.param_count(),
          "+head" if args.arcface else "")

    t0 = time.time()
    n_ref = X_ref.shape[0]
    iters_per_epoch = args.iters
    for ep in range(1, args.epochs + 1):
        net.train()
        if args.arcface:
            head.train()
        tot = 0.0
        nb = 0
        n_ref_use = int(args.batch * args.ref_weight)
        n_q_use = args.batch - n_ref_use
        for _ in range(iters_per_epoch):
            ri = torch.randint(0, n_ref, (n_ref_use,))
            xb = aug(X_ref[ri])
            yb = y_ref[ri]
            if X_q is not None and n_q_use > 0:
                qi = torch.randint(0, X_q.shape[0], (n_q_use,))
                xb = torch.cat([xb, aug(X_q[qi])])
                yb = torch.cat([yb, y_q[qi]])
            e = net(xb)
            if args.arcface:
                logits = head(e, yb)
            else:
                _, logits = net(xb, with_emb=True)
            loss = crit(logits, yb)
            if args.rot_cons > 0:
                # rotation-consistency: embed again under a random rotation, pull closer
                xr = aug(torch.ones_like(xb) * 0 + xb)  # fresh random rotation
                er = net.embed_only(xr)
                loss = loss + args.rot_cons * (1.0 - (e * er).sum(1)).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
            nb += 1
        if args.cosine_lr:
            sched.step()
        acc = _acc(net, X_ref, head if args.arcface else None, args)
        line = "epoch %3d loss %.4f ref_cls_acc %.3f [%.0fs]" % (ep, tot / max(1, nb), acc, time.time() - t0)
        print(line)
        with open(os.path.join(common.OUT_DIR, "train_log_%s.txt" % args.tag), "a") as f:
            f.write(line + "\n")
        if args.eval_every and ep % args.eval_every == 0:
            _quick_eval(net, args.tag, ep, head if args.arcface else None, args)

    path = os.path.join(common.OUT_DIR, "%s.pt" % args.tag)
    save = {"state": net.state_dict(), "emb": args.emb,
            "n_classes": n_classes, "src2id": src2id, "ch": ch, "res": side}
    if args.arcface:
        save["head_state"] = head.state_dict()
    torch.save(save, path)
    print("saved", path, "params", net.param_count())


def _acc(net, X, head, args):
    dev = next(net.parameters()).device
    net.eval()
    with torch.no_grad():
        idx = torch.arange(min(1500, X.shape[0]), device=dev)
        e = net(X[idx])
        if head is not None:
            logits = head(e, idx)
        else:
            _, logits = net(X[idx], with_emb=True)
        a = (logits.argmax(1) == idx).float().mean().item()
    net.train()
    return a


def _quick_eval(net, tag, ep, head, args):
    """Save weights then run eval.py on in-domain + cross quickly."""
    path = os.path.join(common.OUT_DIR, "%s_ep%d.pt" % (tag, ep))
    save = {"state": net.state_dict(), "emb": net.emb_dim,
            "n_classes": net.head.out_features if net.head else args.src_subset,
            "src2id": {}, "ch": net.ch, "res": args.res,
            "attn": getattr(net, "attn", "none")}
    if head is not None:
        save["head_state"] = head.state_dict()
    torch.save(save, path)
    import subprocess, sys
    subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), "eval.py"),
                    "--model", path, "--rot-scan", "8"], timeout=900)
    os.remove(path)


if __name__ == "__main__":
    main()
