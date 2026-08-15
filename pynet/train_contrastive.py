"""Contrastive embedding training (InfoNCE over query-ref pairs).

The retrieval metric IS cosine@k, so optimize it directly: sample a batch of
queries and their canonical refs; the loss pulls each query to its own ref and
pushes away from all other refs in the batch (in-batch negatives). This is the
standard 'query-anchored' contrastive objective.

  python pynet/train_contrastive.py --max-q 8000 --epochs 40 --tag ctr
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

import common
import model as model_mod
from preprocess import query_sprite, ref_sprite
from dataset import build_query_cache_canon


class SpriteAug:
    """Rotation + scale + shift + bg composite + downscale-blur (render sim)."""

    def __init__(self, rot_p=0.7, scale_p=0.6, shift_p=0.4, bg_p=0.8, blur_p=0.7,
                 scale_lo=0.8, scale_hi=1.2):
        self.rot_p, self.scale_p, self.shift_p, self.bg_p = rot_p, scale_p, shift_p, bg_p
        self.blur_p = blur_p
        self.scale_lo, self.scale_hi = scale_lo, scale_hi

    def __call__(self, x):
        B = x.shape[0]
        S = x.shape[-1]
        theta = torch.eye(2, 3, device=x.device).repeat(B, 1, 1)
        for i in range(B):
            if random.random() < self.rot_p:
                r = np.deg2rad(random.random() * 360)
                c, s = np.cos(r), np.sin(r)
                theta[i, 0, 0], theta[i, 0, 1] = c, -s
                theta[i, 1, 0], theta[i, 1, 1] = s, c
            if random.random() < self.scale_p:
                sc = np.random.uniform(self.scale_lo, self.scale_hi)
                theta[i, :2, :2] *= sc
            if random.random() < self.shift_p:
                theta[i, 0, 2] = np.random.uniform(-0.08, 0.08)
                theta[i, 1, 2] = np.random.uniform(-0.08, 0.08)
        grid = F.affine_grid(theta, x.shape, align_corners=False)
        out = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        if self.bg_p > 0:
            for i in range(B):
                if random.random() < self.bg_p:
                    bg = torch.rand(3, device=x.device) * 0.9 + 0.05
                    alpha = out[i, 3:4]
                    out[i, :3] = out[i, :3] * alpha + bg.view(3, 1, 1) * (1 - alpha)
        if self.blur_p > 0:
            # simulate render downscale loss: avg-pool then upscale
            k = 2
            for i in range(B):
                if random.random() < self.blur_p:
                    small = F.avg_pool2d(out[i:i + 1], k)
                    out[i:i + 1] = F.interpolate(small, scale_factor=k,
                                                  mode="bilinear", align_corners=False)
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-q", type=int, default=8000)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--batch", type=int, default=128, help="query pairs per step")
    ap.add_argument("--emb", type=int, default=64)
    ap.add_argument("--ch", type=str, default="16,32,48,64")
    ap.add_argument("--res", type=int, default=64)
    ap.add_argument("--temp", type=float, default=0.15, help="InfoNCE temperature")
    ap.add_argument("--iters", type=int, default=150)
    ap.add_argument("--tag", default="ctr")
    ap.add_argument("--rot-p", type=float, default=0.7)
    ap.add_argument("--eval-every", type=int, default=0)
    ap.add_argument("--cosine-lr", action="store_true")
    ap.add_argument("--raw", action="store_true",
                    help="use raw (non-de-rotated) query sprites as anchors")
    ap.add_argument("--blur-p", type=float, default=0.7)
    ap.add_argument("--hard-neg", type=int, default=0,
                    help="number of mask-duplicate hard negatives per src (0=off)")
    ap.add_argument("--attn", default="none", choices=["none", "se", "cbam"])
    ap.add_argument("--ref-anchor", type=float, default=0.0,
                    help="weight of ref-as-anchor self-contrastive stream (0=off)")
    ap.add_argument("--scale-lo", type=float, default=0.8)
    ap.add_argument("--scale-hi", type=float, default=1.2)
    args = ap.parse_args()

    os.makedirs(common.OUT_DIR, exist_ok=True)
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    common.SPRITE_N = args.res

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
    else:
        print("CPU mode (no CUDA)")

    # load query sprites (canonical de-rotated by default, or raw with --raw)
    cache = os.path.join(common.CACHE_DIR, "qcache_%s_%d.npz" % (
        "raw" if args.raw else "canon", args.res))
    if args.raw:
        from dataset import build_query_cache
        q_sprites, q_names, _ = build_query_cache(
            json.load(open(common.TRAIN_VIT_MANIFEST, encoding="utf-8")),
            common.TRAIN_VIT_DIR, args.res, cache, 0)
    else:
        q_sprites, q_names, _, _ = build_query_cache_canon(
            json.load(open(common.TRAIN_VIT_MANIFEST, encoding="utf-8")),
            common.TRAIN_VIT_DIR, args.res, cache, 0)
    if args.max_q and args.max_q < len(q_sprites):
        q_sprites = q_sprites[:args.max_q]
        q_names = q_names[:args.max_q]

    # ALL reference icons (not just those appearing in queries) so every gallery
    # ref gets a trained embedding. Ref-as-anchor stream teaches "augmented ref
    # -> clean ref" for every ref.
    all_ref_names = sorted(f for f in os.listdir(common.SCRAPED_TRAIN_DIR) if f.endswith(".png"))
    src2id = {s: i for i, s in enumerate(all_ref_names)}
    ref_cache = {s: ref_sprite(os.path.join(common.SCRAPED_TRAIN_DIR, s), args.res)
                 for s in all_ref_names}
    R_sp = np.stack([ref_cache[s] for s in all_ref_names])
    X_ref = torch.from_numpy(R_sp).float() / 255.0
    X_ref = X_ref.permute(0, 3, 1, 2).contiguous().to(dev)
    n_refs = X_ref.shape[0]

    # queries: keep only those whose src is in the ref set (should be all)
    q_keep = [i for i, s in enumerate(q_names) if s in src2id]
    q_sprites = q_sprites[q_keep]
    q_names = [q_names[i] for i in q_keep]
    X_q = torch.from_numpy(q_sprites).float() / 255.0
    X_q = X_q.permute(0, 3, 1, 2).contiguous().to(dev)
    y_q = torch.tensor([src2id[s] for s in q_names], dtype=torch.long).to(dev)

    ch = tuple(int(c) for c in args.ch.split(","))
    net = model_mod.EmbedCNN(in_ch=4, emb_dim=args.emb, n_classes=None, ch=ch, attn=args.attn).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    aug = SpriteAug(rot_p=args.rot_p, blur_p=args.blur_p,
                    scale_lo=args.scale_lo, scale_hi=args.scale_hi)
    if args.cosine_lr:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * 0.02)
    print("queries", X_q.shape[0], "refs", n_refs, "params", net.param_count(), "attn", args.attn)

    # hard-negative sampling: per-src list of refs whose mask is similar
    dup_map = None
    if args.hard_neg:
        # use already-normalized ref sprites R_sp (N x S x S x 4)
        masks = (R_sp[..., 3] > 10).astype(np.float32)
        n_src = n_refs
        flat = masks.reshape(n_src, -1)
        inter = flat @ flat.T
        union = flat.sum(1)[:, None] + flat.sum(1)[None, :] - inter
        iou = inter / np.maximum(union, 1)
        np.fill_diagonal(iou, 0)
        dup_map = []
        for i in range(n_src):
            cand = np.argsort(-iou[i])[:args.hard_neg]
            dup_map.append([int(c) for c in cand if iou[i, c] > 0.5])
        print("hard-negative map built (avg dups/src: %.1f)" % np.mean([len(x) for x in dup_map]))

    t0 = time.time()
    nq = X_q.shape[0]
    nr = X_ref.shape[0]
    for ep in range(1, args.epochs + 1):
        net.train()
        tot = 0.0
        for _ in range(args.iters):
            qi = torch.randint(0, nq, (args.batch,))
            # positive ref for query qi is the ref of the same source
            # y_q holds the src id; refs are indexed by src id in X_ref
            ri = y_q[qi]
            if dup_map is not None:
                # sample a second ref set that includes hard negatives (mask dups)
                r2 = []
                for s in ri.tolist():
                    dups = dup_map[s]
                    if dups and random.random() < 0.7:
                        r2.append(random.choice(dups))
                    else:
                        r2.append(random.randint(0, nr - 1))
                r2 = torch.tensor(r2, device=dev)
                all_r = torch.cat([ri, r2])
                qe = net.embed_only(aug(X_q[qi]))
                re = net.embed_only(aug(X_ref[all_r]))
                # positives are ri at their column; negatives include dup copies
                logits = qe @ re.T / args.temp
                labels = torch.arange(args.batch, device=qe.device)
                loss = F.cross_entropy(logits, labels)
            else:
                qe = net.embed_only(aug(X_q[qi]))       # query embeddings
                re = net.embed_only(aug(X_ref[ri]))     # ref embeddings
                # InfoNCE: each query's positive is the ref with same src
                labels = torch.arange(args.batch, device=qe.device)
                logits = qe @ re.T / args.temp
                loss = F.cross_entropy(logits, labels)
            # ref-as-anchor stream: augmented ref -> clean ref, for ALL refs
            if args.ref_anchor > 0:
                ri2 = torch.randint(0, nr, (args.batch,))
                ra = net.embed_only(aug(X_ref[ri2]))     # augmented ref
                rc = net.embed_only(X_ref[ri2])          # clean ref
                labels2 = torch.arange(args.batch, device=qe.device)
                logits2 = ra @ rc.T / args.temp
                loss = loss + args.ref_anchor * F.cross_entropy(logits2, labels2)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
        if args.cosine_lr:
            sched.step()
        # estimate: query->ref retrieval accuracy in-batch
        net.eval()
        with torch.no_grad():
            qe = net.embed_only(X_q[:1000])
            re = net.embed_only(X_ref)
            sims = qe @ re.T
            correct = (sims.argmax(1) == y_q[:1000]).float().mean().item()
        net.train()
        line = "epoch %3d loss %.4f q2r_acc %.3f [%.0fs]" % (ep, tot / args.iters, correct, time.time() - t0)
        print(line)
        with open(os.path.join(common.OUT_DIR, "train_log_%s.txt" % args.tag), "a") as f:
            f.write(line + "\n")
        if args.eval_every and ep % args.eval_every == 0:
            _quick_eval(net, args.tag, ep, args)

    path = os.path.join(common.OUT_DIR, "%s.pt" % args.tag)
    torch.save({"state": net.state_dict(), "emb": args.emb,
                "n_classes": n_refs, "src2id": src2id,
                "ch": ch, "res": args.res, "attn": args.attn}, path)
    print("saved", path, "params", net.param_count())


def _quick_eval(net, tag, ep, args):
    path = os.path.join(common.OUT_DIR, "%s_ep%d.pt" % (tag, ep))
    torch.save({"state": net.state_dict(), "emb": net.emb_dim,
                "n_classes": 0, "src2id": {}, "ch": net.ch, "res": args.res,
                "attn": getattr(net, "attn", "none")}, path)
    import subprocess, sys
    subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), "eval.py"),
                    "--model", path, "--rot-scan", "8"], timeout=900)
    os.remove(path)


if __name__ == "__main__":
    main()
