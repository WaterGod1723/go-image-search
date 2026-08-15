"""Evaluate icon retrieval: recall@k on in-domain and cross-domain test sets.

Pipeline: query sprite -> embed; gallery ref sprite -> embed; cosine top-k.

  python pynet/eval.py --model out/model.pt [--rot-scan 8]
"""
import argparse
import os
import json
import numpy as np
import torch
import torch.nn.functional as F

import common
from model import EmbedCNN
from preprocess import query_sprite, ref_sprite
from dataset import load_manifest


def embed_sprites(sprites, net, device, rot_scan=0):
    """sprites: list of NxNx4 uint8 -> (N, emb) L2-normalized.
    If rot_scan>0, average-pool embeddings over rot_scan rotations.
    """
    net.eval()
    embs = []
    with torch.no_grad():
        for sp in sprites:
            t = torch.from_numpy(sp).float() / 255.0
            t = t.permute(2, 0, 1).unsqueeze(0).to(device)
            if rot_scan <= 1:
                e = net.embed_only(t)
            else:
                es = []
                for k in range(rot_scan):
                    ang = 360.0 * k / rot_scan
                    tt = _rotate(t, ang)
                    es.append(net.embed_only(tt))
                e = F.normalize(torch.stack(es).mean(0), dim=1)
            embs.append(e.squeeze(0).cpu().numpy())
    return np.stack(embs)


def _rotate(x, ang):
    import math
    from torch.nn import functional as Fn
    rad = math.radians(ang)
    c, s = math.cos(rad), math.sin(rad)
    theta = torch.zeros(1, 2, 3).to(x.device)
    theta[0, 0, 0], theta[0, 0, 1], theta[0, 0, 2] = c, -s, 0
    theta[0, 1, 0], theta[0, 1, 1], theta[0, 1, 2] = s, c, 0
    grid = Fn.affine_grid(theta, x.shape, align_corners=False)
    return Fn.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)


def eval_set(manifest_path, img_dir, ref_dir, net, device, rot_scan=0, topk=(1, 5)):
    entries = load_manifest(manifest_path)
    # gallery refs
    ref_names = sorted(f for f in os.listdir(ref_dir) if f.endswith(".png"))
    ref_map = {n: i for i, n in enumerate(ref_names)}
    ref_sprites = [ref_sprite(os.path.join(ref_dir, n), common.SPRITE_N) for n in ref_names]
    ref_embs = embed_sprites(ref_sprites, net, device, rot_scan=1)
    ref_embs = ref_embs / (np.linalg.norm(ref_embs, axis=1, keepdims=True) + 1e-8)

    hits = {k: 0 for k in topk}
    miss_list = []
    for e in entries:
        sp = query_sprite(os.path.join(img_dir, e["image"]), common.SPRITE_N)
        if sp is None:
            miss_list.append((e["image"], "seg_fail"))
            continue
        if rot_scan <= 1:
            q = embed_sprites([sp], net, device, 1)[0]
            sims = ref_embs @ q
        else:
            # max-over-rotations: embed query at K rotations, take best sim per ref
            q_embs = embed_sprites([sp], net, device, rot_scan)  # (K, emb) via mean-pool
            # fall back to per-angle max
            angles = [360.0 * k / rot_scan for k in range(rot_scan)]
            per_ang = []
            for a in angles:
                t = torch.from_numpy(sp).float() / 255.0
                t = t.permute(2, 0, 1).unsqueeze(0).to(device)
                tt = _rotate(t, a)
                per_ang.append(net.embed_only(tt).squeeze(0).detach().cpu().numpy())
            per_ang = np.stack(per_ang)
            sims = np.max(per_ang @ ref_embs.T, axis=0)
        order = np.argsort(-sims)
        rank = int(np.where(order == ref_map[e["src"]])[0][0]) + 1
        for k in topk:
            if rank <= k:
                hits[k] += 1
    n = len(entries)
    out = {}
    for k in topk:
        out[k] = hits[k] / n
    out["miss"] = miss_list
    out["n"] = n
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--rot-scan", type=int, default=1)
    ap.add_argument("--indomain", action="store_true", default=True)
    ap.add_argument("--cross", action="store_true", default=True)
    args = ap.parse_args()

    ck = torch.load(args.model, map_location="cpu", weights_only=False)
    res = ck.get("res", common.SPRITE_N)
    common.SPRITE_N = res
    has_head = any(k.startswith("head.") for k in ck["state"])
    attn = ck.get("attn", "none")
    if not has_head:
        net = EmbedCNN(in_ch=4, emb_dim=ck["emb"], n_classes=0, ch=ck["ch"], attn=attn)
    else:
        net = EmbedCNN(in_ch=4, emb_dim=ck["emb"], n_classes=ck["n_classes"], ch=ck["ch"], attn=attn)
    net.load_state_dict(ck["state"])
    net.eval()
    device = torch.device("cpu")
    print("params", net.param_count())

    if args.indomain:
        r = eval_set(common.TEST_INDOMAIN_MANIFEST, common.TEST_INDOMAIN_DIR,
                     common.TEST_PNGS_DIR, net, device, args.rot_scan)
        print("in-domain (66 refs) recall@1=%.4f @5=%.4f n=%d" % (r[1], r[5], r["n"]))
    if args.cross:
        r = eval_set(common.TEST_CROSS_MANIFEST, common.TEST_CROSS_DIR,
                     common.SCRAPED_TEST_DIR, net, device, args.rot_scan)
        print("cross-domain (1105 refs) recall@1=%.4f @5=%.4f n=%d" % (r[1], r[5], r["n"]))


if __name__ == "__main__":
    main()
