"""Evaluate on same-library scenario: queries over the TRAINED refs gallery.

This is the realistic production case: the library is fixed, the model has
seen each ref during training (as a clean canonical sprite), and the user
uploads a render (rotated/scaled/on-bg) of an icon from that library.

  python pynet/eval_lib.py --model out/xxx.pt --manifest train_set_vit/manifest.json
"""
import argparse
import json
import os
import numpy as np
import torch
import torch.nn.functional as F

import common
from model import EmbedCNN
from preprocess import query_sprite, ref_sprite
from dataset import load_manifest
from eval import embed_sprites, _rotate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--manifest", default=common.TRAIN_VIT_MANIFEST)
    ap.add_argument("--img-dir", default=common.TRAIN_VIT_DIR)
    ap.add_argument("--ref-dir", default=common.SCRAPED_TRAIN_DIR)
    ap.add_argument("--max-q", type=int, default=1000)
    ap.add_argument("--skip-first", type=int, default=0, help="skip N manifest entries (holdout)")
    ap.add_argument("--rot-scan", type=int, default=8)
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

    ref_names = sorted(f for f in os.listdir(args.ref_dir) if f.endswith(".png"))
    ref_map = {n: i for i, n in enumerate(ref_names)}
    ref_sprites = [ref_sprite(os.path.join(args.ref_dir, n), common.SPRITE_N) for n in ref_names]
    ref_embs = embed_sprites(ref_sprites, net, device, rot_scan=1)
    ref_embs = ref_embs / (np.linalg.norm(ref_embs, axis=1, keepdims=True) + 1e-8)

    entries = load_manifest(args.manifest)[args.skip_first:]
    if args.max_q:
        entries = entries[:args.max_q]

    hits = {1: 0, 5: 0}
    for e in entries:
        sp = query_sprite(os.path.join(args.img_dir, e["image"]), common.SPRITE_N)
        if sp is None:
            continue
        if args.rot_scan <= 1:
            q = embed_sprites([sp], net, device, 1)[0]
            sims = ref_embs @ q
        else:
            per_ang = []
            for k in range(args.rot_scan):
                ang = 360.0 * k / args.rot_scan
                t = torch.from_numpy(sp).float() / 255.0
                t = t.permute(2, 0, 1).unsqueeze(0).to(device)
                per_ang.append(net.embed_only(_rotate(t, ang)).squeeze(0).detach().cpu().numpy())
            per_ang = np.stack(per_ang)
            sims = np.max(per_ang @ ref_embs.T, axis=0)
        order = np.argsort(-sims)
        rank = int(np.where(order == ref_map[e["src"]])[0][0]) + 1
        for k in (1, 5):
            if rank <= k:
                hits[k] += 1
    n = len(entries)
    print("same-library recall@1=%.3f recall@5=%.3f n=%d (gallery=%d)" % (
        hits[1] / n, hits[5] / n, n, len(ref_names)))


if __name__ == "__main__":
    main()
