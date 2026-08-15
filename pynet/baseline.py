"""Baseline: rotation-scan normalized template matching (no ML).

Establishes the well-posedness ceiling: how high recall can go if rotation
were perfectly handled. Rotates the query sprite through K angles, computes
mask IoU + RGB correlation against each ref at 0deg, takes the best angle.
"""
import os
import numpy as np
from PIL import Image

import common
from preprocess import query_sprite, ref_sprite
from dataset import load_manifest


def sprite_feats(sp):
    mask = (sp[..., 3] > 10).astype(np.float32)
    rgb = sp[..., :3].astype(np.float32) / 255.0
    rgb = rgb * mask[..., None]
    return mask, rgb


def rotate_sprite(sp, ang):
    from PIL import Image
    im = Image.fromarray(sp).rotate(ang, resample=Image.BILINEAR)
    return np.asarray(im)


def score_matrix(qm, qr, ref_masks, ref_rgbs):
    """Vectorized scores vs all refs. ref_masks: (R,S,S), ref_rgbs: (R,S,S,3)."""
    R = len(ref_masks)
    qm_b = qm > 0.5
    inter = (ref_masks > 0.5) & qm_b[None]
    inter_n = inter.sum((1, 2)).astype(np.float64)
    union = (ref_masks > 0.5) | qm_b[None]
    union_n = union.sum((1, 2)).astype(np.float64)
    iou = inter_n / (union_n + 1e-8)
    fg = inter  # (R,S,S) bool
    fg_count = fg.sum((1, 2)).astype(np.float64)
    corr = np.zeros(R)
    valid = fg_count > 10
    if valid.any():
        f = fg[valid].astype(np.float64)  # (R,S,S)
        cnt = fg_count[valid]  # (R,)
        a = (qr[None].astype(np.float64) * f[..., None])
        b = (ref_rgbs[valid].astype(np.float64) * f[..., None])
        amean = (a.sum((1, 2)) / cnt[:, None])[:, None, None, :]
        bmean = (b.sum((1, 2)) / cnt[:, None])[:, None, None, :]
        a = a - amean
        b = b - bmean
        dot = (a * b).sum((1, 2, 3))
        na = np.sqrt((a * a).sum((1, 2, 3))) + 1e-8
        nb = np.sqrt((b * b).sum((1, 2, 3))) + 1e-8
        corr[valid] = dot / (na * nb)
    return 0.5 * iou + 0.5 * np.clip(corr, 0, None)


def eval_set(manifest_path, img_dir, ref_dir, K=16):
    entries = load_manifest(manifest_path)
    ref_names = sorted(f for f in os.listdir(ref_dir) if f.endswith(".png"))
    ref_map = {n: i for i, n in enumerate(ref_names)}
    ref_f = [sprite_feats(ref_sprite(os.path.join(ref_dir, n), common.SPRITE_N)) for n in ref_names]
    ref_masks = np.stack([r[0] for r in ref_f])
    ref_rgbs = np.stack([r[1] for r in ref_f])

    hits1 = hits5 = 0
    fails = 0
    for e in entries:
        sp = query_sprite(os.path.join(img_dir, e["image"]), common.SPRITE_N)
        if sp is None:
            fails += 1
            continue
        qm, qr = sprite_feats(sp)
        scores = []
        for ang in np.linspace(0, 360, K, endpoint=False):
            rsp = rotate_sprite(sp, ang)
            qm2, qr2 = sprite_feats(rsp)
            scores.append(score_matrix(qm2, qr2, ref_masks, ref_rgbs))
        scores = np.stack(scores)
        best_over_angles = scores.max(0)
        order = np.argsort(-best_over_angles)
        rank = int(np.where(order == ref_map[e["src"]])[0][0]) + 1
        if rank <= 1:
            hits1 += 1
        if rank <= 5:
            hits5 += 1
    n = len(entries)
    return {"recall@1": hits1 / n, "recall@5": hits5 / n, "n": n, "seg_fail": fails}


if __name__ == "__main__":
    import sys, time
    only = sys.argv[1] if len(sys.argv) > 1 else "all"
    t0 = time.time()
    for name, m, d, r in [
        ("in-domain", common.TEST_INDOMAIN_MANIFEST, common.TEST_INDOMAIN_DIR, common.TEST_PNGS_DIR),
        ("cross", common.TEST_CROSS_MANIFEST, common.TEST_CROSS_DIR, common.SCRAPED_TEST_DIR),
    ]:
        if only != "all" and name != only:
            continue
        res = eval_set(m, d, r, K=16)
        print("%-10s recall@1=%.3f recall@5=%.3f n=%d segfail=%d [%.0fs]" % (
            name, res["recall@1"], res["recall@5"], res["n"], res["seg_fail"], time.time() - t0))
