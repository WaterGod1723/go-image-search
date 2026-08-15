"""Log-polar FFT-pool retrieval: exact rotation-invariance without learning.

Uses per-ring |FFT| along the angular axis of the log-polar grid as the
representation, cosine retrieval. Establishes the ceiling of a rotation-
invariant feature, then a small CNN can sharpen it.
"""
import os
import sys
import numpy as np
import json
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
from logpolar import sprite_to_logpolar, fft_pool
from preprocess import query_sprite, ref_sprite
from dataset import load_manifest


def lp_feat(sp, R=24, T=32):
    f = fft_pool(*sprite_to_logpolar(sp, R, T))
    n = np.linalg.norm(f)
    return f / (n + 1e-8)


def eval_set(manifest_path, img_dir, ref_dir, R=24, T=32, topk=(1, 5)):
    entries = load_manifest(manifest_path)
    ref_names = sorted(f for f in os.listdir(ref_dir) if f.endswith(".png"))
    ref_map = {n: i for i, n in enumerate(ref_names)}
    ref_embs = np.stack([lp_feat(ref_sprite(os.path.join(ref_dir, n), common.SPRITE_N), R, T)
                         for n in ref_names])
    hits = {k: 0 for k in topk}
    segfail = 0
    for e in entries:
        sp = query_sprite(os.path.join(img_dir, e["image"]), common.SPRITE_N)
        if sp is None:
            segfail += 1
            continue
        q = lp_feat(sp, R, T)
        sims = ref_embs @ q
        order = np.argsort(-sims)
        rank = int(np.where(order == ref_map[e["src"]])[0][0]) + 1
        for k in topk:
            if rank <= k:
                hits[k] += 1
    n = len(entries)
    return {"recall@1": hits[1] / n, "recall@5": hits[5] / n, "n": n, "segfail": segfail}


if __name__ == "__main__":
    only = sys.argv[1] if len(sys.argv) > 1 else "all"
    for name, m, d, r in [
        ("in-domain", common.TEST_INDOMAIN_MANIFEST, common.TEST_INDOMAIN_DIR, common.TEST_PNGS_DIR),
        ("cross", common.TEST_CROSS_MANIFEST, common.TEST_CROSS_DIR, common.SCRAPED_TEST_DIR),
    ]:
        if only != "all" and name != only:
            continue
        res = eval_set(m, d, r)
        print("%-10s recall@1=%.3f recall@5=%.3f n=%d segfail=%d" % (
            name, res["recall@1"], res["recall@5"], res["n"], res["segfail"]))
