"""Dataset builders: cache query sprites + ref sprites into numpy arrays."""
import json
import os
import numpy as np
import common
from preprocess import query_sprite, ref_sprite


def load_manifest(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_query_cache(manifest, img_dir, side, cache_path, max_n=None):
    """Extract sprites for queries. Returns (sprites NxNx4 uint8, src_names list)."""
    if os.path.exists(cache_path):
        d = np.load(cache_path, allow_pickle=True)
        return d["sprites"], list(d["srcs"]), list(d["imgs"])
    entries = manifest if max_n is None else manifest[:max_n]
    sprites, srcs, imgs = [], [], []
    ok = 0
    for e in entries:
        p = os.path.join(img_dir, e["image"])
        sp = query_sprite(p, side)
        if sp is None:
            continue
        sprites.append(sp)
        srcs.append(e["src"])
        imgs.append(e["image"])
        ok += 1
    arr = np.stack(sprites)
    np.savez_compressed(cache_path, sprites=arr, srcs=np.array(srcs), imgs=np.array(imgs))
    print("built query cache", cache_path, "ok", ok, "/", len(entries))
    return arr, srcs, imgs


def build_query_cache_canon(manifest, img_dir, side, cache_path, max_n=None):
    """Extract query sprites and de-rotate to canonical using manifest rotation.

    Returns (sprites NxNx4 uint8, src_names list, rotation list). The CNN then
    only needs to discriminate; rotation robustness is handled by rotation-scan
    at inference time.
    """
    if os.path.exists(cache_path):
        d = np.load(cache_path, allow_pickle=True)
        return d["sprites"], list(d["srcs"]), list(d["rots"]), list(d["imgs"])
    from PIL import Image
    entries = manifest if max_n is None else manifest[:max_n]
    sprites, srcs, rots, imgs = [], [], [], []
    ok = 0
    for e in entries:
        p = os.path.join(img_dir, e["image"])
        sp = query_sprite(p, side)
        if sp is None:
            continue
        im = Image.fromarray(sp).rotate(-e["rotation"], resample=Image.BILINEAR)
        sprites.append(np.asarray(im))
        srcs.append(e["src"])
        rots.append(e["rotation"])
        imgs.append(e["image"])
        ok += 1
    arr = np.stack(sprites)
    np.savez_compressed(cache_path, sprites=arr, srcs=np.array(srcs),
                        rots=np.array(rots), imgs=np.array(imgs))
    print("built canonical query cache", cache_path, "ok", ok, "/", len(entries))
    return arr, srcs, rots, imgs


def build_ref_cache(ref_dir, side, cache_path):
    """Extract sprites for gallery refs. Returns dict src_name -> sprite."""
    if os.path.exists(cache_path):
        d = np.load(cache_path, allow_pickle=True)
        return dict(zip(d["names"], d["sprites"]))
    names = sorted(f for f in os.listdir(ref_dir) if f.endswith(".png"))
    sprites, ok = [], []
    for n in names:
        sp = ref_sprite(os.path.join(ref_dir, n), side)
        sprites.append(sp)
    arr = np.stack(sprites)
    np.savez_compressed(cache_path, names=np.array(names), sprites=arr)
    print("built ref cache", cache_path, len(names))
    return dict(zip(names, sprites))


def get_src_to_gallery_refs(query_srcs):
    """Map query src name -> base source id for source-level eval (unused)."""
    return None


if __name__ == "__main__":
    pass
