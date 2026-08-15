"""Sprite extraction and normalization.

Query images are rendered icons on a colored canvas with interference
(text/lines/curves) placed strictly OUTSIDE the sprite's screen rect, so the
sprite is the largest foreground blob after background removal. We must NOT
use manifest bg_hex at inference time, so this pipeline estimates the
background from the image border pixels (same rule for train and eval so
there is no shortcut leakage).
"""
import numpy as np
from PIL import Image

def load_rgb(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)

def estimate_bg(img):
    h, w = img.shape[:2]
    b = 8
    ring = np.concatenate([
        img[:b].reshape(-1, 3), img[-b:].reshape(-1, 3),
        img[:, :b].reshape(-1, 3), img[:, -b:].reshape(-1, 3),
    ], axis=0)
    return np.median(ring, axis=0)

def adaptive_dist_threshold(dist, bins=256):
    """Go adaptiveFGMask: bg peak then first significant valley."""
    bg_peak = int(np.argmax(dist))
    for i in range(bg_peak + 1, bins - 1):
        if dist[i] < dist[i - 1] and dist[i] < dist[i + 1] and dist[i] < dist[bg_peak] / 8:
            return i / bins
    floor = 30 / bins
    low = (bg_peak + 12) / bins
    if low > floor:
        floor = low
    return floor


def extract_sprite(img, bg=None, close_k=3):
    """Return (sprite RGBA on transparent bg, ok_flag).

    Port of Go extractQuery/adaptiveFGMask: distance-to-bg histogram, valley
    threshold, morphological close (dilate+erode x close_k), keep largest
    interior component + touching fragments.
    """
    if img.shape[0] == 0 or img.shape[1] == 0:
        return None, False
    if bg is None:
        bg = estimate_bg(img)
    dist = np.sqrt(((img.astype(np.float32) - bg) ** 2).sum(axis=2)) / 255.0
    # histogram of distances (256 bins)
    hist, _ = np.histogram(dist.clip(0, 1.0), bins=256, range=(0.0, 1.0))
    thr = adaptive_dist_threshold(hist)
    fg = dist > thr
    if fg.sum() < 20:
        return None, False
    # morphological closing (dilate then erode) to heal thin-stroke gaps
    if close_k >= 1:
        from scipy import ndimage
        struct = np.ones((2 * close_k + 1, 2 * close_k + 1))
        fg = ndimage.binary_closing(fg, structure=struct)
    # connected components
    labels = _cc(fg)
    if labels is None:
        return None, False
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    if sizes.max() < 20:
        return None, False
    main = sizes.argmax()
    keep = labels == main
    # add touching components (fragments)
    for l in range(1, len(sizes)):
        if l == main:
            continue
        if _touches(keep, labels == l):
            keep = keep | (labels == l)
    ys, xs = np.nonzero(keep)
    if len(ys) == 0:
        return None, False
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    sprite = np.zeros((y1 - y0 + 1, x1 - x0 + 1, 4), dtype=np.uint8)
    crop = img[y0:y1 + 1, x0:x1 + 1]
    sprite[..., :3] = crop
    sprite[..., 3] = np.where(keep[y0:y1 + 1, x0:x1 + 1], 255, 0)
    return sprite, True

def _cc(fg):
    from scipy import ndimage
    try:
        labels, n = ndimage.label(fg)
        if n == 0:
            return None
        return labels
    except Exception:
        return None

def _touches(a, b):
    # bbox intersection test is a cheap proxy for adjacency
    ys, xs = np.nonzero(a)
    y2, x2 = np.nonzero(b)
    if len(ys) == 0 or len(y2) == 0:
        return False
    return not (ys.max() < y2.min() or y2.max() < ys.min() or
                xs.max() < x2.min() or x2.max() < xs.min())

def trim_transparent(img):
    """Trim alpha==0 border from a sprite; return (rgb_crop, mask_crop)."""
    if img.shape[2] == 3:
        rgb = img
        mask = np.ones(img.shape[:2], dtype=np.uint8) * 255
    else:
        rgb = img[..., :3]
        mask = img[..., 3]
    ys, xs = np.nonzero(mask > 10)
    if len(ys) == 0:
        return img, mask
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    return rgb[y0:y1 + 1, x0:x1 + 1], mask[y0:y1 + 1, x0:x1 + 1]

def to_square(rgb, mask, side):
    """Letterbox sprite (keep aspect) into side x side RGBA."""
    h, w = rgb.shape[:2]
    s = max(h, w)
    if s == 0:
        return np.zeros((side, side, 4), dtype=np.uint8)
    pad_rgb = np.zeros((s, s, 3), dtype=np.uint8)
    pad_mask = np.zeros((s, s), dtype=np.uint8)
    y0 = (s - h) // 2
    x0 = (s - w) // 2
    pad_rgb[y0:y0 + h, x0:x0 + w] = rgb
    pad_mask[y0:y0 + h, x0:x0 + w] = mask
    im = Image.fromarray(pad_rgb)
    mk = Image.fromarray(pad_mask)
    im = im.resize((side, side), Image.BILINEAR)
    mk = mk.resize((side, side), Image.BILINEAR)
    out = np.zeros((side, side, 4), dtype=np.uint8)
    out[..., :3] = np.asarray(im)
    out[..., 3] = np.asarray(mk)
    # alpha-mask rgb: zero out pixels below alpha threshold so ref/query share
    # the same transparent-region color (refs: black bg, queries: colored bg
    # would otherwise leak in and confuse the CNN)
    a = out[..., 3].astype(np.float32)
    out[..., :3] = (out[..., :3].astype(np.float32) * (a / 255.0)[..., None]).astype(np.uint8)
    return out

def query_sprite(path, side=48):
    """Full query -> normalized sprite tensor input (N x N x 4 RGBA)."""
    img = load_rgb(path)
    sprite, ok = extract_sprite(img)
    if not ok:
        return None
    rgb, mask = trim_transparent(sprite)
    return to_square(rgb, mask, side)

def ref_sprite(path, side=48):
    """Reference icon -> normalized sprite (N x N x 4 RGBA)."""
    img = np.asarray(Image.open(path).convert("RGBA"), dtype=np.uint8)
    rgb, mask = trim_transparent(img)
    return to_square(rgb, mask, side)

def sprite_to_tensor(sp):
    """RGBA (N x N x 4 uint8) -> float tensor (4 x N x N) normalized [0,1]."""
    t = sp.astype(np.float32) / 255.0
    return t.transpose(2, 0, 1)
