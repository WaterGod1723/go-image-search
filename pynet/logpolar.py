"""Log-polar transform of sprites for exact rotation invariance.

Rotation of the input becomes a circular shift along the angular axis of the
log-polar grid. Taking the FFT-magnitude (or global pooling) along that axis
makes the representation rotation-invariant BY CONSTRUCTION — no learning
required. The CNN then only needs to be discriminative.
"""
import math
import numpy as np
from PIL import Image


def build_logpolar(sp_rgb, sp_mask, R=24, T=32, inner=0.10):
    """sp: RGBA HxW. Returns (lp_rgb R x T x 3, lp_mask R x T).

    r in [inner*Rmax, Rmax] log-sampled; theta in [0, 2pi) with T bins.
    Rotation by dtheta => circular shift along T.
    """
    h, w = sp_rgb.shape[:2]
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    rmax = math.hypot(cy, cx)  # max distance to center (corner)
    rmin = rmax * inner
    log_lo, log_hi = math.log(rmin + 1e-9), math.log(rmax + 1e-9)
    lp_rgb = np.zeros((R, T, 3), dtype=np.float32)
    lp_mask = np.zeros((R, T), dtype=np.float32)
    for rr in range(R):
        # log radius in [rmin, rmax]
        t = rr / (R - 1) if R > 1 else 0.5
        rad = math.exp(log_lo + t * (log_hi - log_lo))
        for tt in range(T):
            th = 2 * math.pi * tt / T
            x = cx + rad * math.cos(th)
            y = cy + rad * math.sin(th)
            # bilinear sample
            x0, y0 = int(math.floor(x)), int(math.floor(y))
            fx, fy = x - x0, y - y0
            ok = True
            for dy in (0, 1):
                for dx in (0, 1):
                    yy, xx = y0 + dy, x0 + dx
                    if yy < 0 or yy >= h or xx < 0 or xx >= w:
                        ok = False
            if not ok:
                continue
            acc_rgb = np.zeros(3, dtype=np.float32)
            acc_w = 0.0
            for dy in (0, 1):
                for dx in (0, 1):
                    wgt = (fx if dx == 1 else 1 - fx) * (fy if dy == 1 else 1 - fy)
                    yy, xx = y0 + dy, x0 + dx
                    acc_rgb += wgt * sp_rgb[yy, xx].astype(np.float32)
                    acc_w += wgt * sp_mask[yy, xx]
            lp_rgb[rr, tt] = acc_rgb
            lp_mask[rr, tt] = acc_w
    return lp_rgb, lp_mask


def sprite_to_logpolar(sp, R=24, T=32):
    """RGBA sprite -> (lp_rgb R x T x 3, lp_mask R x T)."""
    rgb = sp[..., :3].astype(np.float32)
    mask = sp[..., 3].astype(np.float32)
    return build_logpolar(rgb, mask, R, T)


def canonicalize_shift(lp):
    """Shift lp so the highest-energy column is at theta=0 (aligns rotation)."""
    pass  # (optional; not needed with FFT pooling)


def fft_pool(lp_rgb, lp_mask):
    """Per-ring |FFT| along angular axis => rotation-invariant feature R x (1+T/2)*3.
    Keeps color by pooling rgb separately; mask channel included.
    """
    R, T = lp_mask.shape
    feats = []
    for rr in range(R):
        m = lp_mask[rr]
        row = [m]
        for c in range(3):
            f = np.fft.rfft(lp_rgb[rr, :, c] * (m > 0.5))
            row.append(np.abs(f))
        feats.append(np.concatenate(row))
    return np.concatenate(feats)  # R x (1 + 3*(T/2+1))


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from preprocess import ref_sprite, query_sprite
    import json

    # verify rotation invariance: rotate a sprite, compare FFT-pool features
    d = json.load(open("train_set_vit/manifest.json", encoding="utf-8"))
    sp = ref_sprite("scraped_icons/train/" + d[0]["src"])
    f0 = fft_pool(*sprite_to_logpolar(sp))
    # rotate by arbitrary angle
    im = Image.fromarray(sp).rotate(37.3, resample=Image.BILINEAR)
    sp2 = np.asarray(im)
    f1 = fft_pool(*sprite_to_logpolar(sp2))
    f0 = f0 / (np.linalg.norm(f0) + 1e-8)
    f1 = f1 / (np.linalg.norm(f1) + 1e-8)
    print("fft-pool sim 0deg vs 37.3deg:", round(float(f0 @ f1), 4))
    # different icon for contrast
    sp3 = ref_sprite("scraped_icons/train/" + d[100]["src"])
    f2 = fft_pool(*sprite_to_logpolar(sp3))
    f2 = f2 / (np.linalg.norm(f2) + 1e-8)
    print("fft-pool sim icon0 vs icon100:", round(float(f0 @ f2), 4))
