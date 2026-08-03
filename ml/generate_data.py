#!/usr/bin/env python3
"""Synthetic data generator for single-object detection.

For each icon in test_pngs_trimmed, compose scenes:
  - solid random background color
  - icon scaled to a random factor in [0.3, 1.0]
  - a short random CN/EN text snippet drawn around the icon at
    font-size 0.3-0.6x the (scaled) icon size, as interference
  - ground-truth bounding box of the placed icon (YOLO-style normalized label)

Icons are split into train/test sets (no icon appears in both) to test
generalization to unseen object shapes.
"""
import os
import random
import glob

from PIL import Image, ImageDraw, ImageFont
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(HERE, "..", "test_pngs_trimmed")
OUT_DIR = os.path.join(HERE, "data")

CANVAS = 320          # canvas side length (px)
ICON_MAX = 200        # normalized longest side before random scale (px)

TRAIN_FRAC = 0.8      # fraction of icons used for training
SAMPLES_TRAIN = 60    # samples per train icon
SAMPLES_TEST = 25     # samples per test icon
SEED = 42

CN_WORDS = [
    "费用", "报销", "审批", "付款", "收入", "支出", "统计", "报表",
    "设置", "首页", "消息", "通知", "搜索", "编辑", "删除", "添加",
    "确认", "取消", "保存", "提交", "订单", "商品", "购物", "支付",
    "退款", "优惠券", "会员", "积分", "签到", "任务", "日程", "备忘",
    "收藏", "分享", "评论", "点赞", "关注", "客服", "帮助", "关于",
]
EN_WORDS = [
    "Home", "Search", "Edit", "Delete", "Add", "Save", "Submit", "Order",
    "Pay", "Refund", "Member", "Score", "Task", "Message", "Setting",
    "Profile", "Cart", "List", "Detail", "Status", "Total", "Amount",
    "Date", "Time", "Name", "Phone", "Email", "Address", "Note", "Tag",
]

FONT_PATHS_CN = [
    "/System/Library/Fonts/STHeiti Light.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
]
FONT_PATHS_EN = [
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/HelveticaNeue.ttc",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
]


def first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None


def load_font_paths():
    cn = first_existing(FONT_PATHS_CN) or first_existing(FONT_PATHS_EN)
    en = first_existing(FONT_PATHS_EN) or cn
    return cn, en


def icon_nontransparent_bbox(img):
    """Return (l, t, r, b) bounding box of non-transparent pixels."""
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    alpha = np.asarray(img)[:, :, 3]
    ys, xs = np.where(alpha > 16)
    if len(xs) == 0:
        return None
    l, r = int(xs.min()), int(xs.max()) + 1
    t, b = int(ys.min()), int(ys.max()) + 1
    return (l, t, r, b)


def normalize_icon(img):
    """Crop to non-transparent bbox, resize longest side to ICON_MAX."""
    bbox = icon_nontransparent_bbox(img)
    if bbox is None:
        return None
    l, t, r, b = bbox
    img = img.crop((l, t, r, b))
    w, h = img.size
    scale = ICON_MAX / float(max(w, h))
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    return img.resize((nw, nh), Image.LANCZOS)


def random_bg_color():
    """A random but not-too-extreme background RGB."""
    h = random.random()
    s = random.uniform(0.3, 0.85)
    v = random.uniform(0.45, 0.95)
    import colorsys
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return (int(r * 255), int(g * 255), int(b * 255))


def contrasting_color(bg):
    """Pick a text color that contrasts with the background."""
    r, g, b = bg
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    if lum > 140:
        return (random.randint(0, 80), random.randint(0, 80), random.randint(0, 80))
    return (random.randint(160, 255), random.randint(160, 255), random.randint(160, 255))


def random_text():
    """A short mixed CN/EN snippet of 2-6 characters/words."""
    parts = []
    n = random.randint(2, 5)
    for _ in range(n):
        if random.random() < 0.6:
            parts.append(random.choice(CN_WORDS))
        else:
            parts.append(random.choice(EN_WORDS))
    return "".join(parts) if random.random() < 0.3 else " ".join(parts)


def draw_text_around(draw, icon_bbox, canvas_size, text, font, color):
    """Draw text on a random side of the icon, fully outside its bbox when possible."""
    x0, y0, x1, y1 = icon_bbox
    W = H = canvas_size

    # measure text
    try:
        bb = draw.textbbox((0, 0), text, font=font)
        tw, th = bb[2] - bb[0], bb[3] - bb[1]
    except Exception:
        tw, th = font.getbbox(text)[2:]
    tw, th = max(1, tw), max(1, th)

    pad = random.randint(4, 12)
    sides = ["top", "bottom", "left", "right"]
    random.shuffle(sides)

    for side in sides:
        if side == "top" and y0 - th - pad >= 0:
            tx = random.randint(0, max(0, W - tw))
            ty = y0 - th - pad
            draw.text((tx, ty), text, fill=color, font=font)
            return
        if side == "bottom" and y1 + th + pad <= H:
            tx = random.randint(0, max(0, W - tw))
            ty = y1 + pad
            draw.text((tx, ty), text, fill=color, font=font)
            return
        if side == "left" and x0 - tw - pad >= 0:
            ty = random.randint(0, max(0, H - th))
            tx = x0 - tw - pad
            draw.text((tx, ty), text, fill=color, font=font)
            return
        if side == "right" and x1 + tw + pad <= W:
            ty = random.randint(0, max(0, H - th))
            tx = x1 + pad
            draw.text((tx, ty), text, fill=color, font=font)
            return

    # fallback: place text overlapping near a corner (still valid interference)
    tx = random.randint(0, max(0, W - tw))
    ty = random.randint(0, max(0, H - th))
    draw.text((tx, ty), text, fill=color, font=font)


def generate_one(icon_img, font_cn, font_en, idx, split):
    """Generate one synthetic sample; return (PIL.Image, label_str)."""
    bg = random_bg_color()
    canvas = Image.new("RGB", (CANVAS, CANVAS), bg)

    icon = normalize_icon(icon_img)
    if icon is None:
        return None, None
    if icon.mode != "RGBA":
        icon = icon.convert("RGBA")

    scale = random.uniform(0.3, 1.0)
    w, h = icon.size
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    icon_s = icon.resize((nw, nh), Image.LANCZOS)

    x = random.randint(0, CANVAS - nw)
    y = random.randint(0, CANVAS - nh)
    canvas.paste(icon_s, (x, y), icon_s)

    bbox = (x, y, x + nw, y + nh)

    # text interference: font size 0.3-0.6 * icon size
    icon_size = max(nw, nh)
    font_size = max(8, int(random.uniform(0.3, 0.6) * icon_size))
    text = random_text()
    color = contrasting_color(bg)
    font_path = random.choice([font_cn, font_en])
    try:
        font = ImageFont.truetype(font_path, font_size)
    except Exception:
        font = ImageFont.load_default()

    draw = ImageDraw.Draw(canvas)
    draw_text_around(draw, bbox, CANVAS, text, font, color)

    # YOLO-style normalized label: class cx cy w h
    cx = (x + nw / 2.0) / CANVAS
    cy = (y + nh / 2.0) / CANVAS
    w_n = nw / CANVAS
    h_n = nh / CANVAS
    label = f"0 {cx:.6f} {cy:.6f} {w_n:.6f} {h_n:.6f}"

    name = f"{split}_{idx:06d}"
    img_path = os.path.join(OUT_DIR, "images", split, name + ".png")
    lbl_path = os.path.join(OUT_DIR, "labels", split, name + ".txt")
    canvas.save(img_path)
    with open(lbl_path, "w") as f:
        f.write(label + "\n")
    return canvas, label


def main():
    random.seed(SEED)
    np.random.seed(SEED)

    font_cn, font_en = load_font_paths()
    print(f"CN font: {font_cn}")
    print(f"EN font: {font_en}")

    icons = sorted(glob.glob(os.path.join(SRC_DIR, "*.png")))
    print(f"Found {len(icons)} source icons")
    if not icons:
        raise SystemExit(f"No icons found in {SRC_DIR}")

    random.shuffle(icons)
    n_train = int(len(icons) * TRAIN_FRAC)
    train_icons = icons[:n_train]
    test_icons = icons[n_train:]
    print(f"Train icons: {len(train_icons)}  Test icons: {len(test_icons)}")

    idx = 0
    for icon_path in train_icons:
        img = Image.open(icon_path).convert("RGBA")
        for _ in range(SAMPLES_TRAIN):
            generate_one(img, font_cn, font_en, idx, "train")
            idx += 1
    n_train_samples = idx

    for icon_path in test_icons:
        img = Image.open(icon_path).convert("RGBA")
        for _ in range(SAMPLES_TEST):
            generate_one(img, font_cn, font_en, idx, "test")
            idx += 1
    n_test_samples = idx - n_train_samples

    print(f"Generated {n_train_samples} train + {n_test_samples} test samples")
    print(f"Output: {OUT_DIR}")


if __name__ == "__main__":
    main()
