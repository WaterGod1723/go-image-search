#!/usr/bin/env python3
"""Train a lightweight single-object detector with spatial soft-argmax.

Architecture: CNN backbone → 20×20 feature map → two 1×1 conv heads
  - heatmap head (center attention) → spatial softmax → soft-argmax (cx, cy)
  - size head (w, h at each cell) → sigmoid → probability-weighted average

Loss: heatmap MSE (Gaussian target) + L1(bbox) + CIoU(bbox).
Exports to ONNX (needs `onnx` pip package) at the end.
"""
import os
import math
import random
import glob

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
MODEL_DIR = os.path.join(HERE, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

IMG_SIZE = 320
GRID = 20              # feature-map side (320 / 2^4 = 20)
BATCH = 64
EPOCHS = 60
LR = 1e-3
WARMUP = 5
SEED = 42


# ── Dataset ──────────────────────────────────────────────────────────────

class IconDetectDataset(Dataset):
    def __init__(self, split, augment=False):
        self.imgs = sorted(glob.glob(os.path.join(DATA_DIR, "images", split, "*.png")))
        self.lbls = [p.replace("images", "labels").replace(".png", ".txt") for p in self.imgs]
        self.augment = augment

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        img = Image.open(self.imgs[i]).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
        arr = (np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0)
        with open(self.lbls[i]) as f:
            parts = f.read().strip().split()
        cx, cy, w, h = map(float, parts[1:5])
        box = np.array([cx, cy, w, h], dtype=np.float32)

        if self.augment:
            if random.random() < 0.5:           # horizontal flip
                arr = arr[:, :, ::-1].copy()
                box[0] = 1.0 - box[0]
            if random.random() < 0.5:           # vertical flip
                arr = arr[:, ::-1, :].copy()
                box[1] = 1.0 - box[1]
            if random.random() < 0.3:           # colour jitter
                j = np.random.uniform(0.85, 1.15, 3).reshape(3, 1, 1).astype(np.float32)
                arr = np.clip(arr * j, 0, 1)
        return torch.from_numpy(arr), torch.from_numpy(box)


# ── Model ─────────────────────────────────────────────────────────────────

class ConvBnAct(nn.Module):
    def __init__(self, cin, cout, k=3, s=2):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(cin, cout, k, stride=s, padding=k // 2, bias=False),
            nn.BatchNorm2d(cout), nn.ReLU6(inplace=True))

    def forward(self, x):
        return self.body(x)


class DetectNet(nn.Module):
    """CNN → 20×20 feature map → heatmap (soft-argmax center) + size head."""

    def __init__(self, grid=GRID):
        super().__init__()
        self.grid = grid
        self.backbone = nn.Sequential(
            ConvBnAct(3, 32, 3, 2),    # 160
            ConvBnAct(32, 64, 3, 2),   # 80
            ConvBnAct(64, 128, 3, 2),  # 40
            ConvBnAct(128, 256, 3, 2),  # 20
        )
        self.neck = nn.Sequential(
            ConvBnAct(256, 256, 3, 1),
            ConvBnAct(256, 256, 3, 1),
        )
        self.heatmap_head = nn.Conv2d(256, 1, 1)
        self.size_head = nn.Conv2d(256, 2, 1)

        # precompute normalised grid coordinates
        gy, gx = torch.meshgrid(
            torch.arange(grid, dtype=torch.float32),
            torch.arange(grid, dtype=torch.float32), indexing="ij")
        self.register_buffer("grid_x", gx / (grid - 1))  # (G, G)
        self.register_buffer("grid_y", gy / (grid - 1))

    def forward(self, x):
        feat = self.neck(self.backbone(x))          # B,256,G,G
        hm = self.heatmap_head(feat)[:, 0]           # B,G,G
        prob = torch.softmax(hm.flatten(1), dim=1)   # B,GG
        gx = self.grid_x.flatten()                   # GG
        gy = self.grid_y.flatten()
        cx = (prob * gx.unsqueeze(0)).sum(1)
        cy = (prob * gy.unsqueeze(0)).sum(1)
        sz = torch.sigmoid(self.size_head(feat))     # B,2,G,G
        szf = sz.flatten(2)                          # B,2,GG
        w = (prob.unsqueeze(1) * szf[:, 0:1]).sum(2)[:, 0]
        h = (prob.unsqueeze(1) * szf[:, 1:2]).sum(2)[:, 0]
        return torch.stack([cx, cy, w, h], dim=1)    # B,4


# ── Loss helpers ──────────────────────────────────────────────────────────

def gaussian_target(cx, cy, grid=GRID, sigma=1.5):
    """Build a 2-D Gaussian heatmap at (cx, cy) on the grid (batched)."""
    gx = cx * (grid - 1)
    gy = cy * (grid - 1)
    yy, xx = torch.meshgrid(
        torch.arange(grid, dtype=torch.float32, device=cx.device),
        torch.arange(grid, dtype=torch.float32, device=cx.device), indexing="ij")
    d2 = (xx[None] - gx[:, None, None]) ** 2 + (yy[None] - gy[:, None, None]) ** 2
    return torch.exp(-d2 / (2 * sigma ** 2))


def ciou_loss(pred, target):
    px1 = pred[:, 0] - pred[:, 2] / 2
    py1 = pred[:, 1] - pred[:, 3] / 2
    px2 = pred[:, 0] + pred[:, 2] / 2
    py2 = pred[:, 1] + pred[:, 3] / 2
    gx1 = target[:, 0] - target[:, 2] / 2
    gy1 = target[:, 1] - target[:, 3] / 2
    gx2 = target[:, 0] + target[:, 2] / 2
    gy2 = target[:, 1] + target[:, 3] / 2
    iw = (torch.min(px2, gx2) - torch.max(px1, gx1)).clamp(min=0)
    ih = (torch.min(py2, gy2) - torch.max(py1, gy1)).clamp(min=0)
    inter = iw * ih
    pw = (px2 - px1).clamp(min=1e-6)
    ph = (py2 - py1).clamp(min=1e-6)
    gw = (gx2 - gx1).clamp(min=1e-6)
    gh = (gy2 - gy1).clamp(min=1e-6)
    union = pw * ph + gw * gh - inter + 1e-6
    iou = inter / union
    ecx1 = torch.min(px1, gx1)
    ecy1 = torch.min(py1, gy1)
    ecx2 = torch.max(px2, gx2)
    ecy2 = torch.max(py2, gy2)
    ecw = (ecx2 - ecx1).clamp(min=1e-6)
    ech = (ecy2 - ecy1).clamp(min=1e-6)
    pcx, pcy = (px1 + px2) / 2, (py1 + py2) / 2
    gcx, gcy = (gx1 + gx2) / 2, (gy1 + gy2) / 2
    rho2 = (pcx - gcx) ** 2 + (pcy - gcy) ** 2
    diag2 = ecw ** 2 + ech ** 2 + 1e-6
    v = (4 / math.pi ** 2) * (torch.atan(gw / gh) - torch.atan(pw / ph)) ** 2
    with torch.no_grad():
        alpha = v / (1 - iou + v + 1e-6)
    return (1 - iou + rho2 / diag2 + alpha * v).mean()


def batch_iou(pred, target):
    px1 = pred[:, 0] - pred[:, 2] / 2
    py1 = pred[:, 1] - pred[:, 3] / 2
    px2 = pred[:, 0] + pred[:, 2] / 2
    py2 = pred[:, 1] + pred[:, 3] / 2
    gx1 = target[:, 0] - target[:, 2] / 2
    gy1 = target[:, 1] - target[:, 3] / 2
    gx2 = target[:, 0] + target[:, 2] / 2
    gy2 = target[:, 1] + target[:, 3] / 2
    iw = (torch.min(px2, gx2) - torch.max(px1, gx1)).clamp(min=0)
    ih = (torch.min(py2, gy2) - torch.max(py1, gy1)).clamp(min=0)
    inter = iw * ih
    union = (px2 - px1).clamp(min=0) * (py2 - py1).clamp(min=0) + \
            (gx2 - gx1).clamp(min=0) * (gy2 - gy1).clamp(min=0) - inter + 1e-6
    return inter / union


# ── Train ─────────────────────────────────────────────────────────────────

def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    train_ds = IconDetectDataset("train", augment=True)
    test_ds = IconDetectDataset("test")
    print(f"Train: {len(train_ds)}  Test: {len(test_ds)}", flush=True)
    train_dl = DataLoader(train_ds, BATCH, shuffle=True, num_workers=0, drop_last=True)
    test_dl = DataLoader(test_ds, 32, shuffle=False, num_workers=0)

    model = DetectNet().to(device)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    best_iou = 0.0
    for ep in range(1, EPOCHS + 1):
        # warmup
        if ep <= WARMUP:
            for g in opt.param_groups:
                g["lr"] = LR * ep / WARMUP

        model.train()
        tl1 = tciou = thm = n = 0.0
        for imgs, boxes in train_dl:
            imgs, boxes = imgs.to(device), boxes.to(device)
            feat = model.neck(model.backbone(imgs))
            hm = model.heatmap_head(feat)[:, 0]             # B,G,G
            prob = torch.softmax(hm.flatten(1), dim=1)     # B,GG
            gx = model.grid_x.flatten()
            gy = model.grid_y.flatten()
            cx = (prob * gx.unsqueeze(0)).sum(1)
            cy = (prob * gy.unsqueeze(0)).sum(1)
            sz = torch.sigmoid(model.size_head(feat))
            szf = sz.flatten(2)
            w = (prob.unsqueeze(1) * szf[:, 0:1]).sum(2)[:, 0]
            h = (prob.unsqueeze(1) * szf[:, 1:2]).sum(2)[:, 0]
            pred = torch.stack([cx, cy, w, h], dim=1)

            hm_target = gaussian_target(boxes[:, 0], boxes[:, 1])
            hm_loss = F.mse_loss(hm, hm_target)
            l1 = F.l1_loss(pred, boxes)
            ciou = ciou_loss(pred, boxes)
            loss = hm_loss + l1 + 2.0 * ciou

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            bs = imgs.size(0)
            tl1 += l1.item() * bs; tciou += ciou.item() * bs
            thm += hm_loss.item() * bs; n += bs
        if ep > WARMUP:
            sched.step()

        # evaluate
        model.eval()
        with torch.no_grad():
            ious = []
            for imgs, boxes in test_dl:
                ious.append(batch_iou(model(imgs.to(device)), boxes.to(device)).tolist())
        all_iou = [x for sub in ious for x in sub]
        miou = float(np.mean(all_iou))
        tag = ""
        if miou > best_iou:
            best_iou = miou
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, "detect_best.pt"))
            tag = " *"
        if ep % 5 == 0 or ep <= 3 or tag:
            print(f"Ep {ep:3d}  hm={thm/n:.4f}  L1={tl1/n:.4f}  ciou={tciou/n:.4f}  "
                  f"testIoU={miou:.4f}  >0.5={sum(1 for x in all_iou if x>0.5)}/{len(all_iou)}{tag}",
                  flush=True)

    # ── Export ONNX ───────────────────────────────────────────────────────
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "detect_best.pt"),
                                     map_location=device))
    model.eval()
    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
    onnx_path = os.path.join(MODEL_DIR, "detect.onnx")
    torch.onnx.export(model, dummy, onnx_path,
                      input_names=["image"], output_names=["bbox"],
                      opset_version=17,
                      dynamic_axes={"image": {0: "batch"}, "bbox": {0: "batch"}})
    print(f"\nONNX → {onnx_path} ({os.path.getsize(onnx_path)/1024:.0f} KB)", flush=True)
    print(f"Best test IoU: {best_iou:.4f}", flush=True)


if __name__ == "__main__":
    main()
