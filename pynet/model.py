"""Small embedding CNN for icon retrieval (rotation-augmented classifier)."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SE(nn.Module):
    """Squeeze-and-Excitation channel attention (very few params)."""

    def __init__(self, c, r=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(c, c // r), nn.ReLU(inplace=True),
            nn.Linear(c // r, c), nn.Sigmoid())

    def forward(self, x):
        b, c, h, w = x.shape
        s = x.mean(dim=(2, 3))
        s = self.fc(s).view(b, c, 1, 1)
        return x * s


class CBAM(nn.Module):
    """Channel attention (SE) + spatial attention. Still tiny and Go-portable."""

    def __init__(self, c, r=8, k=7):
        super().__init__()
        self.se = SE(c, r)
        self.sp = nn.Conv2d(2, 1, kernel_size=k, padding=k // 2)

    def forward(self, x):
        x = self.se(x)
        avg = x.mean(dim=1, keepdim=True)
        mx, _ = x.max(dim=1, keepdim=True)
        att = torch.sigmoid(self.sp(torch.cat([avg, mx], dim=1)))
        return x * att


class EmbedCNN(nn.Module):
    """RGBA NxN -> L2-normalized embedding.

    Small enough to port to pure-Go. Classification head used only for
    training (softmax over source classes); the head is dropped at inference.
    """

    def __init__(self, in_ch=4, emb_dim=64, n_classes=None, ch=(16, 32, 48, 64),
                 attn="none"):
        super().__init__()
        self.ch = ch
        self.emb_dim = emb_dim
        self.attn = attn
        blocks = []
        c_in = in_ch
        for c in ch:
            blocks += [
                nn.Conv2d(c_in, c, 3, padding=1),
                nn.BatchNorm2d(c),
                nn.ReLU(inplace=True),
                nn.Conv2d(c, c, 3, padding=1),
                nn.BatchNorm2d(c),
                nn.ReLU(inplace=True),
            ]
            if attn == "se":
                blocks.append(SE(c))
            elif attn == "cbam":
                blocks.append(CBAM(c))
            blocks.append(nn.MaxPool2d(2))
            c_in = c
        self.body = nn.Sequential(*blocks)
        self.embed = nn.Linear(ch[-1], emb_dim)
        self.head = nn.Linear(emb_dim, n_classes) if n_classes else None

    def forward(self, x, with_emb=False):
        h = self.body(x)
        h = h.mean(dim=(2, 3))  # global avg pool
        e = self.embed(h)
        e = F.normalize(e, dim=1)
        if self.head is None or not with_emb:
            return e
        return e, self.head(e)

    def embed_only(self, x):
        return self.forward(x)

    def param_count(self):
        return sum(p.numel() for p in self.parameters())


class ArcMarginHead(nn.Module):
    """Additive angular margin softmax (ArcFace) classification head.

    Class centroids live on the unit hypersphere; logits =
    s * cos(theta + m) for the true class, s * cos(theta) otherwise.
    Produces sharper, more separable embeddings than plain softmax.
    """

    def __init__(self, emb_dim, n_classes, s=20.0, m=0.30):
        super().__init__()
        self.s = s
        self.m = m
        self.W = nn.Parameter(torch.randn(emb_dim, n_classes) * 0.01)

    def forward(self, emb, labels):
        Wn = F.normalize(self.W, dim=0)
        cos_theta = emb @ Wn
        cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
        theta = torch.acos(cos_theta)
        target_logit = torch.cos(theta + self.m)
        onehot = torch.zeros_like(cos_theta)
        onehot.scatter_(1, labels.view(-1, 1), 1.0)
        logits = torch.where(onehot > 0, target_logit, cos_theta) * self.s
        return logits


class RotationAug:
    """On-the-fly random rotation of sprites for rotation invariance."""

    def __init__(self, p=0.9):
        self.p = p

    def __call__(self, sp):
        # sp: B x 4 x N x N float tensor in [0,1]
        if self.p <= 0:
            return sp
        angles = (torch.rand(sp.shape[0]) * 360) * (torch.rand(sp.shape[0]) < self.p)
        # vectorized rotation via affine grid
        theta = torch.zeros(sp.shape[0], 2, 3)
        for i, a in enumerate(angles):
            if a == 0:
                theta[i, 0, 0], theta[i, 1, 1] = 1, 1
                continue
            rad = math.radians(a.item())
            c, s = math.cos(rad), math.sin(rad)
            theta[i, 0, 0], theta[i, 0, 1], theta[i, 0, 2] = c, -s, 0
            theta[i, 1, 0], theta[i, 1, 1], theta[i, 1, 2] = s, c, 0
        grid = F.affine_grid(theta, sp.shape, align_corners=False)
        out = F.grid_sample(sp, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        return out
