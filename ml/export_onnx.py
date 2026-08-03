#!/usr/bin/env python3
"""Export the trained DetectNet (soft-argmax) to ONNX and evaluate."""
import os, glob, math
import numpy as np
from PIL import Image
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
MODEL_DIR = os.path.join(HERE, "models")
IMG_SIZE = 320
GRID = 20

class ConvBnAct(nn.Module):
    def __init__(self, cin, cout, k=3, s=2):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(cin, cout, k, stride=s, padding=k//2, bias=False),
            nn.BatchNorm2d(cout), nn.ReLU6(inplace=True))
    def forward(self, x): return self.body(x)

class DetectNet(nn.Module):
    def __init__(self, grid=GRID):
        super().__init__()
        self.grid = grid
        self.backbone = nn.Sequential(
            ConvBnAct(3,32,3,2), ConvBnAct(32,64,3,2),
            ConvBnAct(64,128,3,2), ConvBnAct(128,256,3,2))
        self.neck = nn.Sequential(ConvBnAct(256,256,3,1), ConvBnAct(256,256,3,1))
        self.heatmap_head = nn.Conv2d(256,1,1)
        self.size_head = nn.Conv2d(256,2,1)
        gy,gx = torch.meshgrid(torch.arange(grid,dtype=torch.float32),
                               torch.arange(grid,dtype=torch.float32),indexing="ij")
        self.register_buffer("grid_x", gx/(grid-1))
        self.register_buffer("grid_y", gy/(grid-1))
    def forward(self, x):
        feat = self.neck(self.backbone(x))
        hm = self.heatmap_head(feat)[:,0]
        prob = torch.softmax(hm.flatten(1), dim=1)
        cx = (prob * self.grid_x.flatten().unsqueeze(0)).sum(1)
        cy = (prob * self.grid_y.flatten().unsqueeze(0)).sum(1)
        sz = torch.sigmoid(self.size_head(feat)).flatten(2)
        w = (prob.unsqueeze(1) * sz[:,0:1]).sum(2)[:,0]
        h = (prob.unsqueeze(1) * sz[:,1:2]).sum(2)[:,0]
        return torch.stack([cx,cy,w,h], dim=1)

class DS(Dataset):
    def __init__(self, split):
        self.imgs = sorted(glob.glob(os.path.join(DATA_DIR,"images",split,"*.png")))
        self.lbls = [p.replace("images","labels").replace(".png",".txt") for p in self.imgs]
    def __len__(self): return len(self.imgs)
    def __getitem__(self, i):
        img = Image.open(self.imgs[i]).convert("RGB").resize((IMG_SIZE,IMG_SIZE),Image.LANCZOS)
        arr = np.asarray(img,dtype=np.float32).transpose(2,0,1)/255.0
        with open(self.lbls[i]) as f: p=f.read().strip().split()
        return torch.from_numpy(arr), torch.tensor([float(p[1]),float(p[2]),float(p[3]),float(p[4])])

def biou(pred, target):
    px1,py1,px2,py2 = pred[:,0]-pred[:,2]/2,pred[:,1]-pred[:,3]/2,pred[:,0]+pred[:,2]/2,pred[:,1]+pred[:,3]/2
    gx1,gy1,gx2,gy2 = target[:,0]-target[:,2]/2,target[:,1]-target[:,3]/2,target[:,0]+target[:,2]/2,target[:,1]+target[:,3]/2
    iw=(torch.min(px2,gx2)-torch.max(px1,gx1)).clamp(min=0)
    ih=(torch.min(py2,gy2)-torch.max(py1,gy1)).clamp(min=0)
    u=(px2-px1).clamp(min=0)*(py2-py1).clamp(min=0)+(gx2-gx1).clamp(min=0)*(gy2-gy1).clamp(min=0)-iw*ih+1e-6
    return iw*ih/u

def main():
    model = DetectNet()
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR,"detect_best.pt"), map_location="cpu"))
    model.eval()
    dl = DataLoader(DS("test"), 32, shuffle=False, num_workers=0)
    ious=[]
    with torch.no_grad():
        for imgs,boxes in dl:
            ious.append(biou(model(imgs), boxes).tolist())
    a=[x for s in ious for x in s]
    print(f"Test IoU: mean={np.mean(a):.4f}  >0.5={sum(1 for x in a if x>0.5)}/{len(a)}  >0.7={sum(1 for x in a if x>0.7)}/{len(a)}")
    onnx_path=os.path.join(MODEL_DIR,"detect.onnx")
    torch.onnx.export(model, torch.randn(1,3,IMG_SIZE,IMG_SIZE), onnx_path,
                      input_names=["image"], output_names=["bbox"],
                      opset_version=17,
                      dynamic_axes={"image":{0:"batch"},"bbox":{0:"batch"}})
    print(f"ONNX → {onnx_path} ({os.path.getsize(onnx_path)/1024:.0f} KB)")

if __name__=="__main__": main()
