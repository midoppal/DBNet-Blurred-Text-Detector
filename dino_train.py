import os, glob, math, cv2, json
import numpy as np
from pathlib import Path
from typing import List, Tuple
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from data.processes.random_motion_blur import RandomMotionBlur 

import timm  


IMG_DIR = r"/workspace/DBNet-Blurred-Text-Detector/datasets/icdar2015/train_images"
GT_DIR  = r"/workspace/DBNet-Blurred-Text-Detector/datasets/icdar2015/train_gts"
CKPT    = r"/workspace/DBNet-Blurred-Text-Detector/fine_tuned_models/dino_text_head.ckpt"

MODEL_NAME = "vit_small_patch14_dinov2.lvd142m" 
IMG_SHORT  = 1024                      
IMG_SIZE   = 700                            
BATCH_SIZE = 8
EPOCHS     = 40          
LR         = 3e-4        
weight_decay = 0.01
W_BCE, W_DICE = 0.5, 2.0  
W_TVERSKY = 1.0       
L_TV = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# --------------------------------------------

def list_images(img_dir):
    exts = (".jpg",".jpeg",".png",".bmp",".tif",".tiff")
    return sorted([p for p in Path(img_dir).iterdir() if p.suffix.lower() in exts])

def parse_icdar15_gt(txt_path: str):
    polys = []
    if txt_path is None: 
        return polys
    with open(txt_path, "r", encoding="utf-8-sig") as f:
        for line in f:
            s = line.strip()
            if not s: 
                continue
            parts = [t.strip() for t in s.split(",")]
            nums = []
            for t in parts:
                try:
                    nums.append(float(t))
                    if len(nums) == 8:
                        break
                except:
             
                    pass
            if len(nums) < 8:
                continue
            poly = np.array(nums[:8], dtype=np.float32).reshape(4, 2)
            polys.append(poly)
    return polys


def letterbox_pad_to_square(img_rgb, size):
    h, w = img_rgb.shape[:2]
    scale = min(size / h, size / w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    img_resized = cv2.resize(img_rgb, (nw, nh), interpolation=cv2.INTER_AREA)
    top = (size - nh) // 2
    left = (size - nw) // 2
    img_pad = np.zeros((size, size, 3), dtype=img_rgb.dtype)
    img_pad[top:top+nh, left:left+nw] = img_resized
    pad_info = (top, left, nh, nw, h, w, scale)
    return img_pad, pad_info

def warp_polys_to_padded(polys, pad_info):
    top, left, nh, nw, h, w, scale = pad_info
    out = []
    for poly in polys:
        xy = poly.copy()
        xy[:, 0] = xy[:, 0] * scale + left
        xy[:, 1] = xy[:, 1] * scale + top
        out.append(xy)
    return out

def valid_region_mask(size, pad_info):
    top, left, nh, nw, *_ = pad_info
    m = np.zeros((size, size), dtype=np.float32)
    m[top:top+nh, left:left+nw] = 1.0
    return m
    
def resize_keep_aspect(img, short_side):
    h, w = img.shape[:2]
    if min(h, w) == short_side:
        return img, (h, w), 1.0, 1.0
    if h < w:
        newh, neww = short_side, int(round(w * (short_side / h)))
        ry = newh / h; rx = neww / w
    else:
        neww, newh = short_side, int(round(h * (short_side / w)))
        ry = newh / h; rx = neww / w
    img2 = cv2.resize(img, (neww, newh), interpolation=cv2.INTER_AREA)
    return img2, (h, w), ry, rx

def rasterize_mask(polys: List[np.ndarray], size_hw: Tuple[int,int]) -> np.ndarray:
    H, W = size_hw
    mask = np.zeros((H, W), dtype=np.uint8)
    for poly in polys:
        cv2.fillPoly(mask, [poly.astype(np.int32)], 1)
    return mask


class IcdarTextDataset(Dataset):
    def __init__(self, img_dir, gt_dir, short_side=768,
                 use_blur_aug=True):
        self.img_paths = list_images(img_dir)
        self.gt_dir = Path(gt_dir)
        self.short_side = short_side
        self.use_blur_aug = use_blur_aug

        self.mean = np.array([0.485,0.456,0.406], dtype=np.float32)[None,None,:]
        self.std  = np.array([0.229,0.224,0.225], dtype=np.float32)[None,None,:]

        self.blur = RandomMotionBlur(
            p_clean=0.10, p_mild=0.20, p_med=0.30, p_heavy=0.40,
            mild=(1,10), med=(11,19), heavy=(20,30),
            profile="gaussian", gamma_aware=True, antialias=True, seed=None
        )
        self.p_hflip_for_right = 0.20 

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        p = self.img_paths[idx]
        img_bgr = cv2.imread(str(p))
        if img_bgr is None:
            raise FileNotFoundError(p)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
        stem = p.stem        
        name = p.name             
        gt_path = self.gt_dir / (p.stem + ".txt")
        candidates = [
            self.gt_dir / f"{name}.txt",   
            self.gt_dir / f"{stem}.txt", 
            self.gt_dir / f"gt_{stem}.txt" 
        ]
        gt_path = next((c for c in candidates if c.exists()), None)
        polys = parse_icdar15_gt(str(gt_path)) if gt_path is not None else []
        
     
        img_sq, pad_info = letterbox_pad_to_square(img_rgb, IMG_SIZE) 
        polys_sq = warp_polys_to_padded(polys, pad_info)               
    
  
      
        do_flip = (np.random.rand() < self.p_hflip_for_right)
        if do_flip:
            img_sq = np.ascontiguousarray(np.flip(img_sq, axis=1))
            if polys_sq:
            
                polys_sq = [np.stack([IMG_SIZE - poly[:, 0], poly[:, 1]], axis=1) for poly in polys_sq]
    
        if self.use_blur_aug:
            data = {"image": img_sq.astype(np.float32, copy=False)}
            data = self.blur(data)
            img_sq = data["image"]
    
        mask_sq = rasterize_mask(polys_sq, (IMG_SIZE, IMG_SIZE)).astype(np.float32)
    
        valid_sq = valid_region_mask(IMG_SIZE, pad_info).astype(np.float32)
    
        x = img_sq.astype(np.float32) / 255.0
        x = (x - self.mean) / self.std
        x = torch.from_numpy(x).permute(2, 0, 1)     
        y = torch.from_numpy(mask_sq)[None, ...]        
        v = torch.from_numpy(valid_sq)[None, ...]      
    
        meta = {"path": str(p), "pad_info": pad_info, "flipped": do_flip}
        return x, y, v, meta


def dice_loss(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits)
    num = 2 * (probs*targets).sum(dim=(2,3)) + eps
    den = (probs*probs).sum(dim=(2,3)) + (targets*targets).sum(dim=(2,3)) + eps
    return 1 - (num/den).mean()

def build_dino_backbone(name: str, device: str, img_size: int):
    import timm

    model = timm.create_model(
        name,
        pretrained=True,
        img_size=img_size,    
        num_classes=0,        
    ).to(device).eval()

    for p in model.parameters():
        p.requires_grad_(False)

 
    patch = model.patch_embed.patch_size
    patch = patch[0] if isinstance(patch, (tuple, list)) else int(patch)

  
    assert img_size % patch == 0, f"IMG_SIZE ({img_size}) must be a multiple of patch size ({patch})."

    embed_dim = model.num_features

    def extract_feats(x):
        with torch.no_grad():
            out = model.forward_features(x)

            tokens = out['x'] if isinstance(out, dict) and 'x' in out else out
         
            if tokens.dim() == 3 and tokens.shape[1] == (img_size // patch) * (img_size // patch) + 1:
                tokens = tokens[:, 1:, :]
            B, N, C = tokens.shape
            h = w = img_size // patch
            expected = h * w
            assert N == expected, f"Token grid mismatch: N={N}, expected={expected} for img_size={img_size}, patch={patch}"
            return tokens.reshape(B, h, w, C).permute(0, 3, 1, 2).contiguous()

    return model, embed_dim, patch, extract_feats

class TextHead(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, 128, kernel_size=3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=2, dilation=2, bias=False)
        self.bn2   = nn.BatchNorm2d(128)
        self.out   = nn.Conv2d(128, 1, kernel_size=1)

      
        for m in [self.conv1, self.conv2]:
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        nn.init.zeros_(self.out.weight)
        nn.init.constant_(self.out.bias, -4.0) 

    def forward(self, f):
        x = F.gelu(self.bn1(self.conv1(f)))
        x = F.gelu(self.bn2(self.conv2(x)))
        return self.out(x)

def tversky_index(probs, targets, mask=None, alpha=0.3, beta=0.7, eps=1e-6):
   
    if mask is None:
        mask = torch.ones_like(targets)
    probs = probs * mask
    targets = targets * mask
    TP = (probs * targets).sum(dim=(2,3))
    FP = (probs * (1 - targets)).sum(dim=(2,3))
    FN = ((1 - probs) * targets).sum(dim=(2,3))
    TI = (TP + eps) / (TP + alpha*FP + beta*FN + eps)
    return TI.mean()

def focal_tversky_loss(probs, targets, mask=None, alpha=0.3, beta=0.7, gamma=1.33, eps=1e-6):
    TI = tversky_index(probs, targets, mask=mask, alpha=alpha, beta=beta, eps=eps)
    return (1.0 - TI).pow(gamma)

    
def train():
    device = DEVICE
    ds = IcdarTextDataset(IMG_DIR, GT_DIR, short_side=IMG_SHORT)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)

    dino, cdim, patch, extract = build_dino_backbone(MODEL_NAME, device, IMG_SIZE)
    head = TextHead(cdim).to(device)
    
    opt = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=weight_decay)

    from torch.optim.lr_scheduler import CosineAnnealingLR
    scheduler = CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-5)
    
    WARMUP_EPOCHS = 2
    base_lr = LR

    bce = nn.BCEWithLogitsLoss()

    for epoch in range(1, EPOCHS+1):
        if epoch <= WARMUP_EPOCHS:
            warmup_lr = base_lr * epoch / max(1, WARMUP_EPOCHS)
            for pg in opt.param_groups:
                pg["lr"] = warmup_lr
        head.train()
        running = 0.0
        for x, y, v, meta in tqdm(dl,
        desc=f"Epoch {epoch}/{EPOCHS}",
        total=len(dl),
        ncols=100,
        leave=False,):
            x = x.to(device, non_blocking=True)   
            y = y.to(device, non_blocking=True)     
            v = v.to(device, non_blocking=True)     
        
            f = extract(x)                   

            
         
            y_ds = F.interpolate(y, size=f.shape[-2:], mode="bilinear", align_corners=False).clamp_(0, 1)
            v_ds = F.interpolate(v, size=f.shape[-2:], mode="nearest")
            

            y_dil = F.max_pool2d(y_ds, kernel_size=3, stride=1, padding=1)  
            
           
            logits = head(f)
            probs  = torch.sigmoid(logits)
            

            pos = ((y_dil * v_ds).sum() / (v_ds.sum().clamp(min=1.0))).detach()
            
            pos_weight = ((1.0 - pos) / (pos + 1e-6)).clamp(1.0, 10.0)
            
            bce_raw = F.binary_cross_entropy_with_logits(logits, y_dil, reduction='none')
            
            y_far = y_ds
            for _ in range(3):  
                y_far = F.max_pool2d(y_far, kernel_size=3, stride=1, padding=1)
            
            far_bg = ((1.0 - y_far) * v_ds).clamp(0, 1)  
            far_bg_weight = 2.0                          
            
            neg_weights = 1.0 + (far_bg_weight - 1.0) * far_bg
            weights = torch.where(y_dil > 0.5, pos_weight, neg_weights)
            
            bce = (bce_raw * weights * v_ds).sum() / ((weights * v_ds).sum().clamp(min=1.0))
            
            ftversky = focal_tversky_loss(
                probs, y_dil, mask=v_ds,
                alpha=0.6,  
                beta=0.4,  
                gamma=1.1,   
            )
            
            tv_h = (probs[:, :, 1:, :] - probs[:, :, :-1, :]).abs().mean()
            tv_w = (probs[:, :, :, 1:] - probs[:, :, :, :-1]).abs().mean()
            tv   = tv_h + tv_w
            
            
            loss = W_BCE * bce + W_TVERSKY * ftversky + L_TV * tv
      
        

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), max_norm=1.0)
            opt.step()

            running += loss.item()

        print(f"[Epoch {epoch}] loss={running/len(dl):.4f}")
        if epoch > WARMUP_EPOCHS:
            scheduler.step()

    torch.save({"head": head.state_dict(), "backbone": MODEL_NAME, "img_short": IMG_SHORT}, CKPT)
    print(f"Saved: {CKPT}")

if __name__ == "__main__":
    train()