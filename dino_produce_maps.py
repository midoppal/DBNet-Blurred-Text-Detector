#!/usr/bin/env python3
##### 1 #####
import os, cv2, math, torch, timm
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch.nn.functional as F
import torch.nn as nn
import time


INPUT_DIR  = r"/workspace/DBNet-Blurred-Text-Detector/datasets/icdar2015/test_images"
OUT_DIR    = r"/workspace/DBNet-Blurred-Text-Detector/datasets/dino_maps"
CKPT_PATH  = r"/workspace/DBNet-Blurred-Text-Detector/fine_tuned_models/dino_4_2.ckpt"
SAVE_NPY   = True   


os.makedirs(OUT_DIR, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"


class TextHead(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, 128, kernel_size=3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=2, dilation=2, bias=False)
        self.bn2   = nn.BatchNorm2d(128)
        self.out   = nn.Conv2d(128, 1, kernel_size=1)
        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity="relu")
        nn.init.kaiming_normal_(self.conv2.weight, nonlinearity="relu")
        nn.init.zeros_(self.out.weight)
        nn.init.constant_(self.out.bias, -4.0)

    def forward(self, f):
        x = F.gelu(self.bn1(self.conv1(f)))
        x = F.gelu(self.bn2(self.conv2(x)))
        return self.out(x)


def letterbox_square(img_bgr, target):
    h, w = img_bgr.shape[:2]
    scale = target / max(h, w)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    top  = (target - nh) // 2
    left = (target - nw) // 2
    bottom = target - nh - top
    right  = target - nw - left
    out = cv2.copyMakeBorder(
        resized, top, bottom, left, right,
        borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )
    return out, (left, top, right, bottom), (h, w), (nh, nw)

def unletterbox_to_original(prob_sq, pads, orig_hw, resized_hw):
    left, top, right, bottom = pads
    nh, nw = resized_hw
    cropped = prob_sq[top:top+nh, left:left+nw]
    H, W = orig_hw  
    return cv2.resize(cropped, (W, H), interpolation=cv2.INTER_CUBIC)


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)[None, None, :]
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)[None, None, :]

def to_tensor_bgr_letterbox(img_bgr, model_img_sz):
    img_sq, pads, orig_hw, resized_hw = letterbox_square(img_bgr, model_img_sz)
    img_rgb = cv2.cvtColor(img_sq, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    x = (img_rgb - IMAGENET_MEAN) / IMAGENET_STD
    x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(device)
    return x, pads, orig_hw, resized_hw


def load_backbone_and_head(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model_name = ckpt.get("backbone", None)
    if model_name is None:
        raise ValueError(f"'backbone' not found in {ckpt_path}.")

    
    model = timm.create_model(model_name, pretrained=True).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    embed_dim = model.num_features

    if hasattr(model, "patch_embed") and hasattr(model.patch_embed, "img_size"):
        model_img_sz = int(model.patch_embed.img_size[0])
    else:
        model_img_sz = int(model.pretrained_cfg["input_size"][1])

    head = TextHead(embed_dim).to(device).eval()
    if "head" not in ckpt:
        raise ValueError(f"'head' state_dict not found in {ckpt_path}.")
    missing, unexpected = head.load_state_dict(ckpt["head"], strict=False)
    if missing or unexpected:
        print("[WARN] load_state_dict: missing:", missing, " unexpected:", unexpected)

    with torch.no_grad():
        means = {k: float(p.detach().abs().mean().cpu()) for k, p in head.named_parameters()}
        print("[INFO] head param |mean|:", {k: f"{m:.5f}" for k, m in means.items()})

    return model, head, model_img_sz

@torch.no_grad()
def extract_feats(model, x):
    """
    Returns feature map as (B,C,H',W').
    Works with timm ViT / DINO / DINOv2 forward_features outputs.
    """
    out = model.forward_features(x)

    if isinstance(out, torch.Tensor):
        tokens = out                        
    elif isinstance(out, dict):
        for key in ("x_norm_patchtokens", "x_norm", "x", "last_feat", "feat"):
            if key in out and isinstance(out[key], torch.Tensor):
                tokens = out[key]
                break
        else:
            tokens = next(v for v in out.values() if isinstance(v, torch.Tensor))
    else:
        raise RuntimeError("Unexpected forward_features() return type")

    B, N, C = tokens.shape

    if hasattr(model, "patch_embed") and hasattr(model.patch_embed, "grid_size"):
        gh, gw = model.patch_embed.grid_size
        n_patch = int(gh * gw)
        if N == n_patch + 1:
            tokens = tokens[:, 1:, :]
            N = n_patch
        elif N > n_patch + 1:
            tokens = tokens[:, 1:1+n_patch, :]
            N = n_patch

    h = int(round(math.sqrt(N)))
    w = h
    assert h * w == N, f"Cannot form square grid from N={N} tokens."

    fmap = tokens.reshape(B, h, w, C).permute(0, 3, 1, 2).contiguous()  
    return fmap

def save_overlay(out_path, img_bgr, prob):
    prob_u8 = np.clip(prob * 255.0, 0, 255).astype(np.uint8)
    heat = cv2.applyColorMap(prob_u8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_bgr, 0.5, heat, 0.5, 0.0)
    cv2.imwrite(out_path, overlay)

def save_gray_png(out_path, prob):
    prob_u8 = np.clip(prob * 255.0, 0, 255).astype(np.uint8)
    cv2.imwrite(out_path, prob_u8)

def run():
    model, head, MODEL_IMG_SZ = load_backbone_and_head(CKPT_PATH)
    print(f"[INFO] Using backbone='{type(model).__name__}' ({MODEL_IMG_SZ}px), head in_ch={model.num_features}")

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    files = [p for p in Path(INPUT_DIR).iterdir() if p.suffix.lower() in exts]
    
    total_model_time = 0.0  
    total_full_time = 0.0 
    n_images = 0
    # ----------------------------------
    
    for p in tqdm(files, ncols=100):
        full_start = time.time() 

        img_bgr = cv2.imread(str(p))
        if img_bgr is None:
            print(f"[WARN] unreadable: {p.name}")
            continue

        x, pads, orig_hw, resized_hw = to_tensor_bgr_letterbox(img_bgr, MODEL_IMG_SZ)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_start = time.time()

        with torch.no_grad():
            f = extract_feats(model, x)          
            logits = head(f)                   
            prob_lr = torch.sigmoid(logits)    
            prob_sq = F.interpolate(
                prob_lr,
                size=(MODEL_IMG_SZ, MODEL_IMG_SZ),
                mode="bilinear",
                align_corners=False
            )[0, 0].cpu().numpy()
            prob = unletterbox_to_original(prob_sq, pads, orig_hw, resized_hw)  

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_end = time.time()


        prob = np.clip(prob, 0.0, 1.0).astype(np.float32)

        out_prob_png = os.path.join(OUT_DIR, p.stem + "_prob.png")
        out_overlay  = os.path.join(OUT_DIR, p.stem + "_overlay.png")
        save_gray_png(out_prob_png, prob)
        save_overlay(out_overlay, img_bgr, prob)

        if SAVE_NPY:
            np.save(os.path.join(OUT_DIR, p.stem + "_prob.npy"), prob)

        full_end = time.time()

        total_model_time += (model_end - model_start)
        total_full_time  += (full_end - full_start)
        n_images += 1

    if n_images > 0 and total_model_time > 0:
        ms_per_img_model = (total_model_time / n_images) * 1000.0
        fps_model = n_images / total_model_time
        print(f"[DINO] Model-only: {ms_per_img_model:.3f} ms/img, FPS = {fps_model:.2f} over {n_images} images")

    if n_images > 0 and total_full_time > 0:
        ms_per_img_full = (total_full_time / n_images) * 1000.0
        fps_full = n_images / total_full_time
        print(f"[DINO] Full pipeline (load + model + save): {ms_per_img_full:.3f} ms/img, FPS = {fps_full:.2f} over {n_images} images")



if __name__ == "__main__":
    run()
