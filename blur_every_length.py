#!/usr/bin/env python3
import os
import math
import glob
import numpy as np
import cv2
import argparse


def _srgb_to_linear(img):
    img = img.astype(np.float32, copy=False)
    img = np.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0)
    img = np.clip(img, 0.0, 1.0)
    a = 0.055
    return np.where(img <= 0.04045, img/12.92, ((img + a) / (1 + a)) ** 2.4)

def _linear_to_srgb(img):
    img = img.astype(np.float32, copy=False)
    img = np.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0)
    img = np.clip(img, 0.0, 1.0)
    a = 0.055
    return np.where(img <= 0.0031308, img*12.92, (1 + a) * (img ** (1/2.4)) - a)

def _profile_weights(n: int, profile: str) -> np.ndarray:
    n = max(1, int(n))
    t = np.linspace(0, 1, n, dtype=np.float32)
    if profile == "box":
        w = np.ones_like(t, dtype=np.float32)
    elif profile == "gaussian":
        u = (t * 2.0 - 1.0)
        w = np.exp(-0.5 * (u/0.6)**2).astype(np.float32)
    elif profile == "cosine":
        w = (np.cos(t * math.pi) * 0.5 + 0.5).astype(np.float32)
    else:
        raise ValueError(f"Unknown profile: {profile}")
    s = float(w.sum())
    if s <= 0:
        w = np.ones_like(t, dtype=np.float32) / len(t)
    else:
        w /= s
    return w

def _one_sided_horizontal_kernel(length: int, smear_dir: str,
                                 profile: str="gaussian",
                                 antialias: bool=True) -> np.ndarray:
    L = max(1, int(length))
    k = 2 * L + 1
    kernel = np.zeros((k, k), dtype=np.float32)
    c = L

    w = _profile_weights(L + 1, profile)

    if smear_dir == "right":
        for i, wi in enumerate(w):
            x = c + i
            if 0 <= x < k:
                kernel[c, x] += wi
    else:  # "left"
        for i, wi in enumerate(w):
            x = c - i
            if 0 <= x < k:
                kernel[c, x] += wi

    if antialias and k >= 5:
        kernel = cv2.GaussianBlur(kernel, (3, 3), 0.5,
                                  borderType=cv2.BORDER_REPLICATE)

    kernel_sum = kernel.sum()
    if kernel_sum > 0:
        kernel /= kernel_sum
    return kernel

def _apply_motion_blur(img_bgr, kernel, gamma_aware=True):
    if img_bgr.dtype.kind in 'ui':
        img = img_bgr.astype(np.float32) / 255.0
        scale_back = 255.0
    else:
        img = img_bgr.astype(np.float32)
        scale_back = 255.0 if img.max() > 1.5 else 1.0
        if scale_back == 255.0:
            img /= 255.0

    if gamma_aware:
        img_lin = _srgb_to_linear(img)
        out_lin = np.zeros_like(img_lin)
        for c in range(3):
            out_lin[..., c] = cv2.filter2D(img_lin[..., c], -1, kernel,
                                           borderType=cv2.BORDER_REPLICATE)
        out = _linear_to_srgb(np.clip(out_lin, 0.0, 1.0))
    else:
        out = np.zeros_like(img)
        for c in range(3):
            out[..., c] = cv2.filter2D(img[..., c], -1, kernel,
                                       borderType=cv2.BORDER_REPLICATE)

    return np.clip(out * scale_back, 0, 255).astype(np.float32)


def collect_images(input_folder):
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(input_folder, ext)))
    return sorted(files)

def apply_blur_levels_to_folder(input_folder, output_root,
                                min_len=1, max_len=30,
                                profile="gaussian",
                                gamma_aware=True,
                                antialias=True):
    os.makedirs(output_root, exist_ok=True)
    image_paths = collect_images(input_folder)

    if not image_paths:
        print(f"[WARN] No images found in '{input_folder}'.")
        return

    print(f"[INFO] Found {len(image_paths)} images.")

    # kernel cache per (L, dir)
    kernel_cache = {}

    blur_lengths = [1, 5, 10, 15, 20, 25, 30]

    for L in blur_lengths:
        level_dir = os.path.join(output_root, f"blur_{L:02d}")
        os.makedirs(level_dir, exist_ok=True)

        print(f"[INFO] Processing blur length {L} → {level_dir}")

        for img_path in image_paths:
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img is None:
                print(f"[WARN] Failed to read {img_path}")
                continue

        
            # 50/50 random left or right motion blur
            smear_dir = "left" if np.random.rand() < 0.5 else "right"
        

            key = (L, smear_dir)
            if key not in kernel_cache:
                kernel_cache[key] = _one_sided_horizontal_kernel(
                    length=L,
                    smear_dir=smear_dir,
                    profile=profile,
                    antialias=antialias,
                )
            kernel = kernel_cache[key]

            blurred = _apply_motion_blur(img, kernel, gamma_aware)
            blurred_u8 = np.clip(blurred, 0, 255).astype(np.uint8)

            filename = os.path.basename(img_path)
            out_path = os.path.join(level_dir, filename)
            cv2.imwrite(out_path, blurred_u8)

    print(f"[DONE] All blur levels saved to {output_root}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--min_len", type=int, default=1)
    parser.add_argument("--max_len", type=int, default=30)
    parser.add_argument("--profile", type=str, default="gaussian",
                        choices=["gaussian", "cosine", "box"])
    parser.add_argument("--no_gamma", action="store_true")
    parser.add_argument("--no_antialias", action="store_true")

    args = parser.parse_args()

    apply_blur_levels_to_folder(
        args.input_folder,
        args.output_root,
        args.min_len,
        args.max_len,
        args.profile,
        not args.no_gamma,
        not args.no_antialias
    )

if __name__ == "__main__":
    main()
