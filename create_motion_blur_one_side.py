
#!/usr/bin/env python3
"""
create_motion_blur_split_directions.py

- Walks all images under IN_DIR.
- For each image and each length L in [0..30], writes a blurred copy to:
    datasets/icdar2015/blurred_test_images_length_{L}_left/<...>
    datasets/icdar2015/blurred_test_images_length_{L}_right/<...>
- Applies one-sided Gaussian blur (default) simulating real video motion.
- Each image randomly gets either left or right smear.
- Progress updates printed per file.
"""

import os, math
import cv2
import numpy as np
from pathlib import Path

IN_DIR = "datasets/icdar2015/original_test_images"
PROFILE = "gaussian"
GAMMA_AWARE = True
ANTIALIAS = True
MAX_LENGTH = 30

def srgb_to_linear(img: np.ndarray) -> np.ndarray:
    a = 0.055
    return np.where(img <= 0.04045, img / 12.92, ((img + a) / (1 + a)) ** 2.4)

def linear_to_srgb(img: np.ndarray) -> np.ndarray:
    a = 0.055
    return np.where(img <= 0.0031308, img * 12.92, (1 + a) * (img ** (1 / 2.4)) - a)

def _profile_weights(n: int, profile: str) -> np.ndarray:
    n = max(1, int(n))
    t = np.linspace(0, 1, n, dtype=np.float32)
    if profile == "box":
        w = np.ones_like(t)
    elif profile == "gaussian":
        u = (t * 2.0 - 1.0)
        w = np.exp(-0.5 * (u / 0.6) ** 2)
    elif profile == "cosine":
        w = (np.cos(t * math.pi) * 0.5 + 0.5)
    else:
        raise ValueError(f"Unknown profile: {profile}")
    w /= max(w.sum(), 1e-8)
    return w

def make_one_sided_horizontal_kernel(length: int, smear_dir: str, profile: str = "gaussian", antialias: bool = True) -> np.ndarray:
    L = max(1, int(length))
    k = 2 * L + 1
    kernel = np.zeros((k, k), dtype=np.float32)
    c = L
    w = _profile_weights(L + 1, profile)
    if smear_dir == "right":
        for i, wi in enumerate(w):
            x = c + i
            if x < k:
                kernel[c, x] += wi
    elif smear_dir == "left":
        for i, wi in enumerate(w):
            x = c - i
            if x >= 0:
                kernel[c, x] += wi
    else:
        raise ValueError("smear_dir must be 'left' or 'right'")
    if antialias and k >= 5:
        kernel = cv2.GaussianBlur(kernel, (3, 3), 0.5, borderType=cv2.BORDER_REPLICATE)
    kernel /= max(kernel.sum(), 1e-8)
    return kernel

def apply_motion_blur(img_bgr: np.ndarray, kernel: np.ndarray, gamma_aware: bool = True) -> np.ndarray:
    if gamma_aware:
        img = img_bgr.astype(np.float32) / 255.0
        img = srgb_to_linear(img)
        out = np.empty_like(img)
        for c in range(3):
            out[..., c] = cv2.filter2D(img[..., c], -1, kernel, borderType=cv2.BORDER_REPLICATE)
        out = np.clip(out, 0, 1)
        out = linear_to_srgb(out)
        return (out * 255.0 + 0.5).astype(np.uint8)
    else:
        out = np.empty_like(img_bgr)
        for c in range(3):
            out[..., c] = cv2.filter2D(img_bgr[..., c], -1, kernel, borderType=cv2.BORDER_REPLICATE)
        return out

def is_image_file(p: Path) -> bool:
    return p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def walk_images(root: Path):
    return [p for p in root.rglob("*") if p.is_file() and is_image_file(p)]

def process():
    inp = Path(IN_DIR)
    inp.mkdir(parents=True, exist_ok=True)
    lengths = list(range(0, MAX_LENGTH + 1))
    files = walk_images(inp)
    total_files = len(files)

    for idx, p in enumerate(files, 1):
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            print(f"[WARN] Could not read {p}")
            continue

        rel = p.relative_to(inp)

        for L in lengths:
            if L == 0:
                out_root = Path(f"datasets/icdar2015/blurred_test_images_length_0")
                out_dir = out_root / rel.parent
                out_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out_dir / p.name), img)
                continue

            smear_dir = "left" if np.random.rand() < 0.5 else "right"
            out_root = Path(f"datasets/icdar2015/blurred_test_images_length_{L}_{smear_dir}")
            out_dir = out_root / rel.parent
            out_dir.mkdir(parents=True, exist_ok=True)

            kernel = make_one_sided_horizontal_kernel(
                length=L,
                smear_dir=smear_dir,
                profile=PROFILE,
                antialias=ANTIALIAS
            )
            out = apply_motion_blur(img, kernel, gamma_aware=GAMMA_AWARE)
            cv2.imwrite(str(out_dir / p.name), out)

        print(f"Processed [{idx}/{total_files}] - {rel}")

    print(f"[DONE] Wrote {total_files} images across {len(lengths)} lengths.")

if __name__ == "__main__":
    process()
