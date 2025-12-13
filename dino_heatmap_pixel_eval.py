#!/usr/bin/env python3

import os
import argparse
from pathlib import Path

import cv2
import numpy as np

def parse_icdar15_gt(txt_path: str):
    polys = []
    ignores = []
    if txt_path is None or not os.path.exists(txt_path):
        return polys, ignores
    with open(txt_path, "r", encoding="utf-8-sig") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            parts = [t.strip() for t in s.split(",")]
            nums, tail = [], []
            for t in parts:
                try:
                    nums.append(float(t))
                except:
                    tail.append(t)
            if len(nums) < 8:
                continue
            poly = np.array(nums[:8], dtype=np.float32).reshape(4, 2)
            ignore = any("###" in t for t in tail)
            polys.append(poly)
            ignores.append(ignore)
    return polys, ignores

def polygons_to_masks(polys, ignores, h, w):
    """
    Returns:
      mask_text   : uint8, 1 where text, 0 elsewhere
      mask_ignore : uint8, 1 where don't-care, 0 elsewhere
    """
    mask_text = np.zeros((h, w), dtype=np.uint8)
    mask_ignore = np.zeros((h, w), dtype=np.uint8)

    for poly, ign in zip(polys, ignores):
        if poly.size == 0:
            continue
        pts = poly.astype(np.int32).reshape(-1, 1, 2)
        if ign:
            cv2.fillPoly(mask_ignore, [pts], 1)
        else:
            cv2.fillPoly(mask_text, [pts], 1)

    mask_text[mask_ignore == 1] = 0
    return mask_text, mask_ignore

def load_heatmap_for_image(stem: str, heatmap_dir: str, exts):
    """
    Try several naming patterns:
      stem + '.png'
      stem + '_prob.png'
      stem + '.npy'
    Returns float32 HxW array in [0,1] or None if not found.
    """
    base_candidates = [
        stem,
        stem + "_prob",
    ]

    for base in base_candidates:
        for ext in exts:
            path = os.path.join(heatmap_dir, base + ext)
            if os.path.exists(path):
                if ext == ".npy":
                    hm = np.load(path)
                    return hm.astype(np.float32)
                else:
                    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        continue
                    hm = img.astype(np.float32) / 255.0
                    return hm
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True,
                    help="dir with test images (e.g., datasets/icdar2015/test_images)")
    ap.add_argument("--gts", required=True,
                    help="dir with GT files (e.g., datasets/icdar2015/test_gts)")
    ap.add_argument("--heatmaps", required=True,
                    help="dir with per-image heatmaps (PNG/NPY)")
    ap.add_argument("--img_ext", default=".jpg")
    ap.add_argument("--thresholds", type=str, default="0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95",
                    help="Comma-separated list of thresholds in [0,1].")
    ap.add_argument("--heatmap_exts", type=str, default=".png,.npy",
                    help="Comma-separated list of allowed heatmap extensions.")
    args = ap.parse_args()

    thresholds = [float(t) for t in args.thresholds.split(",") if t.strip()]
    hm_exts = [s.strip() for s in args.heatmap_exts.split(",") if s.strip()]

    img_paths = sorted(
        [p for p in Path(args.images).glob(f"*{args.img_ext}") if p.is_file()]
    )
    if not img_paths:
        raise RuntimeError(f"No images found in {args.images} with ext {args.img_ext}")

    sums = {thr: {"tp": 0, "fp": 0, "fn": 0} for thr in thresholds}

    num_used = 0

    for img_path in img_paths:
        name = img_path.name           
        stem = img_path.stem                 

        # Load image just to know H,W (should match heatmap)
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[WARN] Cannot read image {img_path}, skipping.")
            continue
        h, w = img.shape[:2]

        # Load heatmap
        hm = load_heatmap_for_image(stem, args.heatmaps, hm_exts)
        if hm is None:
            print(f"[WARN] No heatmap found for {stem}, skipping.")
            continue

        # If heatmap size doesn't match, resize
        if hm.shape[:2] != (h, w):
            hm = cv2.resize(hm, (w, h), interpolation=cv2.INTER_LINEAR)

        # Clip to [0,1]
        hm = np.clip(hm, 0.0, 1.0).astype(np.float32)

        # Load GT polygons
        gt_candidates = [
            os.path.join(args.gts, name + ".txt"),
            os.path.join(args.gts, stem + ".txt"),
            os.path.join(args.gts, "gt_" + stem + ".txt"),
        ]
        gt_path = next((c for c in gt_candidates if os.path.exists(c)), None)
        polys, ignores = parse_icdar15_gt(gt_path) if gt_path is not None else ([], [])

        # Build masks
        mask_text, mask_ignore = polygons_to_masks(polys, ignores, h, w)
        valid = (mask_ignore == 0)

        if mask_text.sum() == 0:
            continue

        num_used += 1

        #Flatten for faster operations, but only where valid
        gt_text = (mask_text == 1)
        gt_bg = ~gt_text
      
        #Apply thresholds on full hm
        for thr in thresholds:
            pred = hm >= thr

            tp = np.logical_and(pred, gt_text & valid).sum()
            fp = np.logical_and(pred, gt_bg & valid).sum()
            fn = np.logical_and(~pred, gt_text & valid).sum()

            sums[thr]["tp"] += int(tp)
            sums[thr]["fp"] += int(fp)
            sums[thr]["fn"] += int(fn)

    if num_used == 0:
        print("[ERROR] No images with valid GT were used.")
        return

    print(f"Used {num_used} images with non-empty GT text regions.")
    print("==== PIXEL-LEVEL METRICS (micro-averaged over dataset) ====")
    print("{:>8s}  {:>10s}  {:>10s}  {:>10s}".format("thr", "Precision", "Recall", "F1"))

    best_thr = None
    best_f1 = -1.0
    best_stats = None

    for thr in thresholds:
        tp = sums[thr]["tp"]
        fp = sums[thr]["fp"]
        fn = sums[thr]["fn"]

        prec = tp / (tp + fp + 1e-8)
        rec  = tp / (tp + fn + 1e-8)
        f1   = 2 * prec * rec / (prec + rec + 1e-8)

        print("{:8.3f}  {:10.4f}  {:10.4f}  {:10.4f}".format(thr, prec, rec, f1))

        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr
            best_stats = (prec, rec)

    print("\n==== BEST F1 OVER THRESHOLDS ====")
    print(f"Best threshold: {best_thr:.3f}")
    print(f"Precision: {best_stats[0]:.4f}")
    print(f"Recall:    {best_stats[1]:.4f}")
    print(f"F1:        {best_f1:.4f}")


if __name__ == "__main__":
    main()
