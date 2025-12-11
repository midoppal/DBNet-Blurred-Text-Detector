#!/usr/bin/env python3
import os, argparse, glob
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import timm
from shapely.geometry import Polygon
import pyclipper
import time


try:
    from concern.icdar2015_eval.detection.iou import DetectionIoUEvaluator
except Exception as e:
    raise RuntimeError(
        "Couldn't import DetectionIoUEvaluator from iou.py. "
        "Place your DBNet iou.py (the one you pasted) next to this script."
    )


def letterbox_pad_to_square(img_rgb, size):
    h, w = img_rgb.shape[:2]
    scale = min(size / h, size / w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    img_resized = cv2.resize(img_rgb, (nw, nh), interpolation=cv2.INTER_AREA)
    top = (size - nh) // 2
    left = (size - nw) // 2
    img_pad = np.zeros((size, size, 3), dtype=img_rgb.dtype)
    img_pad[top:top+nh, left:left+nw] = img_resized
    return img_pad, (top, left, nh, nw, h, w, scale)

def to_tensor_normalized(img_rgb, mean, std):
    x = img_rgb.astype(np.float32) / 255.0
    x = (x - mean) / std
    return torch.from_numpy(x).permute(2,0,1).unsqueeze(0)  # 1,3,H,W


def box_score_fast(bitmap, box):
    h, w = bitmap.shape[:2]
    pts = box.copy()
    xmin = int(np.clip(np.floor(pts[:,0].min()), 0, w-1))
    xmax = int(np.clip(np.ceil (pts[:,0].max()), 0, w-1))
    ymin = int(np.clip(np.floor(pts[:,1].min()), 0, h-1))
    ymax = int(np.clip(np.ceil (pts[:,1].max()), 0, h-1))
    mask = np.zeros((ymax-ymin+1, xmax-xmin+1), dtype=np.uint8)
    pts[:,0] = pts[:,0] - xmin
    pts[:,1] = pts[:,1] - ymin
    cv2.fillPoly(mask, pts.reshape(1,-1,2).astype(np.int32), 1)
    return cv2.mean(bitmap[ymin:ymax+1, xmin:xmax+1], mask)[0]

def unclip(box, unclip_ratio=1.5):
    poly = Polygon(box)
    if poly.area < 1e-6 or poly.length < 1e-6:
        return np.empty((0,4,2), dtype=np.float32)
    distance = poly.area * unclip_ratio / poly.length
    offset = pyclipper.PyclipperOffset()
    offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    expanded = offset.Execute(distance)
    if not expanded:
        return np.empty((0,4,2), dtype=np.float32)
    return np.array(expanded, dtype=np.float32)

def get_mini_boxes(contour):
    rect = cv2.minAreaRect(contour)
    pts  = cv2.boxPoints(rect)  
    pts  = sorted(list(pts), key=lambda x: x[0])
    idx1, idx2, idx3, idx4 = 0,1,2,3
    idx1, idx4 = (0,1) if pts[1][1] > pts[0][1] else (1,0)
    idx2, idx3 = (2,3) if pts[3][1] > pts[2][1] else (3,2)
    box = [pts[idx1], pts[idx2], pts[idx3], pts[idx4]]
    sside = min(rect[1])
    return box, sside

def boxes_from_prob(prob, thresh=0.35, box_thresh=0.55, min_size=3, unclip_ratio=0.0, erode_px=1):

    H, W = prob.shape


    bitmap = (prob > thresh).astype(np.uint8)
    if erode_px > 0:
        k = 2*erode_px + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        bitmap = cv2.erode(bitmap, kernel, iterations=1)

    contours, _ = cv2.findContours((bitmap*255).astype(np.uint8),
                                   cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    boxes, scores = [], []
    for cnt in contours[:5000]:
        pts, sside = get_mini_boxes(cnt)
        if sside < min_size:
            continue
        pts = np.array(pts, dtype=np.float32)

       
        score = box_score_fast(prob, pts.reshape(-1, 2))
        if score < box_thresh:
            continue

        
        if unclip_ratio and unclip_ratio > 0.01:
            ex = unclip(pts, unclip_ratio=unclip_ratio)
            if ex.shape[0] == 0:
                continue
            ex = max(ex, key=lambda p: Polygon(p).area)
            bb = np.array(ex, dtype=np.float32).reshape(-1, 1, 2)
            bb, sside2 = get_mini_boxes(bb)
            if sside2 < (min_size + 2):
                continue
            box = np.array(bb, dtype=np.float32)
        else:
            box = pts  

       
        box[:, 0] = np.clip(np.round(box[:, 0]), 0, W - 1)
        box[:, 1] = np.clip(np.round(box[:, 1]), 0, H - 1)

        boxes.append(box)
        scores.append(float(score))
    return boxes, scores


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
            poly = np.array(nums[:8], dtype=np.float32).reshape(4,2)
            ignore = any('###' in t for t in tail)  # don't-care
            polys.append(poly)
            ignores.append(ignore)
    return polys, ignores

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
        

def build_backbone_and_head(model_name, head_ckpt, device, img_size):
    model = timm.create_model(model_name, pretrained=True, img_size=img_size).to(device).eval()
    for p in model.parameters(): 
        p.requires_grad_(False)
    cdim = model.num_features
    patch = int(getattr(model, "patch_embed").patch_size[0])


    head = TextHead(cdim).to(device)

    sd = torch.load(head_ckpt, map_location="cpu")
    
    trained_img_size = None
    if isinstance(sd, dict) and "img_short" in sd:
        trained_img_size = int(sd["img_short"])
        
   
    if isinstance(sd, dict) and "head" in sd and isinstance(sd["head"], dict):
        state = sd["head"]
    elif isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        raw = sd["state_dict"]
        state = {k.replace("head.", ""): v for k, v in raw.items() if k.startswith("head.")}
        if not state:  
            state = raw
    elif isinstance(sd, dict):
        state = sd
    else:
        raise RuntimeError("Unsupported checkpoint format for head.")

    missing, unexpected = head.load_state_dict(state, strict=False)
    if missing or unexpected:
        print("[WARN] load_state_dict: missing:", missing, " unexpected:", unexpected)

    with torch.no_grad():
        p_means = {k: float(p.detach().abs().mean().cpu())
                   for k, p in head.named_parameters()}
        b_means = {k: float(b.detach().float().abs().mean().cpu())
                   for k, b in head.named_buffers()}  
        print("[INFO] head param |mean|:", {k: f"{m:.5f}" for k, m in p_means.items()})
        print("[INFO] head buffer |mean|:", {k: f"{m:.5f}" for k, m in b_means.items()})
        if all(m == 0.0 for m in p_means.values()):
            print("[ERROR] Head parameters look zero; check your checkpoint path/keys.")

    head.eval()


    def extract_feats(x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
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

            h_sq = int(round(N ** 0.5))
            if h_sq * h_sq == N:
                H = W = h_sq
            else:
                n_wo_cls = N - 1
                h_sq = int(round(n_wo_cls ** 0.5))
                if h_sq * h_sq != n_wo_cls:
                    raise RuntimeError(f"Cannot infer grid from token count N={N} (not k^2 or k^2+1).")
                tokens = tokens[:, 1:, :]             
                H = W = h_sq
    
            feat = tokens.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()  
            return feat

    return model, head, extract_feats, patch

def draw_and_save_boxes(img_bgr, boxes, scores, out_path, score_min=0.0):
    vis = img_bgr.copy()
    H, W = vis.shape[:2]

    thick = max(1, int(round(0.002 * (H + W))))
    font  = cv2.FONT_HERSHEY_SIMPLEX
    for b, s in zip(boxes, scores):
        if s < score_min:
            continue
        pts = b.astype(np.int32).reshape(-1, 1, 2)  
        cv2.polylines(vis, [pts], isClosed=True, color=(0, 255, 0), thickness=thick)
        
        x, y = int(b[0,0]), int(b[0,1])
        cv2.putText(vis, f"{s:.2f}", (x, max(0, y - 4)),
                    font, 0.5, (0, 255, 0), thickness=max(1, thick-1), lineType=cv2.LINE_AA)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, vis)

def save_prob_overlay(img_bgr: np.ndarray, prob: np.ndarray, out_path: str,
                      alpha: float = 0.45):

    prob8 = (np.clip(prob, 0, 1) * 255).astype(np.uint8)
    heat  = cv2.applyColorMap(prob8, cv2.COLORMAP_JET)    
    a = float(np.clip(alpha, 0.0, 1.0))
    overlay = cv2.addWeighted(heat, a, img_bgr, 1.0 - a, 0.0)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, overlay)
    return prob8  



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="dir with test images (e.g., datasets/icdar2015/test_images)")
    ap.add_argument("--gts",     required=True, help="dir with GT files (e.g., datasets/icdar2015/test_gts)")
    ap.add_argument("--head_ckpt", required=True, help="path to trained DINO head checkpoint")
    ap.add_argument("--model_name", default="vit_small_patch14_dinov2.lvd142m")
    ap.add_argument("--img_size",   type=int, default=700)

    ap.add_argument("--thresh",      type=float, default=0.3)
    ap.add_argument("--box_thresh",  type=float, default=0.2)
    ap.add_argument("--min_size",    type=int,   default=3)
    ap.add_argument("--unclip_ratio",type=float, default=0.0)
    
    ap.add_argument("--iou_thr",     type=float, default=0.3)
    ap.add_argument("--match_mode",  type=str,   default="pairwise", choices=["pairwise","one_to_one"])
    ap.add_argument("--viz_out", type=str, default="/workspace/DBNet-Blurred-Text-Detector/datasets/dino_text_boxes",
                help="If set, saves images with predicted boxes drawn here.")
    ap.add_argument("--viz_score_min", type=float, default=0.0,
                    help="Only draw boxes with score >= this.")
    ap.add_argument("--viz_prob_alpha", type=float, default=0.45,
                help="Blend weight for probability overlay (0..1, higher = more heatmap).")
    ap.add_argument("--viz_prob_raw", action="store_true",
                    help="Also dump the raw grayscale prob map (uint8) alongside the overlay.")
    ap.add_argument("--debug_every", type=int, default=100,
                help="Print debug stats every N images (0=off).")
    ap.add_argument("--dump_probs", type=str, default=None,
                    help="If set, saves grayscale probability maps here (uint8).")
    ap.add_argument("--relax", action="store_true",
                help="Temporarily relax postprocess: thresh=0.15, box_thresh=0.2, min_size=1, unclip=1.5")
    args = ap.parse_args()
    

    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, head, extract, patch = build_backbone_and_head(args.model_name, args.head_ckpt, device, args.img_size)
        
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[None,None,:]
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)[None,None,:]

    if args.relax:
        args.thresh = 0.15
        args.box_thresh = 0.20
        args.min_size = 1
        args.unclip_ratio = 1.5
        print(f"[RELAX] Using thresh={args.thresh}, box_thresh={args.box_thresh}, min_size={args.min_size}, unclip={args.unclip_ratio}")
        
    img_paths = sorted(glob.glob(os.path.join(args.images, "*.jpg")))
    # evaluator = DetectionIoUEvaluator(iou_constraint=args.iou_thr, area_precision_constraint=args.ignore_iof_thr)
    evaluator = DetectionIoUEvaluator(iou_constraint=args.iou_thr, area_precision_constraint=0.5)
    
    evaluator.match_mode = args.match_mode

    per_results = []

    fps_list = []
    warmup = True

    for p in img_paths:
        name = os.path.basename(p)                     
        gt_candidates = [
            os.path.join(args.gts, name + ".txt"),      
            os.path.join(args.gts, Path(name).stem + ".txt"),
            os.path.join(args.gts, "gt_" + Path(name).stem + ".txt"),
        ]
        gt_path = next((c for c in gt_candidates if os.path.exists(c)), None)

        img_bgr = cv2.imread(p)
        if img_bgr is None:
            print(f"[WARN] cannot read {p}, skipping.")
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


        sq, pad = letterbox_pad_to_square(img_rgb, args.img_size)
        x = to_tensor_normalized(sq, mean, std).to(device, non_blocking=True)

        if warmup:
            with torch.no_grad():
                _ = extract(x)
                _ = head(_)
            torch.cuda.synchronize()
            warmup = False
    
        
        torch.cuda.synchronize()
        t0 = time.time()


    
        with torch.no_grad():
            feat = extract(x)                          
            logits = head(feat)                         
            prob_lr = torch.sigmoid(logits)            
            prob_sq = F.interpolate(
                prob_lr, size=(args.img_size, args.img_size),
                mode="bilinear", align_corners=False
            )[0,0].cpu().numpy()                      


        torch.cuda.synchronize()
        t1 = time.time()
        
     
        infer_time = t1 - t0
        fps_list.append(1.0 / infer_time)
        
     
        if isinstance(prob_sq, torch.Tensor):
            prob_sq = prob_sq.cpu().numpy()


       
        top,left,nh,nw,h0,w0,scale = pad
        prob_content = prob_sq[top:top+nh, left:left+nw]
       
        interp = cv2.INTER_AREA if (prob_content.shape[0] > img_rgb.shape[0]) else cv2.INTER_NEAREST
        prob_full = cv2.resize(prob_content, (img_rgb.shape[1], img_rgb.shape[0]), interpolation=interp)

        pmin, pmax, pmean = float(prob_full.min()), float(prob_full.max()), float(prob_full.mean())
        if args.debug_every and (len(per_results) % args.debug_every == 0):
            print(f"[DBG] {name}: prob min/max/mean = {pmin:.4f}/{pmax:.4f}/{pmean:.4f}")
        
        if args.dump_probs:
            os.makedirs(args.dump_probs, exist_ok=True)
            cv2.imwrite(os.path.join(args.dump_probs, Path(name).stem + "_prob.png"),
                        (np.clip(prob_full, 0, 1) * 255).astype(np.uint8))

     
        if args.viz_out:
            os.makedirs(args.viz_out, exist_ok=True)
            base = os.path.splitext(name)[0]
            overlay_path = os.path.join(args.viz_out, f"{base}_prob_overlay.jpg")
            prob8 = save_prob_overlay(img_bgr, prob_full, overlay_path, alpha=args.viz_prob_alpha)
            if args.viz_prob_raw:
                cv2.imwrite(os.path.join(args.viz_out, f"{base}_prob.png"), prob8)

        
        boxes, scores = boxes_from_prob(prob_full, thresh=args.thresh, box_thresh=args.box_thresh,
                                        min_size=args.min_size, unclip_ratio=args.unclip_ratio)

        if args.debug_every and (len(per_results) % args.debug_every == 0):
            print(f"[DBG] {name}: num boxes = {len(boxes)} (thresh={args.thresh}, box_thresh={args.box_thresh})")


        if args.viz_out:
            os.makedirs(args.viz_out, exist_ok=True)
            out_path = os.path.join(args.viz_out, name)
            draw_and_save_boxes(img_bgr, boxes, scores, out_path, score_min=args.viz_score_min)



        preds = [{"points": b.tolist(), "text": "", "ignore": False} for b in boxes]

        
        gt_polys, gt_ignores = parse_icdar15_gt(gt_path) if gt_path is not None else ([], [])
        gts = [{"points": g.tolist(), "text": "", "ignore": bool(ign)} for g, ign in zip(gt_polys, gt_ignores)]

    
        R = evaluator.evaluate_image(gts, preds)
        per_results.append(R)

    
    metrics = evaluator.combine_results(per_results)

    if len(fps_list) > 0:
        avg_fps = sum(fps_list) / len(fps_list)
        med_fps = np.median(fps_list)
        print("==== SPEED ====")
        print(f"Average FPS: {avg_fps:.2f}")
        print(f"Median FPS:  {med_fps:.2f}")
        
    print("==== DINO-head (DBNet-style IoU eval) ====")
    print(f"Images: {len(per_results)}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F-Measure: {metrics['hmean']:.4f}")


if __name__ == "__main__":
    main()
