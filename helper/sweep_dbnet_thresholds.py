
import os, sys, csv, time, re, json, math, itertools, subprocess, datetime
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List

import numpy as np
import matplotlib.pyplot as plt


PY_EXE      = sys.executable  
EVAL_SCRIPT = "eval.py"
YAML_PATH   = r"experiments/seg_detector/ic15_resnet50_deform_thre.yaml"
RESUME_PATH = r"./workspace/SegDetectorModel-L1BalanceCELoss/model/300_plus150_plus300lr003_1_15"

THRESH_VALUES     = np.round(np.linspace(0.10, 0.70, 13), 2)
BOX_THRESH_VALUES = np.round(np.linspace(0.10, 0.90, 17), 2)
EVAL_CWD = Path(".")


ROOT_OUT = Path("./sweeps")


def mk_run_id(t: float, b: float) -> str:
    return f"t{t:.2f}_b{b:.2f}".replace(".", "p")


def parse_metrics_from_text(txt: str):
    
    p_match = re.search(r'precision\s*:\s*([0-9.]+)', txt, flags=re.IGNORECASE)
    r_match = re.search(r'recall\s*:\s*([0-9.]+)', txt, flags=re.IGNORECASE)
    f_match = re.search(r'fmeasure\s*:\s*([0-9.]+)', txt, flags=re.IGNORECASE)

    precision = float(p_match.group(1)) if p_match else None
    recall    = float(r_match.group(1)) if r_match else None
    f1        = float(f_match.group(1)) if f_match else (
        2 * precision * recall / (precision + recall)
        if precision and recall and (precision + recall) > 0 else None
    )

    return precision, recall, f1


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def f1_score(p: Optional[float], r: Optional[float]) -> Optional[float]:
    if p is None or r is None: return None
    if p + r == 0: return 0.0
    return 2 * p * r / (p + r)


def load_existing(csv_path: Path) -> Dict[str, Dict[str, Any]]:
    done = {}
    if csv_path.exists():
        with csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                done[row["run_id"]] = row
    return done


def write_header_if_needed(csv_path: Path):
    if not csv_path.exists():
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "run_id", "thresh", "box_thresh",
                "precision", "recall", "f1",
                "duration_sec", "exit_code", "timestamp"
            ])


def append_result(csv_path: Path, row: Dict[str, Any]):
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            row["run_id"], row["thresh"], row["box_thresh"],
            row["precision"], row["recall"], row["f1"],
            row["duration_sec"], row["exit_code"], row["timestamp"]
        ])


def plot_results(csv_path: Path, out_dir: Path):
    import pandas as pd
    if not csv_path.exists(): return
    df = pd.read_csv(csv_path)

    for col in ["precision", "recall", "thresh", "box_thresh"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    valid = df.dropna(subset=["precision", "recall"])

    plt.figure(figsize=(7,6))
    sc = plt.scatter(valid["recall"], valid["precision"], c=valid["thresh"], s=36)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("DBNet Precision–Recall across (thresh, box_thresh)")
    cb = plt.colorbar(sc)
    cb.set_label("binarization thresh")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_dir / "pr_scatter.png", dpi=150)
    plt.close()

    
    pts = valid[["recall", "precision"]].values
    if len(pts) >= 2:
        is_dominated = np.zeros(len(pts), dtype=bool)
        for i, (rx, px) in enumerate(pts):
            dom = ((pts[:,0] >= rx) & (pts[:,1] > px)) | ((pts[:,0] > rx) & (pts[:,1] >= px))
            is_dominated[i] = np.any(dom)
        pf = valid.loc[~is_dominated].sort_values(["recall", "precision"])
        
        plt.figure(figsize=(7,6))
        plt.scatter(valid["recall"], valid["precision"], s=20, alpha=0.35)
        plt.plot(pf["recall"], pf["precision"], lw=2, label="Pareto front")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Pareto Front of (Recall, Precision)")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "pareto_front.png", dpi=150)
        plt.close()

        pf.to_csv(out_dir / "pareto_front.csv", index=False)


def run_one(thresh: float, box_thresh: float, out_dir: Path) -> Dict[str, Any]:
    run_id = mk_run_id(thresh, box_thresh)
    logs_dir = out_dir / "logs"
    ensure_dir(logs_dir)
    log_path = logs_dir / f"{run_id}.txt"

    cmd = [
        PY_EXE, EVAL_SCRIPT,
        YAML_PATH, "--resume", RESUME_PATH,
        "--thresh", str(thresh),
        "--box_thresh", str(box_thresh)
    ]

    t0 = time.time()
    exit_code = None

    with log_path.open("w", encoding="utf-8", errors="replace") as lf:
        lf.write(f"# CMD: {' '.join(cmd)}\n")
        lf.write(f"# START: {datetime.datetime.now().isoformat(timespec='seconds')}\n\n")
        lf.flush()

        proc = subprocess.Popen(
            cmd,
            cwd=str(EVAL_CWD),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace"
        )
        try:
            for line in proc.stdout:
                lf.write(line)
            proc.wait()
            exit_code = proc.returncode
        except Exception as e:
            
            lf.write(f"\n[SWEEP] Exception while running subprocess: {repr(e)}\n")
            exit_code = -999

        elapsed = time.time() - t0
        lf.write(f"\n# END: {datetime.datetime.now().isoformat(timespec='seconds')}\n")
        lf.write(f"# EXIT_CODE: {exit_code}\n")
        lf.write(f"# DURATION_SEC: {elapsed:.3f}\n")

    try:
        txt = log_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        txt = ""

    p, r, f1 = parse_metrics_from_text(txt)

    if p is None or r is None:
        print(f"[WARN] metrics not found for {run_id}. See log: {log_path}")

    return {
        "run_id": run_id,
        "thresh": thresh,
        "box_thresh": box_thresh,
        "precision": None if p is None else round(p, 6),
        "recall":    None if r is None else round(r, 6),
        "f1":        None if f1 is None else round(f1, 6),
        "duration_sec": round(elapsed, 3),
        "exit_code": exit_code,
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
    }


def main():
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT_OUT / f"DBNet_thresh_sweep_{stamp}"
    ensure_dir(out_dir)
    print(f"[INFO] Writing all outputs to: {out_dir.resolve()}")

    csv_path = out_dir / "results.csv"
    write_header_if_needed(csv_path)

    done = load_existing(csv_path)
    total = len(THRESH_VALUES) * len(BOX_THRESH_VALUES)
    k = 0

    for t in THRESH_VALUES:
        for b in BOX_THRESH_VALUES:
            k += 1
            run_id = mk_run_id(t, b)
            if run_id in done:
                print(f"[SKIP] {run_id} ({k}/{total}) already in results.csv")
                continue

            print(f"[RUN ] {run_id} ({k}/{total})")
            row = run_one(t, b, out_dir)
            append_result(csv_path, row)

            try:
                plot_results(csv_path, out_dir)
            except Exception as e:
                print(f"[WARN] Plotting failed: {e}")

         
            print(f"[OK  ] {run_id}  P={row['precision']}  R={row['recall']}  F1={row['f1']}  "
                  f"time={row['duration_sec']}s  exit={row['exit_code']}")

    print(f"\n[DONE] Sweep complete. Results: {csv_path.resolve()}")
    print(f"       Plots: {str(out_dir.resolve())}\\pr_scatter.png and pareto_front.png")
    print(f"       Logs per run: {str((out_dir/'logs').resolve())}")

if __name__ == "__main__":
    main()
