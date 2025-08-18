#!/usr/bin/env python3
import sys, re, csv
from pathlib import Path

# Run: python compile_scores.py [path/to/eval_results_ic15]

# NEW: match "length_12" or "length_12_left"/"length_12_right"
FOLDER_RE = re.compile(r"^length_(\d+)(?:_(left|right))?$", re.I)

PREC_RE = re.compile(r"precision\s*:\s*([0-9]*\.?[0-9]+)", re.I)
REC_RE  = re.compile(r"recall\s*:\s*([0-9]*\.?[0-9]+)", re.I)
FME_RE  = re.compile(r"(?:fmeasure|f-measure|f1|fscore)\s*:\s*([0-9]*\.?[0-9]+)", re.I)

def parse_scores(p: Path):
    txt = p.read_text(encoding="utf-8", errors="ignore")
    mp = PREC_RE.search(txt); mr = REC_RE.search(txt); mf = FME_RE.search(txt)
    if not (mp and mr and mf):
        return None
    return float(mp.group(1)), float(mr.group(1)), float(mf.group(1))

def main():
    # Default back to eval_results_ic15 (pass an argument to override)
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("eval_results_ic15")
    if not root.exists():
        print(f"[ERROR] Not found: {root.resolve()}"); sys.exit(1)

    rows = []
    for sub in sorted(root.iterdir()):
        if not sub.is_dir():
            continue
        m = FOLDER_RE.search(sub.name)
        if not m:
            continue
        scores = sub / "scores.txt"
        if not scores.exists():
            continue
        parsed = parse_scores(scores)
        if not parsed:
            continue

        length = int(m.group(1))
        side = (m.group(2) or "").lower()  # "", "left", or "right"
        precision, recall, fmeasure = parsed
        rows.append((length, side, precision, recall, fmeasure))

    # Sort by length, then side: (no side) < left < right
    side_order = {"": -1, "left": 0, "right": 1}
    rows.sort(key=lambda r: (r[0], side_order.get(r[1], 2)))

    out_csv = root / "summary.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["length", "side", "precision", "recall", "fmeasure"])
        w.writerows(rows)

    print(f"[OK] Wrote {len(rows)} rows → {out_csv.resolve()}")

if __name__ == "__main__":
    main()
