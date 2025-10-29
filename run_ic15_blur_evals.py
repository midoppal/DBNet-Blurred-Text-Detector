
import os, sys, shutil, subprocess, re, time
from pathlib import Path
import signal
from contextlib import contextmanager
signal.signal(signal.SIGINT,  lambda *_: sys.exit(130))
signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))

# ---------- CONFIG ----------
REPO_ROOT = Path(__file__).resolve().parent
YAML_CFG  = "experiments/seg_detector/ic15_resnet50_deform_thre.yaml"
# CKPT_PATH = "./pretrained/ic15_resnet50.pth"
# CKPT_PATH = "./workspace/SegDetectorModel-L1BalanceCELoss/model/model_epoch_100_minibatch_100500"
# CKPT_PATH = "./workspace/SegDetectorModel-L1BalanceCELoss/model/final" 
CKPT_PATH = "./workspace/SegDetectorModel-L1BalanceCELoss/model/300_plus150_plus300lr003_1_15"
# CKPT_PATH = "./fine_tuned_models/0_30_left_100_epochs_10_90_split_model"

DATASET_ROOT = REPO_ROOT / "datasets" / "icdar2015"
TEST_LIST = DATASET_ROOT / "test_list.txt"
TEST_IMAGES  = DATASET_ROOT / "test_images"
ORIG_IMAGES  = DATASET_ROOT / "original_test_images"  # optional baseline location

# blurred set naming pattern
# BLUR_GLOB   = "blurred_test_images_length_*_jitter_*"
BLUR_GLOB   = "blurred_test_images_length_*"
RESULTS_DIR = REPO_ROOT / "eval_results_ic15"   # where logs & scores go

PYTHON_EXE  = sys.executable  # use current interpreter
# ----------------------------

def is_windows() -> bool:
    return os.name == "nt"

def tail_last_n_nonempty(text: str, n: int = 4):
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    return lines[-n:] if len(lines) >= n else lines

def remove_path(p: Path):
    # Handle both real dirs and junctions on Windows
    if not p.exists() and not os.path.lexists(p):
        return
    if os.name == "nt":
        # Use rmdir to remove directory or junction reliably
        try:
            subprocess.run(
                ["cmd", "/c", "rmdir", "/S", "/Q", str(p)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False
            )
            time.sleep(0.1)  # tiny pause so the FS catches up
        except Exception:
            pass
        return
    # POSIX fallback
    if p.is_symlink():
        p.unlink(missing_ok=True)
    elif p.is_dir():
        shutil.rmtree(p, ignore_errors=True)
    else:
        p.unlink(missing_ok=True)

def create_junction(target: Path, source: Path) -> bool:
    if os.name != "nt":
        return False
    try:
        # ensure target gone (covers dir/junction/symlink)
        remove_path(target)
        if target.exists() or os.path.lexists(target):
            # still there? bail to copy mode
            return False
        target.parent.mkdir(parents=True, exist_ok=True)
        proc = subprocess.run(["cmd", "/c", "mklink", "/J", str(target), str(source)],
                              capture_output=True, text=True)
        return proc.returncode == 0
    except Exception:
        return False

def copy_tree(src: Path, dst: Path):
    remove_path(dst)
    shutil.copytree(src, dst)

def switch_test_images_to(src_dir: Path):
    """
    Point datasets/icdar2015/test_images to src_dir.
    Try junction on Windows for speed; fall back to copy otherwise.
    """
    if not src_dir.exists():
        raise FileNotFoundError(f"Source images not found: {src_dir}")
    # Remove existing test_images folder or link
    remove_path(TEST_IMAGES)
    # Try junction
    if create_junction(TEST_IMAGES, src_dir):
        return "junction"
    # Fallback: copy
    copy_tree(src_dir, TEST_IMAGES)
    return "copy"

def run_eval():
    cmd = [PYTHON_EXE, "eval.py", YAML_CFG, "--resume", CKPT_PATH]
    if is_windows():
        proc = subprocess.Popen(cmd, cwd=REPO_ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
    else:
        proc = subprocess.Popen(cmd, cwd=REPO_ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        stdout, stderr = proc.communicate()
        return proc.returncode, stdout, stderr
    except KeyboardInterrupt:
        # forward interrupt to the child, give it a moment, then kill if needed
        try:
            if is_windows():
                proc.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            proc.kill()
        raise

# def parse_LJ_from_name(name: str):
#     # name like 'blurred_test_images_length_7_jitter_3'
#     m = re.search(r"length_(\d+)_jitter_(\d+)", name)
#     if not m:
#         return None, None
#     return int(m.group(1)), int(m.group(2))

def fmt_hms(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h:d}:{m:02d}:{s:02d}"
    else:
        return f"{m:02d}:{s:02d}"

def parse_length_dir_from_name(name: str):
    """
    Accepts names like:
      - blurred_test_images_length_0
      - blurred_test_images_length_15_left
      - blurred_test_images_length_15_right
    Returns: (length:int | None, direction:str | None)
    """
    m = re.search(r"length_(\d+)(?:_(left|right))?$", name, re.I)
    if not m:
        return None, None
    length = int(m.group(1))
    direction = m.group(2)  # None, 'left', or 'right'
    return length, direction

# ---------------------------- .txt swapper ----------------------------
def _read_canonical_names():
    return [l.strip() for l in TEST_LIST.read_text(encoding="utf-8").splitlines() if l.strip()]

def _available_name_map(search_dirs):
    """
    Return dict mapping BOTH full names and stems -> actual Path
    so we handle ext changes (.jpg -> .png) gracefully.
    """
    m = {}
    for d in search_dirs:
        if not d.exists(): 
            continue
        for p in d.glob("*.*"):
            m[p.name] = p
            # if two files share a stem, keep the first seen
            m.setdefault(p.stem, p)
    return m

def _filter_names_for_folder(blur_dir):
    # try common layouts: <blur>/test_images, <blur>/left/test_images, <blur>/right/test_images,
    # and (if you saved flat) <blur> itself
    candidates = [
        blur_dir / "test_images",
        blur_dir / "left" / "test_images",
        blur_dir / "right" / "test_images",
        blur_dir
    ]
    avail = _available_name_map(candidates)

    filtered = []
    for nm in _read_canonical_names():
        if nm in avail:
            filtered.append(nm)  # exact match
        else:
            stem = Path(nm).stem
            if stem in avail:
                filtered.append(avail[stem].name)  # use actual filename (ext may differ)
    return filtered

@contextmanager
def temporarily_replace_test_list(new_names):
    """
    Overwrite datasets/icdar2015/test_list.txt with new_names, then restore the original.
    """
    bak = TEST_LIST.with_suffix(".txt.bak")
    shutil.copy2(TEST_LIST, bak)
    try:
        TEST_LIST.write_text("\n".join(new_names) + "\n", encoding="utf-8", newline="\n")
        yield
    finally:
        bak.replace(TEST_LIST)
# ------------------------------------------------------------

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    blur_sets = list((DATASET_ROOT).glob(BLUR_GLOB))
    def _sort_key(p: Path):
        L, D = parse_length_dir_from_name(p.name)
        d_order = {None: -1, "left": 0, "right": 1}
        return ((L if L is not None else 10**9), d_order.get(D, 2), p.name)
    blur_sets.sort(key=_sort_key)
    if not blur_sets:
        print(f"[ERROR] No blurred sets found under {DATASET_ROOT} matching '{BLUR_GLOB}'")
        sys.exit(1)

    total_sets = len(blur_sets)
    print(f"[INFO] Found {total_sets} blurred sets. Starting evals...")
    t0_all = time.time()

    for idx, src in enumerate(blur_sets, start=1):
        t0 = time.time()
        # L, J = parse_LJ_from_name(src.name)
        # tag = f"length_{L}_jitter_{J}" if L is not None else src.name
        # if (src.name != "blurred_test_images_length_30_left"):
        if (src.name != "blurred_test_images_length_30_right"):
            continue
        L, D = parse_length_dir_from_name(src.name)
        # if D != "right" and src.name != "blurred_test_images_length_0":
        #     print(f"[skip] {src.name}: only 'left' direction supported for now.")
        #     continue
        tag = (f"length_{L}_{D}" if (L is not None and D) else
               (f"length_{L}" if L is not None else src.name))
        out_dir = RESULTS_DIR / tag
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[STEP] ({idx}/{total_sets}) Switching test_images -> {src} ({tag})")
        # mode = switch_test_images_to(src)
        # print(f"[STEP] test_images set via {mode}. Running eval...")

        # rc, stdout, stderr = run_eval()
        mode = switch_test_images_to(src)
        names = _filter_names_for_folder(src)
        if not names:
            print(f"[skip] {tag}: no matching files found for test_list.txt")
            continue
        print(f"[STEP] test_images set via {mode}. Using {len(names)} names. Running eval...")
        with temporarily_replace_test_list(names):
            rc, stdout, stderr = run_eval()

        (out_dir / "stdout.log").write_text(stdout, encoding="utf-8")
        (out_dir / "stderr.log").write_text(stderr or "", encoding="utf-8")

        # last4 = tail_last_n_nonempty(stdout, 4)
        METRIC_RE = re.compile(r'(precision|recall|fmeasure)\s*:\s*[\d.]+', re.I)
        combined = (stdout or "") + "\n" + (stderr or "")
        lines = [ln.rstrip() for ln in combined.splitlines() if METRIC_RE.search(ln)]
        last4 = lines[-4:] if len(lines) >= 4 else tail_last_n_nonempty(combined, 4)


        (out_dir / "scores.txt").write_text("\n".join(last4) + "\n", encoding="utf-8")

        status = "OK" if rc == 0 else f"RC={rc}"
        print(f"[DONE] {tag}: {status}. Last 4 lines:")
        for ln in last4:
            print("   ", ln)

        # -------- Progress summary after each folder --------
        dt = time.time() - t0
        elapsed = time.time() - t0_all
        avg_per_set = elapsed / idx
        remaining = avg_per_set * (total_sets - idx)
        print(f"[PROGRESS] Completed {idx}/{total_sets} | {tag} | took {dt:.1f}s | elapsed {fmt_hms(elapsed)} | ETA {fmt_hms(remaining)}")
        (out_dir / "timing.txt").write_text(f"took_seconds={dt:.3f}\nelapsed_seconds={elapsed:.3f}\n", encoding="utf-8")
        # ----------------------------------------------------

    # Restore to original_test_images if it exists
    if ORIG_IMAGES.exists():
        print("\n[FINAL] Restoring test_images to original_test_images...")
        switch_test_images_to(ORIG_IMAGES)
        print("[FINAL] Restore complete.")

    print(f"\n[ALL DONE] Results saved under: {RESULTS_DIR.resolve()}")

if __name__ == "__main__":
    main()
