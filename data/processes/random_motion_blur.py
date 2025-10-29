


import math
import numpy as np
import cv2


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

# ---------- profile weights ----------
def _profile_weights(n: int, profile: str) -> np.ndarray:
    n = max(1, int(n))
    t = np.linspace(0, 1, n, dtype=np.float32)  # one-sided domain
    if profile == "box":
        w = np.ones_like(t, dtype=np.float32)
    elif profile == "gaussian":
        # smooth falloff; map t in [0,1] -> [-1,1] then apply gaussian
        u = (t * 2.0 - 1.0)
        w = np.exp(-0.5 * (u/0.6)**2).astype(np.float32)
    elif profile == "cosine":
        # raised-cosine over [0,1]: high at 0, low at 1
        w = (np.cos(t * math.pi) * 0.5 + 0.5).astype(np.float32)
    else:
        raise ValueError(f"Unknown profile: {profile}")
    s = float(w.sum())
    if s <= 0:
        w = np.ones_like(t, dtype=np.float32) / len(t)
    else:
        w /= s
    return w

# ---------- kernel builder (one-sided, horizontal) ----------
def _one_sided_horizontal_kernel(length: int, smear_dir: str, profile: str = "gaussian",
                                 antialias: bool = True) -> np.ndarray:
    L = max(1, int(length))
    k = 2 * L + 1
    kernel = np.zeros((k, k), dtype=np.float32)
    c = L  # center index

    # Build 1D horizontal weights on [0..L] (inclusive of center position 0)
    w = _profile_weights(L + 1, profile)  # positions: 0,1,2,...,L

    if smear_dir == "right":
        for i, wi in enumerate(w):           
            x = c + i
            if 0 <= x < k:
                kernel[c, x] += wi


    elif smear_dir == "left":
        for i, wi in enumerate(w):          
            x = c - i
            if 0 <= x < k:
                kernel[c, x] += wi
    else:
        raise ValueError("smear_dir must be 'left' or 'right'")

    # Gentle 2D antialias to reduce ringing
    if antialias and k >= 5:
        kernel = cv2.GaussianBlur(kernel, (3, 3), 0.5, borderType=cv2.BORDER_REPLICATE)

    s = kernel.sum()
    if s > 0:
        kernel /= s
    # ensure odd shape stays odd (already is)
    return kernel

# ---------- image ops ----------
def _apply_motion_blur(img_bgr: np.ndarray, kernel: np.ndarray, gamma_aware: bool=True) -> np.ndarray:
    # Detect input range & normalize to [0,1] float for filtering
    if img_bgr.dtype.kind in 'ui':        # uint8/uint16
        img = img_bgr.astype(np.float32) / 255.0
        scale_back = 255.0
    else:
        img = img_bgr.astype(np.float32)
        if img.max() > 1.5:               # looks like 0..255 floats
            img /= 255.0
            scale_back = 255.0
        else:                              # already 0..1 floats
            scale_back = 1.0

    if gamma_aware:
        img_lin = _srgb_to_linear(img)
        out_lin = np.empty_like(img_lin, dtype=np.float32)
        for c in range(3):
            out_lin[..., c] = cv2.filter2D(img_lin[..., c], -1, kernel, borderType=cv2.BORDER_REPLICATE)
        out_lin = np.clip(out_lin, 0.0, 1.0)
        out = _linear_to_srgb(out_lin)
    else:
        out = np.empty_like(img, dtype=np.float32)
        for c in range(3):
            out[..., c] = cv2.filter2D(img[..., c], -1, kernel, borderType=cv2.BORDER_REPLICATE)

    # Return float32 in 0..255 for NormalizeImage
    out = np.clip(out * scale_back, 0.0, 255.0).astype(np.float32)
    return out

class RandomMotionBlur:
    def __init__(self,
                 p_clean=0, p_mild=0, p_med=0, p_heavy=1,
                 mild=(1,10), med=(1,15), heavy=(16,30),
                 profile="gaussian", gamma_aware=True, antialias=True, seed=None,
                 **kwargs):
        # Tolerate extra framework-specific keys
        kwargs.pop('class', None)
        kwargs.pop('type', None)
        # print("INIT VALUES:", p_clean, p_mild, p_med, p_heavy)
        # print("KWARGS:", kwargs)
        ps = np.array([p_clean, p_mild, p_med, p_heavy], np.float32)
        self.p_cum = np.cumsum(ps / (ps.sum() + 1e-8))
        self.bins = [None, mild, med, heavy]
        self.profile, self.gamma_aware, self.antialias = profile, gamma_aware, antialias
        self.rng = np.random.RandomState(seed)

        # small kernel cache to avoid rebuilding within a batch
        self._kernel_cache = {}

    def _sample_length(self, lo_hi):
        if lo_hi is None:
            return None
        lo, hi = int(lo_hi[0]), int(lo_hi[1])
        if hi < lo: lo, hi = hi, lo
        return int(self.rng.randint(lo, hi + 1))

    def __call__(self, data):
        # print("\n\n\n")
        # print(self.p_clean)
        # print("\n\n\n")
        img = data['image']
        r = self.rng.rand()
        idx = int(np.searchsorted(self.p_cum, r, side="right"))
        lo_hi = self.bins[idx]

        if lo_hi is not None:
            L = self._sample_length(lo_hi)
            if L <= 0:
                data['image'] = img.astype(np.float32, copy=False)
                return data     # true "clean" if L==0
            # smear_dir = "left" if (self.rng.rand() < 0.5) else "right"
            smear_dir = "left"
            key = (L, smear_dir, self.profile, self.antialias)
            k = self._kernel_cache.get(key)
            if k is None:
                k = _one_sided_horizontal_kernel(L, smear_dir, profile=self.profile, antialias=self.antialias)
                if len(self._kernel_cache) > 64: self._kernel_cache.clear()
                self._kernel_cache[key] = k
            img = _apply_motion_blur(img, k, gamma_aware=self.gamma_aware)

        data['image'] = img.astype(np.float32, copy=False)
        return data