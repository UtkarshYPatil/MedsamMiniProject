"""CT preprocessing: HU windowing and ROI cropping."""
import numpy as np


def apply_window(vol_hu, level=-600, width=1500):
    """Hounsfield volume -> uint8 using a lung window."""
    lo, hi = level - width / 2, level + width / 2
    v = np.clip(vol_hu.astype(np.float32), lo, hi)
    return (((v - lo) / (hi - lo)) * 255).astype(np.uint8)


def crop_roi(vol, centroid_zyx, size=128, depth=16):
    """Crop a block around the nodule centroid. Returns (crop, origin_zyx)."""
    z, y, x = (int(round(c)) for c in centroid_zyx)
    Z, Y, X = vol.shape
    h = size // 2
    z0, z1 = max(0, z - depth), min(Z, z + depth)
    y0, y1 = max(0, y - h), min(Y, y + h)
    x0, x1 = max(0, x - h), min(X, x + h)
    return vol[z0:z1, y0:y1, x0:x1], (z0, y0, x0)


def to_rgb(vol_u8):
    """(D,H,W) uint8 -> (D,H,W,3). MedSAM2 expects 3 channels."""
    if vol_u8.ndim == 2:
        return np.repeat(vol_u8[..., None], 3, axis=-1)
    return np.repeat(vol_u8[..., None], 3, axis=-1)
