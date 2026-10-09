"""CT preprocessing: Hounsfield windowing and ROI cropping."""
import numpy as np


def apply_window(volume_hu, level=-600, width=1500):
    """Lung window -> uint8. volume_hu is already in Hounsfield Units."""
    lo, hi = level - width / 2, level + width / 2
    v = np.clip(volume_hu, lo, hi)
    v = (v - lo) / (hi - lo)
    return (v * 255).astype(np.uint8)


def crop_roi(volume, centroid_zyx, size=96, depth=16):
    """Crop a (2*depth, size, size) block around the nodule centroid."""
    z, y, x = (int(round(c)) for c in centroid_zyx)
    Z, Y, X = volume.shape
    h = size // 2

    z0, z1 = max(0, z - depth), min(Z, z + depth)
    y0, y1 = max(0, y - h), min(Y, y + h)
    x0, x1 = max(0, x - h), min(X, x + h)

    return volume[z0:z1, y0:y1, x0:x1], (z0, y0, x0)


def to_rgb(slice_u8):
    """MedSAM2 expects 3-channel input."""
    return np.repeat(slice_u8[..., None], 3, axis=-1)
