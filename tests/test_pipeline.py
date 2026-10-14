import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluate.metrics import dice, iou, measure, max_diameter_mm
from src.prompts.bbox import bbox_from_mask, bbox_from_click, best_slice
from src.data.preprocess import apply_window, crop_roi
from src.report.density import classify, hu_profile, core_mask
from src.report.sizing import summarize, flag_quality
from src.models.medsam2 import (DummySegmenter, largest_connected_component,
                                resize_grayscale_to_rgb_and_resize)


def sphere(D=20, H=64, W=64, r=6, cz=10, cy=32, cx=32):
    zz, yy, xx = np.ogrid[:D, :H, :W]
    return ((zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2) <= r * r


def hu_volume(mask, inside=-50.0, outside=-850.0):
    v = np.full(mask.shape, outside, dtype=np.float32)
    v[mask] = inside
    return v


# -- metrics -----------------------------------------------------------

def test_dice_identical():
    m = sphere()
    assert dice(m, m) == 1.0


def test_dice_disjoint():
    assert dice(sphere(cx=16), sphere(cx=52)) == 0.0


def test_iou_bounds():
    assert 0.0 < iou(sphere(), sphere(cx=36)) < 1.0


def test_empty_masks_are_perfect():
    z = np.zeros((4, 8, 8), bool)
    assert dice(z, z) == 1.0 and iou(z, z) == 1.0


def test_diameter_scales_with_spacing():
    m = sphere(r=6)
    d1 = max_diameter_mm(m, (1.0, 1.0, 1.0))
    d2 = max_diameter_mm(m, (1.0, 2.0, 2.0))
    assert d2 == pytest.approx(2 * d1, rel=1e-6)


def test_volume_matches_voxel_count():
    m = sphere()
    out = measure(m, np.full(m.shape, -500.0), (1.0, 1.0, 1.0))
    assert out["volume_mm3"] == pytest.approx(m.sum())


# -- preprocessing -----------------------------------------------------

def test_window_clips_to_uint8():
    out = apply_window(np.array([[-2000.0, -600.0, 500.0]]), -600, 1500)
    assert out.dtype == np.uint8 and out.min() == 0 and out.max() == 255


def test_crop_stays_in_bounds():
    c, origin = crop_roi(np.zeros((20, 64, 64)), (0, 0, 0), size=32, depth=8)
    assert all(s > 0 for s in c.shape) and origin == (0, 0, 0)


# -- prompts -----------------------------------------------------------

def test_bbox_wraps_mask():
    m = np.zeros((32, 32), bool)
    m[10:20, 12:18] = True
    x0, y0, x1, y1 = bbox_from_mask(m, padding=2)
    assert x0 <= 12 and y0 <= 10 and x1 >= 17 and y1 >= 19


def test_bbox_from_empty_is_none():
    assert bbox_from_mask(np.zeros((8, 8), bool)) is None


def test_click_box_clamped():
    x0, y0, x1, y1 = bbox_from_click(2, 2, 10, (32, 32))
    assert x0 == 0 and y0 == 0 and x1 <= 31 and y1 <= 31


def test_best_slice_finds_widest():
    assert abs(best_slice(sphere(cz=7)) - 7) <= 1


# -- density -----------------------------------------------------------

def test_density_solid():
    m = sphere()
    assert classify(hu_volume(m, -60.0), m)[0] == "solid"


def test_density_ground_glass():
    m = sphere()
    assert classify(hu_volume(m, -780.0, -850.0), m)[0] == "ground_glass"


def test_density_empty_mask():
    assert classify(np.zeros((2, 4, 4)), np.zeros((2, 4, 4), bool))[0] == "unknown"


def test_density_is_never_claimed_calibrated():
    """Guards the honest-reporting contract -- see src/report/density.py."""
    m = sphere()
    _, ev = classify(hu_volume(m), m)
    assert ev["calibrated"] is False and ev["provisional"] is True


def test_core_mask_erodes_large_nodules():
    m = sphere(r=8)
    core, eroded = core_mask(m)
    assert eroded is True and 0 < core.sum() < m.sum()


def test_core_mask_falls_back_for_tiny_nodules():
    m = np.zeros((6, 16, 16), bool)
    m[3, 8, 8] = True
    core, eroded = core_mask(m)
    assert eroded is False and core.sum() == m.sum()


def test_hu_profile_reports_core_and_full():
    m = sphere(r=8)
    p = hu_profile(hu_volume(m), m)
    assert p["core_voxels"] < m.sum()
    assert set(p) >= {"hu_p50", "core_hu_p50", "core_solid_fraction"}


# -- confidence flagging -----------------------------------------------

def test_ground_glass_always_low_confidence():
    m = sphere()
    assert summarize(m, hu_volume(m, -780.0, -850.0), (1.0, 1.0, 1.0)).confidence == "low"


def test_tiny_mask_is_low_confidence():
    m = np.zeros((6, 16, 16), bool)
    m[3, 8, 8] = True
    level, flags = flag_quality(m, {"n_voxels": 1, "diameter_mean_mm": 0.5}, "solid")
    assert level == "low" and any("20 voxels" in f for f in flags)


def test_single_slice_mask_is_low_confidence():
    m = np.zeros((6, 16, 16), bool)
    m[3, 4:12, 4:12] = True
    level, flags = flag_quality(m, {"n_voxels": 64, "diameter_mean_mm": 8.0}, "solid")
    assert level == "low" and any("single slice" in f for f in flags)


def test_clean_solid_nodule_is_high_confidence():
    m = sphere(r=8)
    r = summarize(m, hu_volume(m), (1.0, 1.0, 1.0))
    assert r.confidence == "high" and r.flags == []


def test_summary_always_carries_hu_statistics():
    """The tool leads with HU, not the provisional label."""
    m = sphere(r=8)
    r = summarize(m, hu_volume(m), (1.0, 1.0, 1.0))
    for field in ("hu_p50", "hu_p90", "core_hu_p50", "core_solid_fraction"):
        assert getattr(r, field) == getattr(r, field)   # not NaN
    assert r.density_calibrated is False


# -- segmenter ---------------------------------------------------------

def test_dummy_segmenter_shape_and_overlap():
    vol = np.zeros((16, 48, 48), np.uint8)
    out = DummySegmenter().segment_volume(vol, 8, [16, 16, 32, 32])
    assert out.shape == (16, 48, 48) and out.dtype == bool and out[8].any()


def test_resize_to_rgb_512():
    out = resize_grayscale_to_rgb_and_resize(np.zeros((4, 64, 64), np.uint8), 512)
    assert out.shape == (4, 3, 512, 512)


def test_largest_connected_component_drops_the_smaller_blob():
    m = np.zeros((8, 32, 32), bool)
    m[2:6, 4:14, 4:14] = True        # big
    m[1, 28, 28] = True              # speck
    out = largest_connected_component(m)
    assert out.sum() < m.sum() and not out[1, 28, 28]


def test_largest_connected_component_handles_empty():
    z = np.zeros((4, 8, 8), bool)
    assert largest_connected_component(z).sum() == 0
