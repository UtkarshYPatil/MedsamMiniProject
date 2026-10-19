"""
Service-layer tests that need no Docker, no Redis, no Postgres and no GPU.

They cover the part of the stack most likely to break quietly: the storage
round-trip, the PNG rendering the UI depends on, and the worker task end to
end against the dummy segmenter.

    python -m pytest tests/test_services.py -v
"""
import os

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _isolated_data_dir(tmp_path, monkeypatch):
    """Point the services at a scratch directory and force the dummy model."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("DRY_RUN", "1")
    monkeypatch.setenv("DEVICE", "cpu")

    # settings reads the environment at import time, so reload it and anything
    # holding a reference to it.
    import importlib
    from services.common import settings as s
    importlib.reload(s)
    from services.common import store as st
    importlib.reload(st)
    yield


# ----------------------------------------------------------------- phantom

def test_phantom_has_the_requested_geometry():
    from services.common import store
    vol, spacing = store.make_phantom(diameter_mm=9.0)

    assert vol.shape == (33, 128, 128)
    assert vol.dtype == np.int16
    assert spacing.shape == (3,)
    # lung background well below the nodule, both inside CT's valid range
    assert -1024 <= vol.min() and vol.max() <= 3071
    assert vol.mean() < -400, "a phantom that is mostly lung should read mostly air"


def test_phantom_diameter_scales():
    from services.common import store
    small, sp = store.make_phantom(diameter_mm=5.0)
    large, _ = store.make_phantom(diameter_mm=20.0)
    # count voxels clearly above lung attenuation
    assert (large > -400).sum() > (small > -400).sum() * 4


# ----------------------------------------------------------------- storage

def test_case_round_trip():
    from services.common import store
    vol, spacing = store.make_phantom()
    case_id = store.save_case(vol, spacing, source="phantom")

    back, back_spacing = store.load_case(case_id)
    assert back.shape == vol.shape
    np.testing.assert_allclose(back_spacing, spacing)
    np.testing.assert_array_equal(back.astype(np.int16), vol)


def test_save_case_rejects_2d():
    from services.common import store
    with pytest.raises(ValueError, match="3D"):
        store.save_case(np.zeros((64, 64), dtype=np.int16), (1, 1, 1))


def test_read_npz_upload_accepts_prepare_data_format(tmp_path):
    from services.common import store
    vol, spacing = store.make_phantom()
    p = tmp_path / "case.npz"
    np.savez_compressed(p, image_hu=vol, mask=np.zeros_like(vol, bool), spacing=spacing)

    image, sp = store.read_npz_upload(p.read_bytes())
    np.testing.assert_array_equal(image, vol)
    np.testing.assert_allclose(sp, spacing)


def test_read_npz_upload_defaults_spacing_when_absent(tmp_path):
    from services.common import store
    p = tmp_path / "nospacing.npz"
    np.savez_compressed(p, image_hu=np.zeros((5, 8, 8), dtype=np.int16))
    _, sp = store.read_npz_upload(p.read_bytes())
    np.testing.assert_allclose(sp, [1.0, 1.0, 1.0])


def test_read_npz_upload_explains_a_missing_image(tmp_path):
    from services.common import store
    p = tmp_path / "wrong.npz"
    np.savez_compressed(p, something_else=np.zeros((4, 4)))
    with pytest.raises(ValueError, match="no image array"):
        store.read_npz_upload(p.read_bytes())


# ---------------------------------------------------------------- rendering

def test_render_png_is_a_png():
    from services.common import store
    vol, _ = store.make_phantom()
    png = store.render_png(vol[16])
    assert png[:8] == b"\x89PNG\r\n\x1a\n"


def test_render_png_with_mask_and_box_still_decodes():
    from services.common import store
    vol, _ = store.make_phantom()
    mask = np.zeros(vol.shape[1:], dtype=bool)
    mask[60:68, 60:68] = True

    png = store.render_png(vol[16], mask, [55, 55, 73, 73])
    from PIL import Image
    import io
    img = Image.open(io.BytesIO(png))
    assert img.size == (vol.shape[2], vol.shape[1])
    assert img.mode == "RGB"


def test_outline_is_the_border_only():
    from services.common.store import _outline
    m = np.zeros((20, 20), dtype=bool)
    m[5:15, 5:15] = True           # 10x10 filled square

    edge = _outline(m)
    assert edge.sum() == 36, "a 10x10 square has 36 border pixels"
    assert not edge[9, 9], "the interior must not be part of the outline"
    assert edge[5, 5], "a corner must be"


def test_outline_of_empty_mask_is_empty():
    from services.common.store import _outline
    assert not _outline(np.zeros((8, 8), dtype=bool)).any()


# ------------------------------------------------------------ worker task

def test_segment_case_end_to_end_with_the_dummy_model():
    """
    The whole worker path: load case -> window -> segment -> measure -> save.

    DRY_RUN means the mask is meaningless, but every step that moves or shapes
    data is real, which is what this is checking.
    """
    from services.common import store
    from services.worker import tasks

    vol, spacing = store.make_phantom(diameter_mm=10.0)
    case_id = store.save_case(vol, spacing, source="phantom")

    res = tasks.segment_case(case_id, prompt_z=16, box=[54.0, 54.0, 74.0, 74.0])

    assert res["dry_run"] is True
    assert res["device"] == "cpu"
    assert res["seconds"] >= 0

    if res.get("empty"):
        pytest.skip("dummy segmenter returned an empty mask on this input")

    for key in ("diameter_mean_mm", "volume_mm3", "confidence", "measurement"):
        assert key in res, f"the UI reads {key} and it is missing"
    assert res["confidence"] in ("high", "medium", "low", "unknown")
    assert res["volume_mm3"] > 0
    assert isinstance(res["flags"], list)


def test_segment_case_rejects_a_slice_outside_the_volume():
    from services.common import store
    from services.worker import tasks

    vol, spacing = store.make_phantom()
    case_id = store.save_case(vol, spacing)

    with pytest.raises(ValueError, match="outside"):
        tasks.segment_case(case_id, prompt_z=999, box=[10.0, 10.0, 30.0, 30.0])


def test_segment_case_reports_a_missing_case():
    from services.worker import tasks
    with pytest.raises(FileNotFoundError):
        tasks.segment_case("deadbeefcafe", prompt_z=0, box=[1.0, 1.0, 9.0, 9.0])


# -------------------------------------------------------------- settings

def test_settings_never_use_the_windows_research_paths():
    """
    configs/config.yaml carries E:/MiniProject paths. A container that resolved
    them would fail at runtime, so the services must not inherit them.
    """
    from services.common import settings
    assert "E:/" not in str(settings.DATA_DIR)
    assert str(settings.DATA_DIR).startswith("/") or str(settings.DATA_DIR)[1:3] != ":\\"


# ------------------------------------------------- dependency completeness

def test_worker_requirements_cover_src_imports():
    """
    Every third-party module src/ imports must be declared for the worker.

    This test exists because it was not. scikit-image and scipy are imported
    *inside* functions (metrics.max_diameter_mm, density.solid_component_mask),
    so a grep for imports at the start of a line missed them, the worker image
    shipped without them, and it died with ModuleNotFoundError on the first
    real job -- while the test suite passed, because both happen to be
    installed in a development environment.

    A lazy import is still a dependency. This checks the declaration, not the
    environment, so it fails on the machine that writes the Dockerfile rather
    than on the machine that runs it.
    """
    import re
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]

    # Provided by the base image or installed separately in the Dockerfile.
    from_base_image = {"torch", "torchvision", "sam2"}
    # Not reachable from the segmentation path: pylidc is dataset preparation,
    # medpy is only metrics.hd95, which measure() does not call.
    not_on_service_path = {"pylidc", "medpy"}
    # import name -> distribution name on PyPI
    dist = {"skimage": "scikit-image", "yaml": "pyyaml", "PIL": "pillow",
            "cv2": "opencv-python", "sklearn": "scikit-learn"}

    found = set()
    for py in (root / "src").rglob("*.py"):
        for line in py.read_text(encoding="utf-8").splitlines():
            m = re.match(r"\s*(?:import|from)\s+([A-Za-z_][\w.]*)", line)
            if m:
                found.add(m.group(1).split(".")[0])

    needed = {m for m in found
              if m not in sys.stdlib_module_names
              and m != "src"
              and m not in from_base_image
              and m not in not_on_service_path}

    declared = (root / "services" / "requirements-worker.txt").read_text().lower()
    missing = sorted(m for m in needed
                     if dist.get(m, m).lower() not in declared)

    assert not missing, (
        f"src/ imports {missing} but services/requirements-worker.txt does not "
        f"declare them -- the worker container will fail at runtime. "
        f"Distribution names: {[dist.get(m, m) for m in missing]}")
