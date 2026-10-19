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

def test_settings_never_use_the_windows_research_paths(monkeypatch):
    """
    configs/config.yaml carries E:/MiniProject paths. A container that resolved
    them would fail at runtime, so the services must not inherit them.

    The check is on the DEFAULT, with DATA_DIR unset. An earlier version
    asserted the resolved path looked POSIX -- which tested the test's own
    tmp_path fixture, and so failed on Windows, where tmp_path is C:\\...
    while the container path it was meant to check was never involved.
    """
    import importlib
    monkeypatch.delenv("DATA_DIR", raising=False)
    from services.common import settings as s
    importlib.reload(s)
    try:
        assert str(s.DATA_DIR) in ("/data", "\\data"), \
            f"the default data directory is {s.DATA_DIR}, not the container's /data"
        assert "MiniProject" not in str(s.DATA_DIR)
        assert "E:" not in str(s.DATA_DIR)
    finally:
        monkeypatch.undo()
        importlib.reload(s)


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

    declared = (root / "services" / "requirements-worker.txt").read_text(encoding="utf-8").lower()
    missing = sorted(m for m in needed
                     if dist.get(m, m).lower() not in declared)

    assert not missing, (
        f"src/ imports {missing} but services/requirements-worker.txt does not "
        f"declare them -- the worker container will fail at runtime. "
        f"Distribution names: {[dist.get(m, m) for m in missing]}")


# ------------------------------------------------------------ device choice

def _fresh_tasks(monkeypatch, device):
    """Re-import the worker with DEVICE set, clearing the cached resolution."""
    import importlib
    monkeypatch.setenv("DEVICE", device)
    from services.common import settings as s
    importlib.reload(s)
    from services.worker import tasks as t
    importlib.reload(t)
    t._device = None
    return t


def test_explicit_cpu_is_honoured(monkeypatch):
    assert _fresh_tasks(monkeypatch, "cpu").resolve_device() == "cpu"


def test_explicit_cuda_is_honoured_even_without_a_gpu(monkeypatch):
    """
    An explicit request is not second-guessed. Someone who asked for the GPU
    should get a loud failure, not a silent 30x slowdown.
    """
    assert _fresh_tasks(monkeypatch, "cuda").resolve_device() == "cuda"


def test_auto_falls_back_to_cpu_when_no_gpu_is_visible(monkeypatch):
    t = _fresh_tasks(monkeypatch, "auto")

    import sys
    import types
    fake = types.ModuleType("torch")
    fake.cuda = types.SimpleNamespace(is_available=lambda: False, device_count=lambda: 0)
    monkeypatch.setitem(sys.modules, "torch", fake)

    assert t.resolve_device() == "cpu"
    assert "requested auto" in t.device_detail()


def test_auto_picks_cuda_when_a_gpu_is_visible(monkeypatch):
    t = _fresh_tasks(monkeypatch, "auto")

    import sys
    import types
    fake = types.ModuleType("torch")
    fake.cuda = types.SimpleNamespace(
        is_available=lambda: True, device_count=lambda: 1,
        get_device_name=lambda i: "NVIDIA GeForce RTX 4060 Laptop GPU")
    monkeypatch.setitem(sys.modules, "torch", fake)

    assert t.resolve_device() == "cuda"
    assert "RTX 4060" in t.device_detail()


def test_auto_survives_torch_being_absent(monkeypatch):
    """The api and ui containers have no torch; importing must not explode."""
    t = _fresh_tasks(monkeypatch, "auto")

    import builtins
    real_import = builtins.__import__

    def no_torch(name, *a, **k):
        if name == "torch":
            raise ImportError("no torch here")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_torch)
    assert t.resolve_device() == "cpu"


def test_resolution_is_cached(monkeypatch):
    t = _fresh_tasks(monkeypatch, "cpu")
    assert t.resolve_device() == "cpu"
    t2 = _fresh_tasks(monkeypatch, "cpu")
    t2._device = "cuda"                 # pretend it already resolved
    assert t2.resolve_device() == "cuda", "a resolved device must not be recomputed"


# ---------------------------------------------------------- module integrity

def test_every_module_imports():
    """
    Import every module in the project.

    compileall only proves the syntax parses. This caught a real bug: a
    docstring in src/report/sizing.py was closed early, which left a section
    heading as a bare statement. Syntax-valid, import-fatal.
    """
    import importlib
    import pathlib

    # Streamlit pages are EXECUTED top to bottom by the framework, not
    # imported. Importing one runs the whole page: it calls the API, and
    # st.stop() -- which halts the script inside Streamlit's runtime -- is only
    # a logged warning outside it, so execution continues past branches that
    # were meant to be terminal and fails on a name that was never bound.
    # That is a fact about Streamlit, not a defect in the page, so these are
    # syntax-checked below instead.
    ENTRY_POINTS = {"services/ui/app.py"}

    root = pathlib.Path(__file__).resolve().parents[1]
    skipped, failed = [], []
    for path in sorted(root.glob("src/**/*.py")) + sorted(root.glob("services/**/*.py")):
        if path.name == "__init__.py":
            continue
        if path.relative_to(root).as_posix() in ENTRY_POINTS:
            src = path.read_text(encoding="utf-8")
            try:
                compile(src, str(path), "exec")
            except SyntaxError as e:
                failed.append((str(path), f"SyntaxError: {e}"))
            continue
        mod = ".".join(path.relative_to(root).with_suffix("").parts)
        try:
            importlib.import_module(mod)
        except ImportError as e:
            # A missing third-party package is an environment fact, not a bug
            # in the module. Anything else -- NameError, SyntaxError, a bad
            # relative import -- is the module's own fault and must fail.
            skipped.append((mod, str(e)))
        except Exception as e:
            failed.append((mod, f"{type(e).__name__}: {e}"))

    assert not failed, "modules that import-fail on their own account:\n" + \
        "\n".join(f"  {m}: {e}" for m, e in failed)


# ------------------------------------------------------------- auto-prompt

def _slab_volume(shape=(33, 128, 128)):
    """A volume whose only bright object is a chest-wall-like slab at the edge."""
    vol = np.full(shape, -800, dtype=np.int16)
    vol[:, :, :12] = 40                       # a wall running down one side
    return vol


def test_suggest_box_finds_a_phantom_nodule():
    from services.common import autoprompt, store
    vol, _ = store.make_phantom(diameter_mm=9.0)

    z, box, info = autoprompt.suggest_box(vol)
    assert z is not None and box is not None, info

    # The phantom sphere sits at the centre of the volume.
    assert abs(z - vol.shape[0] // 2) <= 3, f"slice {z} is not near the nodule"
    cx = (box[0] + box[2]) / 2
    cy = (box[1] + box[3]) / 2
    assert abs(cx - vol.shape[2] / 2) < 12 and abs(cy - vol.shape[1] / 2) < 12, \
        f"box centre {(cx, cy)} is not on the sphere"
    assert box[2] > box[0] and box[3] > box[1]


def test_suggest_box_refuses_a_volume_holding_only_chest_wall():
    """
    The failure this exists to prevent: a 3185 mm3 "nodule" that was the chest
    wall. Every component touching the border is dropped, so when the wall is
    all there is, the answer must be "nothing plausible" rather than the wall.
    """
    from services.common import autoprompt
    z, box, info = autoprompt.suggest_box(_slab_volume())

    assert z is None and box is None, \
        f"proposed {box} on a volume containing no nodule -- {info}"
    assert info.get("dropped_touching_border", 0) >= 1


def test_suggest_box_prefers_a_nodule_over_a_larger_chest_wall():
    """The wall is far bigger. Size must not win against a border component."""
    from services.common import autoprompt, store
    vol, _ = store.make_phantom(diameter_mm=8.0)
    vol = vol.copy()
    vol[:, :, :12] = 40                       # add a wall beside the nodule

    z, box, info = autoprompt.suggest_box(vol)
    assert box is not None, info
    cx = (box[0] + box[2]) / 2
    assert cx > 24, f"box centre x={cx} landed on the wall, not the nodule"
    assert info.get("dropped_touching_border", 0) >= 1


def test_suggest_box_agrees_with_and_without_scipy(monkeypatch):
    """
    The union-find fallback runs when scipy is absent. If it disagrees with
    scipy, the API and the worker would propose different boxes for the same
    case depending on which image answered.
    """
    from services.common import autoprompt, store
    vol, _ = store.make_phantom(diameter_mm=9.0)

    with_scipy = autoprompt.suggest_box(vol)
    monkeypatch.setattr(autoprompt, "_label", autoprompt._label_numpy)
    without = autoprompt.suggest_box(vol)

    assert with_scipy[0] == without[0]
    assert with_scipy[1] == without[1]


def test_centred_fallback_stays_inside_the_volume():
    from services.common import autoprompt
    z, box = autoprompt.centred_fallback((33, 64, 64), half=16)
    assert 0 <= z < 33
    assert box[0] >= 0 and box[1] >= 0 and box[2] <= 64 and box[3] <= 64
    assert box[2] > box[0] and box[3] > box[1]


# ------------------------------------------------------- shape plausibility

def _mask_with(shape, zslice, ys, xs):
    m = np.zeros(shape, dtype=bool)
    m[zslice, ys, xs] = True
    return m


def test_slab_shape_drops_confidence_to_low():
    """The chest-wall mask measured 1.77; no real nodule in 137 reached 1.6."""
    from src.report.sizing import flag_quality
    mask = _mask_with((8, 64, 64), slice(1, 7), slice(20, 40), slice(20, 40))
    m = {"n_voxels": 4000, "diameter_mean_mm": 17.4, "diameter_max_mm": 30.8}

    level, flags = flag_quality(mask, m, "solid")
    assert level == "low"
    assert any("slab-like" in f for f in flags), flags


def test_real_nodule_ratios_are_not_flagged_as_slabs():
    """Every ratio observed across the 137-nodule set must stay unflagged."""
    from src.report.sizing import flag_quality
    mask = _mask_with((8, 64, 64), slice(2, 6), slice(28, 36), slice(28, 36))

    for ratio in (1.02, 1.16, 1.38, 1.55):       # min, median, p95, max observed
        m = {"n_voxels": 500, "diameter_mean_mm": 8.0,
             "diameter_max_mm": 8.0 * ratio}
        _, flags = flag_quality(mask, m, "solid")
        assert not any("slab-like" in f for f in flags), \
            f"ratio {ratio} seen on real nodules was flagged: {flags}"


def test_mask_touching_the_roi_edge_is_flagged():
    from src.report.sizing import flag_quality
    mask = _mask_with((8, 64, 64), slice(2, 6), slice(0, 10), slice(20, 30))
    m = {"n_voxels": 500, "diameter_mean_mm": 8.0, "diameter_max_mm": 9.0}

    level, flags = flag_quality(mask, m, "solid")
    assert level in ("medium", "low")
    assert any("edge of the ROI" in f for f in flags), flags


def test_a_contained_mask_is_not_flagged_for_edges():
    from src.report.sizing import flag_quality
    mask = _mask_with((8, 64, 64), slice(0, 8), slice(20, 40), slice(20, 40))
    m = {"n_voxels": 500, "diameter_mean_mm": 8.0, "diameter_max_mm": 9.0}

    _, flags = flag_quality(mask, m, "solid")
    # Z spans the full depth deliberately: that is legitimate for a large
    # nodule in a tight crop and must not trigger the in-plane edge check.
    assert not any("edge of the ROI" in f for f in flags), flags


def test_flag_quality_survives_a_mask_without_indexing():
    """The edge check is defensive; a non-array mask must not crash sizing."""
    from src.report.sizing import flag_quality

    class NotAnArray:
        ndim = 3
        shape = (4, 4, 4)
        def reshape(self, *a):
            return np.zeros((4, 16), dtype=bool)

    level, flags = flag_quality(NotAnArray(), {
        "n_voxels": 500, "diameter_mean_mm": 8.0, "diameter_max_mm": 9.0}, "solid")
    assert level in ("high", "medium", "low")


# ------------------------------------------------------------ upload limits

def test_upload_cap_is_configurable_and_sane():
    import importlib
    from services.common import settings as s
    importlib.reload(s)
    assert 1 <= s.MAX_UPLOAD_MB <= 1024
    assert s.CORS_ORIGINS, "an empty origin list would reject every browser call"


def test_upload_cap_reads_the_environment(monkeypatch):
    import importlib
    monkeypatch.setenv("MAX_UPLOAD_MB", "7")
    monkeypatch.setenv("CORS_ORIGINS", "http://a.test, http://b.test")
    from services.common import settings as s
    importlib.reload(s)
    try:
        assert s.MAX_UPLOAD_MB == 7
        assert s.CORS_ORIGINS == ["http://a.test", "http://b.test"]
    finally:
        monkeypatch.undo()
        importlib.reload(s)


def test_device_default_is_auto(monkeypatch):
    import importlib
    monkeypatch.delenv("DEVICE", raising=False)
    from services.common import settings as s
    importlib.reload(s)
    try:
        assert s.DEVICE == "auto", \
            "the default must self-detect, or a laptop without a GPU crashes"
    finally:
        monkeypatch.undo()
        importlib.reload(s)


# ----------------------------------------------------------- worker status

def test_worker_status_reports_what_it_can_verify(monkeypatch):
    import importlib
    monkeypatch.setenv("CHECKPOINT", "/nonexistent/model.pt")
    from services.common import settings as s
    importlib.reload(s)
    from services.worker import tasks
    importlib.reload(tasks)
    try:
        st = tasks.worker_status()
        assert st["checkpoint_present"] is False
        assert st["device"] in ("cpu", "cuda")
        assert st["device_requested"] == s.DEVICE
        assert st["updated_at"]
    finally:
        monkeypatch.undo()
        importlib.reload(s)
        importlib.reload(tasks)


def test_checkpoint_present_detects_a_real_file(tmp_path, monkeypatch):
    import importlib
    ckpt = tmp_path / "model.pt"
    ckpt.write_bytes(b"not a real checkpoint, but it exists")
    monkeypatch.setenv("CHECKPOINT", str(ckpt))
    from services.common import settings as s
    importlib.reload(s)
    from services.worker import tasks
    importlib.reload(tasks)
    try:
        assert tasks.checkpoint_present() is True
    finally:
        monkeypatch.undo()
        importlib.reload(s)
        importlib.reload(tasks)


def test_publish_status_never_raises_without_redis(monkeypatch):
    """
    A status line is telemetry. If it can throw, it can take down the worker
    that is doing the real work.
    """
    import importlib
    monkeypatch.setenv("REDIS_URL", "redis://127.0.0.1:1/0")   # nothing there
    from services.common import settings as s
    importlib.reload(s)
    from services.worker import tasks
    importlib.reload(tasks)
    try:
        st = tasks.publish_status()          # must not raise
        assert st["device"] in ("cpu", "cuda")
    finally:
        monkeypatch.undo()
        importlib.reload(s)
        importlib.reload(tasks)


def test_the_worker_status_key_has_one_definition():
    """
    The key is a contract between two containers. Hardcoded at both ends, a
    rename stops the API from ever seeing a worker again and raises nothing.
    """
    import pathlib
    from services.common import settings
    from services.worker import tasks

    assert tasks.STATUS_KEY == settings.WORKER_STATUS_KEY

    root = pathlib.Path(__file__).resolve().parents[1]
    literal = settings.WORKER_STATUS_KEY
    offenders = []
    for path in sorted(root.glob("services/**/*.py")):
        if path.name == "settings.py":
            continue                 # the one place it is allowed to be spelt
        text = path.read_text(encoding="utf-8")
        if f'"{literal}"' in text or f"'{literal}'" in text:
            offenders.append(str(path.relative_to(root)))

    assert not offenders, \
        f"{literal!r} is hardcoded in {offenders}; use settings.WORKER_STATUS_KEY"


def test_text_files_are_read_as_utf8():
    """
    read_text() with no encoding uses the LOCALE encoding, which is cp1252 on
    a default Windows install and UTF-8 on Linux. Code written on Linux reads
    fine there and crashes on Windows the moment a file contains an em dash or
    an emoji -- which services/ui/app.py does, in its banner icons.

    Three of this project's files were read that way. The fix is cheap and the
    bug is invisible until someone runs it on the other platform, so the rule
    is: every text read states its encoding.
    """
    import pathlib
    import re

    root = pathlib.Path(__file__).resolve().parents[1]
    bad = []
    pattern = re.compile(r"\.(read_text|write_text)\(|(?<![\w.])open\(")

    for path in (sorted(root.glob("src/**/*.py")) + sorted(root.glob("services/**/*.py"))
                 + sorted(root.glob("scripts/**/*.py")) + sorted(root.glob("tests/**/*.py"))):
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not pattern.search(line):
                continue
            if "encoding=" in line or '"rb"' in line or "'rb'" in line \
                    or '"wb"' in line or "'wb'" in line or "BytesIO" in line:
                continue
            # A call split across lines carries its encoding on a later line.
            nxt = path.read_text(encoding="utf-8").splitlines()[n:n + 2]
            if any("encoding=" in l for l in nxt):
                continue
            bad.append(f"{path.relative_to(root).as_posix()}:{n}: {line.strip()}")

    assert not bad, "text I/O without an explicit encoding:\n  " + "\n  ".join(bad)


# ------------------------------------------------------- distribution shape

def test_pull_compose_builds_nothing_and_bakes_in_no_weights():
    """
    docker-compose.pull.yml is the file a recipient gets on its own. It must
    pull rather than build (they have no repository to build from), and the
    checkpoint must arrive as a read-only host mount.

    The licence is the reason: MedSAM2's weights are research-and-education-use
    only, so an image carrying them is not ours to publish. That constraint is
    easy to break by adding a COPY and never noticing.
    """
    import pathlib
    import yaml

    root = pathlib.Path(__file__).resolve().parents[1]
    path = root / "docker-compose.pull.yml"
    assert path.exists(), "the standalone compose file is missing"

    spec = yaml.safe_load(path.read_text(encoding="utf-8"))
    services = spec["services"]

    for name, svc in services.items():
        assert "build" not in svc, \
            f"{name} has a build context; a recipient has no repository to build from"
        assert "image" in svc, f"{name} has no image to pull"

    worker = services["worker"]
    mounts = [m for m in worker.get("volumes", []) if "checkpoints" in m]
    assert mounts, "the worker has no checkpoint mount, so it can never find the weights"
    assert all(m.endswith(":ro") for m in mounts), \
        f"the checkpoint mount must be read-only: {mounts}"

    # Every service the stack needs must be present, or it comes up broken.
    assert set(services) >= {"db", "redis", "api", "worker", "ui"}


def test_no_dockerfile_copies_the_weights():
    """A COPY of the checkpoint would put the weights inside a published image."""
    import pathlib
    import re

    root = pathlib.Path(__file__).resolve().parents[1]
    offenders = []
    for df in sorted((root / "docker").glob("*.Dockerfile")):
        for n, line in enumerate(df.read_text(encoding="utf-8").splitlines(), 1):
            if re.match(r"^\s*(COPY|ADD)\b", line) and re.search(
                    r"checkpoints?|\.pt\b|MedSAM2", line, re.I):
                offenders.append(f"{df.name}:{n}: {line.strip()}")

    assert not offenders, \
        "weights would be baked into an image (they are not ours to redistribute):\n  " \
        + "\n  ".join(offenders)
