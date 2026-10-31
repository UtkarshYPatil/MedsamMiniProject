"""
Case and mask storage, plus the PNG rendering the UI displays.

A "case" is one ROI volume: image_hu (Z, Y, X) int16 and spacing (z, y, x) mm.
That is deliberately the unit of exchange rather than a whole scan -- the model
consumes a 128x128x33 crop, so a case is about 1 MB where a full series is
200 MB. See docs/phase3_deployment.md.
"""
import io
import re
import uuid
import zipfile

import numpy as np

from backend.common import settings


def guard_decompression_bomb(raw: bytes, limit_mb: int, what: str = "upload"):
    """
    Refuse an archive that expands far beyond its transfer size.

    Both .npz and a DICOM .zip are zip archives, and the upload cap bounds the
    COMPRESSED bytes only. Zeros compress about 1000:1, so a 0.2 MB .npz
    expands to a 210 MB array and a 64 MB one to roughly 66 GB -- measured,
    not estimated. Reading it is what exhausts the machine, so the size has to
    be checked from the header before anything is read.

    The zip header's declared sizes are attacker-controlled, so this is a
    cheap first gate, not a guarantee; the real protection is that nothing
    is materialised until it passes.
    """
    try:
        zf = zipfile.ZipFile(io.BytesIO(raw))
    except zipfile.BadZipFile:
        return                      # not a zip; the caller's parser will refuse it
    total = sum(i.file_size for i in zf.infolist())
    limit = limit_mb * 1024 * 1024
    if total > limit:
        raise ValueError(
            f"{what} expands to {total / 1e6:.0f} MB, over the "
            f"{limit_mb} MB limit, from only {len(raw) / 1e6:.1f} MB "
            f"transferred. Refusing to read it.")


def new_id() -> str:
    return uuid.uuid4().hex[:12]


_ID_OK = re.compile(r"\A[A-Za-z0-9_-]{1,64}\Z")


def _safe_id(value: str, kind: str) -> str:
    """
    Ids come from the URL and are concatenated into a filesystem path.

    Today every route that reaches the filesystem looks the id up in the
    database first, so a traversal attempt never gets this far. That is a
    property of the callers, not of this function, and one refactor would
    remove it. Ids we mint are uuid4 hex, so this costs nothing.
    """
    if not isinstance(value, str) or not _ID_OK.match(value):
        raise ValueError(f"invalid {kind}: {value!r}")
    return value


def case_path(case_id: str):
    return settings.CASES_DIR / f"{_safe_id(case_id, 'case id')}.npz"


def mask_path(job_id: str):
    return settings.MASKS_DIR / f"{_safe_id(job_id, 'job id')}.npz"


def save_case(image_hu: np.ndarray, spacing, case_id=None, source="upload",
              reference: np.ndarray | None = None) -> str:
    """
    Persist an ROI volume. Returns the case id.

    `reference` is the radiologist consensus mask that LIDC ships with the
    annotated data. It is kept so the viewer can SHOW the reader where the
    lesion is -- a ground-glass nodule is close to invisible by eye, and
    without it there is nothing to aim the box at. It is reference data and
    never reaches the model: nothing in the segmentation path reads it.
    """
    if image_hu.ndim != 3:
        raise ValueError(f"image_hu must be 3D (Z, Y, X), got shape {image_hu.shape}")
    case_id = case_id or new_id()
    settings.ensure_dirs()
    arrays = dict(
        image_hu=image_hu.astype(np.int16),
        spacing=np.asarray(spacing, dtype=np.float32),
        source=np.asarray(source),
    )
    if reference is not None and np.asarray(reference).shape == image_hu.shape:
        arrays["reference"] = np.asarray(reference).astype(bool)
    np.savez_compressed(case_path(case_id), **arrays)
    return case_id


def load_reference(case_id: str):
    """The stored consensus mask, or None when the case carries no annotation."""
    p = case_path(case_id)
    if not p.exists():
        raise FileNotFoundError(f"no such case: {case_id}")
    with np.load(p, allow_pickle=False) as d:
        if "reference" not in d.files:
            return None
        return d["reference"].astype(bool)


def reference_from_npz(raw: bytes):
    """
    Pull the consensus mask out of an uploaded npz, if it has one.

    scripts/02_prepare_data.py writes it as 'mask'. Never raises: a case
    without an annotation is the normal situation, not an error.
    """
    try:
        with np.load(io.BytesIO(raw), allow_pickle=False) as d:
            for name in ("mask", "reference", "gt"):
                if name in d.files:
                    m = np.asarray(d[name])
                    return m.astype(bool) if m.ndim == 3 else None
    except Exception:
        pass
    return None


def load_case(case_id: str):
    p = case_path(case_id)
    if not p.exists():
        raise FileNotFoundError(f"no such case: {case_id}")
    d = np.load(p, allow_pickle=False)
    return d["image_hu"].astype(np.float32), d["spacing"]


def read_npz_upload(raw: bytes):
    """
    Parse an uploaded .npz into (image_hu, spacing).

    Accepts the same files scripts/02_prepare_data.py writes, so a processed
    ROI can be dropped straight in. 'spacing' defaults to 1 mm isotropic when
    absent, which keeps millimetre outputs honest rather than silently wrong:
    they will be wrong by a known factor the caller can see.
    """
    guard_decompression_bomb(raw, settings.MAX_DECOMPRESSED_MB, "this .npz")
    with np.load(io.BytesIO(raw), allow_pickle=False) as d:
        keys = set(d.files)
        for name in ("image_hu", "image", "vol", "volume"):
            if name in keys:
                image = np.asarray(d[name])
                break
        else:
            raise ValueError(
                f"no image array in the npz (looked for image_hu/image/vol/volume); "
                f"found {sorted(keys)}")
        spacing = np.asarray(d["spacing"], dtype=np.float32) if "spacing" in keys \
            else np.asarray([1.0, 1.0, 1.0], dtype=np.float32)
    if image.ndim != 3:
        raise ValueError(f"image must be 3D (Z, Y, X), got {image.shape}")
    return image.astype(np.int16), spacing


def save_mask(job_id: str, mask: np.ndarray, prompt_z: int, box):
    settings.ensure_dirs()
    np.savez_compressed(
        mask_path(job_id),
        mask=mask.astype(bool),
        prompt_z=np.asarray(int(prompt_z)),
        box=np.asarray(box, dtype=np.float32),
    )


def load_mask(job_id: str):
    p = mask_path(job_id)
    if not p.exists():
        raise FileNotFoundError(f"no mask for job {job_id}")
    d = np.load(p, allow_pickle=False)
    return d["mask"], int(d["prompt_z"]), d["box"]


# ------------------------------------------------------------------ render

def _window_u8(slice_hu: np.ndarray) -> np.ndarray:
    lo = settings.WINDOW_LEVEL - settings.WINDOW_WIDTH / 2
    hi = settings.WINDOW_LEVEL + settings.WINDOW_WIDTH / 2
    x = np.clip((slice_hu - lo) / (hi - lo), 0, 1)
    return (x * 255).astype(np.uint8)


def _outline(mask2d: np.ndarray) -> np.ndarray:
    """Boundary pixels of a boolean mask, without pulling in scipy."""
    if not mask2d.any():
        return np.zeros_like(mask2d, dtype=bool)
    m = mask2d.astype(bool)
    eroded = m.copy()
    eroded[1:, :] &= m[:-1, :]
    eroded[:-1, :] &= m[1:, :]
    eroded[:, 1:] &= m[:, :-1]
    eroded[:, :-1] &= m[:, 1:]
    return m & ~eroded


def render_png(slice_hu: np.ndarray, mask2d=None, box=None,
               mask_rgb=(255, 77, 77), box_rgb=(255, 212, 0)) -> bytes:
    """Windowed slice as PNG, optionally with a mask outline and prompt box."""
    from PIL import Image, ImageDraw

    g = _window_u8(slice_hu)
    rgb = np.dstack([g, g, g])

    if mask2d is not None and np.asarray(mask2d).any():
        edge = _outline(np.asarray(mask2d))
        rgb[edge] = mask_rgb

    img = Image.fromarray(rgb, mode="RGB")

    if box is not None:
        x0, y0, x1, y1 = (float(v) for v in box)
        ImageDraw.Draw(img).rectangle([x0, y0, x1, y1], outline=box_rgb, width=1)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ----------------------------------------------------------------- phantom

def make_phantom(shape=(33, 128, 128), diameter_mm=9.0, spacing=(1.0, 0.7, 0.7),
                 hu_nodule=-120.0, hu_lung=-850.0, noise=40.0, seed=42):
    """
    A synthetic ROI with one spherical 'nodule'.

    This exists so `docker compose up` is demonstrable on a machine with no
    dataset: it gives the UI something to segment end to end. It is a phantom,
    not a patient, and the API labels any case made this way as source=phantom
    so a measurement taken from it can never be mistaken for a real one.
    """
    rng = np.random.default_rng(seed)
    Z, Y, X = shape
    sz, sy, sx = spacing

    zz = (np.arange(Z) - (Z - 1) / 2) * sz
    yy = (np.arange(Y) - (Y - 1) / 2) * sy
    xx = (np.arange(X) - (X - 1) / 2) * sx
    r = np.sqrt(zz[:, None, None] ** 2 + yy[None, :, None] ** 2 + xx[None, None, :] ** 2)

    vol = np.full(shape, hu_lung, dtype=np.float32)
    inside = r <= (diameter_mm / 2.0)
    vol[inside] = hu_nodule
    # soften the rim so the edge is not a perfect step, as a real nodule is not
    rim = (r > diameter_mm / 2.0) & (r <= diameter_mm / 2.0 + max(sy, sx))
    vol[rim] = (hu_nodule + hu_lung) / 2.0
    vol += rng.normal(0, noise, size=shape).astype(np.float32)

    return np.clip(vol, -1024, 3071).astype(np.int16), np.asarray(spacing, dtype=np.float32)
