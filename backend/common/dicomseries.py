"""
Turn an uploaded DICOM series into the (Z, Y, X) HU volume the rest of the
stack already speaks.

WHY THIS IS NOT JUST "READ THE PIXELS"
--------------------------------------
Four things have to be right or every millimetre we report afterwards is
wrong, and three of them are silent when they are wrong:

1. RESCALE. Stored pixel values are not Hounsfield units. HU = slope * raw +
   intercept, per slice. Skip it and a nodule at -600 HU reads as +424, the
   density classifier inverts, and nothing looks obviously broken.

2. SLICE ORDER. Files in a zip arrive in whatever order the filesystem gives.
   Sorting by filename puts IM10 before IM2. We sort by the z component of
   ImagePositionPatient, which is the scanner's own geometry.

3. SLICE SPACING. SliceThickness is what the scanner reconstructed; the gap
   between slice centres is what the volume actually has, and they differ
   whenever there is overlap or a gap. Volume is spacing_z * spacing_y *
   spacing_x per voxel, so taking the wrong one scales every volume we report.
   We measure it from consecutive positions and fall back to SliceThickness.

4. AXIS ORDER. pixel_array is (rows, cols) = (Y, X). Stacking gives (Z, Y, X),
   which is what src/ and backend/ use throughout. PixelSpacing is
   [row, col] = [y, x], so spacing is (dz, PixelSpacing[0], PixelSpacing[1]).
   Getting this backwards on a non-isotropic scan is a silent error -- the
   image still looks like a chest.

PATIENT DATA
------------
We read pixels and geometry. Nothing that identifies a patient is returned or
stored -- not name, ID, birth date, accession or institution. `describe()`
returns only what a radiologist needs to confirm the right series loaded.

This is research and education software operating on LIDC-IDRI, a public
de-identified dataset. It is not a medical device and real patient scans do
not belong in it.
"""
import io
import zipfile

import numpy as np

# A CT slice is 512x512 and a series is a few hundred of them. These bounds
# reject something that is not a chest CT before it becomes a memory problem.
MIN_SLICES = 3
MAX_SLICES = 1024
MAX_IN_PLANE = 1024


class DicomError(ValueError):
    """Raised with a message meant to be shown to the person uploading."""


def _read_one(data: bytes):
    import pydicom
    try:
        return pydicom.dcmread(io.BytesIO(data), force=False)
    except Exception:
        return None


def _z_of(ds) -> float:
    """Slice position along the scanner's z axis."""
    ipp = getattr(ds, "ImagePositionPatient", None)
    if ipp is not None and len(ipp) == 3:
        return float(ipp[2])
    loc = getattr(ds, "SliceLocation", None)
    if loc is not None:
        return float(loc)
    num = getattr(ds, "InstanceNumber", None)
    if num is not None:
        return float(num)
    raise DicomError("a slice has no position information "
                     "(ImagePositionPatient, SliceLocation or InstanceNumber)")


def _to_hu(ds) -> np.ndarray:
    """Stored values -> Hounsfield units, per slice."""
    arr = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
    intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
    return arr * slope + intercept


def read_zip(raw: bytes):
    """
    (volume_hu (Z, Y, X) int16, spacing (z, y, x) float32, info dict)

    Accepts a .zip of one DICOM series, nested folders included. When several
    series are present the largest CT series wins and `info` says so, because
    a scan export routinely carries scouts and dose reports alongside it.
    """
    try:
        zf = zipfile.ZipFile(io.BytesIO(raw))
    except zipfile.BadZipFile:
        raise DicomError("that is not a readable .zip file")

    names = [n for n in zf.namelist()
             if not n.endswith("/") and not n.split("/")[-1].startswith(".")]
    if not names:
        raise DicomError("the zip is empty")

    slices, unreadable = [], 0
    for n in names:
        try:
            data = zf.read(n)
        except Exception:
            unreadable += 1
            continue
        ds = _read_one(data)
        if ds is None or not hasattr(ds, "PixelData"):
            unreadable += 1          # DICOMDIR, reports, stray files
            continue
        slices.append(ds)

    if not slices:
        raise DicomError(
            f"no DICOM images found in the zip ({len(names)} files examined). "
            "Zip the folder containing the .dcm slices, not a parent folder of "
            "unrelated files.")

    return _build(slices, extra_info={"files_in_zip": len(names),
                                      "files_skipped": unreadable})


def read_files(blobs):
    """Same as read_zip for a list of raw file contents already in hand."""
    slices = [ds for ds in (_read_one(b) for b in blobs)
              if ds is not None and hasattr(ds, "PixelData")]
    if not slices:
        raise DicomError("none of the uploaded files are DICOM images")
    return _build(slices, extra_info={"files_uploaded": len(blobs)})


def _build(slices, extra_info=None):
    info = dict(extra_info or {})

    # --- one series only -------------------------------------------------
    by_series = {}
    for ds in slices:
        uid = str(getattr(ds, "SeriesInstanceUID", "unknown"))
        by_series.setdefault(uid, []).append(ds)

    def series_rank(item):
        _, group = item
        is_ct = sum(1 for d in group
                    if str(getattr(d, "Modality", "")).upper() == "CT")
        return (is_ct > 0, len(group))

    uid, chosen = max(by_series.items(), key=series_rank)
    info["series_found"] = len(by_series)
    if len(by_series) > 1:
        info["note"] = (f"{len(by_series)} series in the zip; used the largest "
                        f"CT series ({len(chosen)} slices)")

    modality = str(getattr(chosen[0], "Modality", "")).upper()
    info["modality"] = modality or "unknown"
    if modality and modality != "CT":
        raise DicomError(
            f"this is a {modality} series, not CT. Every threshold here is in "
            "Hounsfield units, which only CT produces, so the measurements "
            "would be meaningless.")

    if len(chosen) < MIN_SLICES:
        raise DicomError(
            f"only {len(chosen)} slice(s) in the series. Measuring a nodule "
            "needs a volume; upload the whole series.")
    if len(chosen) > MAX_SLICES:
        raise DicomError(f"{len(chosen)} slices exceeds the {MAX_SLICES} limit")

    # --- order by geometry, not by filename ------------------------------
    chosen.sort(key=_z_of)
    zs = np.array([_z_of(d) for d in chosen], dtype=np.float64)

    # Duplicate positions mean the export contains the same slice twice, or
    # two interleaved acquisitions. Either way the volume would be wrong.
    dz_all = np.diff(zs)
    keep = np.concatenate([[True], dz_all != 0])
    if not keep.all():
        info["duplicate_positions_dropped"] = int((~keep).sum())
        chosen = [d for d, k in zip(chosen, keep) if k]
        zs = zs[keep]
        dz_all = np.diff(zs)

        # Re-check the count: the first check ran before this drop, so a
        # series whose slices all share one position passed it and would
        # otherwise be built into a one-slice "volume" and measured as if it
        # were a stack.
        if len(chosen) < MIN_SLICES:
            raise DicomError(
                f"after removing {info['duplicate_positions_dropped']} slice(s) "
                f"at duplicate positions, only {len(chosen)} remain. The series "
                "does not describe a volume.")

    info["slices_used"] = len(chosen)      # after dropping duplicates

    # --- geometry --------------------------------------------------------
    first = chosen[0]
    rows, cols = int(first.Rows), int(first.Columns)
    if rows > MAX_IN_PLANE or cols > MAX_IN_PLANE:
        raise DicomError(f"slices are {rows}x{cols}, larger than the "
                         f"{MAX_IN_PLANE} limit")

    px = getattr(first, "PixelSpacing", None)
    if px is None or len(px) != 2:
        raise DicomError("the series has no PixelSpacing, so millimetres "
                         "cannot be computed from it")
    sy, sx = float(px[0]), float(px[1])      # [row, col] = [y, x]

    if len(dz_all) and np.all(np.isfinite(dz_all)):
        dz = float(np.median(np.abs(dz_all)))
        spread = float(np.max(np.abs(dz_all)) - np.min(np.abs(dz_all)))
        if dz > 0 and spread > 0.01 * dz:
            # Irregular spacing: a single number cannot describe this volume,
            # so volumes computed from it carry an error we cannot bound.
            info["irregular_spacing_mm"] = round(spread, 3)
    else:
        dz = 0.0
    if dz <= 0:
        dz = float(getattr(first, "SliceThickness", 0) or 0)
        info["spacing_z_source"] = "SliceThickness"
    else:
        info["spacing_z_source"] = "ImagePositionPatient"
    if dz <= 0:
        raise DicomError("slice spacing could not be determined from the series")

    thickness = float(getattr(first, "SliceThickness", 0) or 0)
    if thickness and abs(thickness - dz) > 0.25 * max(thickness, dz):
        info["thickness_vs_spacing"] = (
            f"SliceThickness {thickness:g} mm but slices are {dz:.2f} mm apart "
            "(overlapping or gapped reconstruction)")

    # --- pixels ----------------------------------------------------------
    vol = np.empty((len(chosen), rows, cols), dtype=np.int16)
    for i, ds in enumerate(chosen):
        if int(ds.Rows) != rows or int(ds.Columns) != cols:
            raise DicomError("slices in this series are not all the same size")
        try:
            hu = _to_hu(ds)
        except Exception as e:
            raise DicomError(
                f"could not decode slice {i} ({e}). Compressed DICOM needs an "
                "extra codec; re-export the series as uncompressed.")
        vol[i] = np.clip(hu, -32768, 32767).astype(np.int16)

    lo, hi = int(vol.min()), int(vol.max())
    info["hu_range"] = [lo, hi]
    if lo > -500:
        # Air in the lungs sits near -1000. Nothing that low means the rescale
        # was not applied, and every HU-based number would be shifted.
        info["warning"] = (
            f"lowest value is {lo} HU; air should be near -1000. The rescale "
            "may not have been applied, which would shift every measurement.")

    spacing = np.array([dz, sy, sx], dtype=np.float32)
    info["shape"] = [int(x) for x in vol.shape]
    info["spacing_mm"] = [round(float(s), 4) for s in spacing]
    return vol, spacing, info


def describe(info: dict) -> str:
    """One line a radiologist can check the import against. No identifiers."""
    z, y, x = info.get("shape", ["?"] * 3)
    sp = info.get("spacing_mm", ["?"] * 3)
    return (f"{z} slices of {y}x{x}, "
            f"{sp[0]} mm apart, {sp[1]}x{sp[2]} mm in plane")
