"""
DICOM import and the ROI crop that makes a whole series usable.

Four things in dicomseries.py are silent when wrong, and all four are tested
here against a synthetic series: rescale to HU, slice ordering by geometry,
slice spacing from positions, and (Z, Y, X) axis order.

    python -m pytest tests/test_dicom.py -v
"""
import io
import zipfile

import numpy as np
import pytest

pydicom = pytest.importorskip("pydicom", reason="pip install pydicom")

from backend.common import dicomseries as D
from backend.common import roi


# ----------------------------------------------------------------- fixtures

def _slice(i, *, rows=16, cols=20, dz=1.25, sy=0.7, sx=0.6, slope=1.0,
           intercept=-1024.0, modality="CT", uid=None, z=None, mark=None):
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    ds = Dataset()
    ds.file_meta = FileMetaDataset()
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
    ds.file_meta.MediaStorageSOPInstanceUID = generate_uid()
    ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
    ds.SOPInstanceUID = generate_uid()
    ds.SeriesInstanceUID = uid or "1.2.3.4"
    ds.Modality = modality
    ds.Rows, ds.Columns = rows, cols
    ds.PixelSpacing = [sy, sx]              # [row, col] = [y, x]
    ds.SliceThickness = dz
    ds.ImagePositionPatient = [0.0, 0.0, (i * dz) if z is None else z]
    ds.InstanceNumber = i + 1
    ds.RescaleSlope, ds.RescaleIntercept = slope, intercept
    ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
    ds.BitsAllocated, ds.BitsStored, ds.HighBit = 16, 16, 15
    ds.PixelRepresentation = 0

    a = np.full((rows, cols), i + 100, dtype=np.uint16)
    if mark:
        a[mark[0], mark[1]] = 900
    ds.PixelData = a.tobytes()

    buf = io.BytesIO()
    pydicom.dcmwrite(buf, ds, enforce_file_format=True)
    return buf.getvalue()


def _zip(blobs, reverse=False, extra=()):
    """Names are IM0..IMn so a filename sort disagrees with slice order."""
    buf = io.BytesIO()
    items = list(enumerate(blobs))
    if reverse:
        items = items[::-1]
    with zipfile.ZipFile(buf, "w") as z:
        for i, b in items:
            z.writestr(f"series/IM{i}.dcm", b)
        for name, data in extra:
            z.writestr(name, data)
    return buf.getvalue()


# ------------------------------------------------------ the four silent ones

def test_pixel_values_are_rescaled_to_hounsfield_units():
    """HU = slope*raw + intercept. Skipping it puts a -600 HU nodule at +424."""
    vol, _, _ = D.read_zip(_zip([_slice(i) for i in range(6)]))
    assert [int(vol[i, 0, 0]) for i in range(6)] == [i + 100 - 1024 for i in range(6)]


def test_slices_are_ordered_by_geometry_not_filename():
    vol, _, _ = D.read_zip(_zip([_slice(i) for i in range(6)], reverse=True))
    assert [int(vol[i, 0, 0]) for i in range(6)] == [i + 100 - 1024 for i in range(6)]


def test_spacing_comes_from_slice_positions_in_zyx_order():
    _, spacing, info = D.read_zip(
        _zip([_slice(i, dz=2.5, sy=0.8, sx=0.5) for i in range(5)]))
    assert np.allclose(spacing, [2.5, 0.8, 0.5]), spacing
    assert info["spacing_z_source"] == "ImagePositionPatient"


def test_axis_order_is_z_y_x():
    """pixel_array is (rows, cols) = (Y, X); a transpose here is invisible."""
    vol, _, _ = D.read_zip(_zip([_slice(i, mark=(3, 7)) for i in range(4)]))
    zz, yy, xx = np.where(vol == 900 - 1024 + 0)
    assert (int(yy[0]), int(xx[0])) == (3, 7)


def test_a_series_with_every_slice_at_one_position_is_refused():
    """
    The slice-count check runs before duplicates are dropped, so this used to
    collapse to a single slice and be measured as if it were a stack.
    """
    blobs = [_slice(i, z=0.0, dz=3.0) for i in range(4)]
    with pytest.raises(D.DicomError, match="does not describe a volume"):
        D.read_zip(_zip(blobs))


# ------------------------------------------------------------ refusals

def test_non_ct_is_refused():
    with pytest.raises(D.DicomError, match="not CT"):
        D.read_zip(_zip([_slice(i, modality="MR") for i in range(5)]))


def test_a_single_slice_is_refused():
    with pytest.raises(D.DicomError, match="volume"):
        D.read_zip(_zip([_slice(0)]))


def test_a_zip_with_no_dicom_is_refused():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("notes.txt", b"hello")
    with pytest.raises(D.DicomError, match="no DICOM images"):
        D.read_zip(buf.getvalue())


def test_not_a_zip_is_refused():
    with pytest.raises(D.DicomError, match="readable .zip"):
        D.read_zip(b"this is not a zip file")


# ------------------------------------------------------------- robustness

def test_non_image_files_in_the_zip_are_skipped():
    raw = _zip([_slice(i) for i in range(5)],
               extra=[("DICOMDIR", b"junk"), ("README.txt", b"hi")])
    _, _, info = D.read_zip(raw)
    assert info["slices_used"] == 5
    assert info["files_skipped"] == 2


def test_duplicate_slice_positions_are_dropped_and_counted():
    blobs = [_slice(i) for i in range(5)] + [_slice(5, z=0.0)]
    _, _, info = D.read_zip(_zip(blobs))
    assert info["duplicate_positions_dropped"] == 1
    assert info["slices_used"] == 5, "the count must be after the drop"


def test_the_largest_ct_series_wins_when_several_are_present():
    blobs = ([_slice(i, uid="1.1") for i in range(7)]
             + [_slice(i, uid="2.2") for i in range(3)])
    _, _, info = D.read_zip(_zip(blobs))
    assert info["series_found"] == 2
    assert info["slices_used"] == 7
    assert "note" in info


def test_an_unrescaled_series_is_flagged():
    """Air is near -1000. Nothing that low means HU were never applied."""
    _, _, info = D.read_zip(_zip([_slice(i, intercept=0.0) for i in range(5)]))
    assert "warning" in info and "-1000" in info["warning"]


def test_describe_leaks_no_identifiers():
    _, _, info = D.read_zip(_zip([_slice(i) for i in range(5)]))
    text = D.describe(info)
    assert "slices" in text and "mm" in text
    for leak in ("PatientName", "PatientID", "1.2.3.4", "AccessionNumber"):
        assert leak not in text


# ------------------------------------------------------------- the ROI crop

def test_an_existing_roi_is_never_cropped():
    """
    33x128x128 is exactly what scripts/02_prepare_data.py writes. An earlier
    version used a 2*depth window and silently clipped a slice off every one.
    """
    shape = (33, 128, 128)
    assert roi.needs_crop(shape) is False
    vol = np.zeros(shape, np.float32)
    sub, z, box, origin = roi.crop_around(vol, 16, [50, 50, 70, 70])
    assert sub.shape == shape
    assert origin == (0, 0, 0) and z == 16 and box == [50, 50, 70, 70]


def test_a_full_series_crops_to_the_validated_geometry():
    vol = np.zeros((250, 512, 512), np.float32)
    assert roi.needs_crop(vol.shape) is True
    sub, _, _, _ = roi.crop_around(vol, 120, [395, 295, 405, 305])
    assert sub.shape == (33, 128, 128)


def test_the_crop_keeps_the_nodule_inside_the_translated_box():
    vol = np.zeros((250, 512, 512), np.float32)
    vol[120, 300, 400] = 999
    sub, sub_z, sub_box, _ = roi.crop_around(vol, 120, [392, 292, 408, 308])
    zz, yy, xx = np.where(sub == 999)
    assert sub_box[0] <= xx[0] <= sub_box[2]
    assert sub_box[1] <= yy[0] <= sub_box[3]
    assert int(zz[0]) == sub_z


def test_crop_and_paste_round_trip_to_the_original_coordinates():
    vol = np.zeros((120, 400, 400), np.float32)
    vol[77, 123, 321] = 1
    sub, _, _, origin = roi.crop_around(vol, 77, [316, 118, 326, 128])
    full = roi.paste(sub == 1, vol.shape, origin)
    zz, yy, xx = np.where(full)
    assert (int(zz[0]), int(yy[0]), int(xx[0])) == (77, 123, 321)


@pytest.mark.parametrize("z,box", [
    (0, [0, 0, 10, 10]),                 # top corner
    (249, [500, 500, 511, 511]),         # bottom corner
    (125, [250, 250, 260, 260]),         # middle
])
def test_the_crop_geometry_is_constant_at_the_edges(z, box):
    """Clipping at an edge would hand the model a different input shape."""
    vol = np.zeros((250, 512, 512), np.float32)
    sub, _, _, origin = roi.crop_around(vol, z, box)
    assert sub.shape == (33, 128, 128)
    assert all(o >= 0 for o in origin)


def test_a_volume_smaller_than_the_window_is_returned_whole():
    vol = np.zeros((9, 40, 40), np.float32)
    sub, _, _, origin = roi.crop_around(vol, 4, [10, 10, 20, 20])
    assert sub.shape == (9, 40, 40) and origin == (0, 0, 0)
