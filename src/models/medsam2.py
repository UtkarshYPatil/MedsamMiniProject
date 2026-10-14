"""
MedSAM2 inference wrapper. The model is FROZEN -- no training, ever.

VERIFIED against the official reference implementation
(bowang-lab/MedSAM2 @ 332f30d, `medsam2_infer_3D_CT.py`) on 2026-10-14.
Every call below matches that script. Notable corrections made during
verification -- the earlier guesses were all wrong:

    build_sam2_video_predictor      ->  build_sam2_video_predictor_npz
    init_state_from_numpy(vol)      ->  init_state(images, video_h, video_w)
    single propagate_in_video pass  ->  forward pass, reset_state,
                                        re-add box, reverse pass

Preprocessing is not optional. MedSAM2 expects, in this exact order:
    uint8 volume (D,H,W) -> RGB resize to 512x512 -> /255
    -> subtract ImageNet mean -> divide by ImageNet std
The bounding box stays in ORIGINAL pixel coordinates; the predictor rescales
it internally because init_state was told the original height and width.

Requires the sam2 package from the MedSAM2 repo:
    git clone https://github.com/bowang-lab/MedSAM2.git external/MedSAM2
    pip install -e external/MedSAM2
"""
from pathlib import Path

import numpy as np

IMG_SIZE = 512
IMG_MEAN = (0.485, 0.456, 0.406)
IMG_STD = (0.229, 0.224, 0.225)


def resize_grayscale_to_rgb_and_resize(array, image_size=IMG_SIZE):
    """(D,H,W) uint8 -> (D,3,size,size) float. Mirrors the reference script."""
    from PIL import Image
    d = array.shape[0]
    out = np.zeros((d, 3, image_size, image_size), dtype=np.float32)
    for i in range(d):
        img = Image.fromarray(array[i].astype(np.uint8)).convert("RGB")
        img = img.resize((image_size, image_size))
        out[i] = np.array(img).transpose(2, 0, 1)
    return out


def largest_connected_component(mask):
    """Keep only the biggest 3D blob -- the reference script's getLargestCC."""
    from skimage import measure
    if not mask.any():
        return mask
    labels = measure.label(mask)
    counts = np.bincount(labels.flat)
    if len(counts) < 2:
        return mask
    return labels == (np.argmax(counts[1:]) + 1)


class MedSAM2Segmenter:
    """Frozen MedSAM2. Build once, reuse for every nodule."""

    def __init__(self, checkpoint, cfg_name="configs/sam2.1_hiera_t512.yaml",
                 device="cuda"):
        self.checkpoint = str(checkpoint)
        self.cfg_name = str(cfg_name)
        self._device = device
        self._predictor = None

    # -- lifecycle ---------------------------------------------------------

    @property
    def device(self):
        import torch
        if self._device == "cuda" and not torch.cuda.is_available():
            return "cpu"
        return self._device

    def load(self):
        from sam2.build_sam import build_sam2_video_predictor_npz

        if not Path(self.checkpoint).exists():
            raise FileNotFoundError(
                f"checkpoint not found: {self.checkpoint}\n"
                "run: python scripts/download_checkpoint.py")

        self._predictor = build_sam2_video_predictor_npz(
            self.cfg_name, self.checkpoint)
        for p in self._predictor.parameters():
            p.requires_grad_(False)
        self._predictor.eval()
        return self

    def _ensure(self):
        if self._predictor is None:
            self.load()
        return self._predictor

    # -- preprocessing -----------------------------------------------------

    def _prepare(self, vol_u8):
        """uint8 (D,H,W) -> normalised tensor on the model's device."""
        import torch
        assert vol_u8.max() < 256, "input must be uint8 in [0, 255]"
        x = resize_grayscale_to_rgb_and_resize(vol_u8, IMG_SIZE) / 255.0
        x = torch.from_numpy(x).to(self.device, dtype=torch.float32)
        mean = torch.tensor(IMG_MEAN, dtype=torch.float32,
                            device=self.device)[:, None, None]
        std = torch.tensor(IMG_STD, dtype=torch.float32,
                           device=self.device)[:, None, None]
        return (x - mean) / std

    # -- inference ---------------------------------------------------------

    def segment_volume(self, vol_u8, prompt_idx, box, propagate=True,
                       keep_largest=True):
        """
        vol_u8     : (D, H, W) uint8, already lung-windowed
        prompt_idx : index of the slice carrying the prompt
        box        : [x0, y0, x1, y1] in ORIGINAL pixel coordinates
        returns    : (D, H, W) bool
        """
        import torch

        pred = self._ensure()
        D, H, W = vol_u8.shape
        images = self._prepare(vol_u8)
        box = np.asarray(box, dtype=np.float32)
        out = np.zeros((D, H, W), dtype=bool)

        use_cuda = self.device == "cuda"
        autocast = (torch.autocast("cuda", dtype=torch.bfloat16) if use_cuda
                    else _NullCtx())

        with torch.inference_mode(), autocast:
            state = pred.init_state(images, H, W)

            directions = [False, True] if propagate else [False]
            for i, reverse in enumerate(directions):
                if i > 0:
                    pred.reset_state(state)
                pred.add_new_points_or_box(
                    inference_state=state,
                    frame_idx=int(prompt_idx),
                    obj_id=1,
                    box=box,
                )
                for frame_idx, _obj_ids, logits in pred.propagate_in_video(
                        state, reverse=reverse):
                    out[frame_idx, (logits[0] > 0.0).cpu().numpy()[0]] = True

            pred.reset_state(state)

        if keep_largest and out.any():
            out = largest_connected_component(out)
        return out


class _NullCtx:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class DummySegmenter:
    """
    Offline stand-in so the pipeline and the UI can be developed and tested
    without a GPU or the checkpoint. Produces an ellipsoid inside the box.
    Never use for reported results -- scripts refuse it unless --dry-run.
    """

    def __init__(self, *a, **k):
        pass

    def load(self):
        return self

    def segment_volume(self, vol_u8, prompt_idx, box, propagate=True,
                       keep_largest=True):
        D, H, W = vol_u8.shape[:3]
        x0, y0, x1, y1 = [float(v) for v in box]
        cy, cx = (y0 + y1) / 2, (x0 + x1) / 2
        ry, rx = max(1.0, (y1 - y0) / 2), max(1.0, (x1 - x0) / 2)
        yy, xx = np.ogrid[:H, :W]

        out = np.zeros((D, H, W), bool)
        span = max(1, int(min(ry, rx)))
        for d in range(D):
            off = abs(d - prompt_idx)
            if off > span:
                continue
            shrink = np.sqrt(max(1e-6, 1 - (off / (span + 1)) ** 2))
            out[d] = (((yy - cy) ** 2) / (ry * shrink) ** 2
                      + ((xx - cx) ** 2) / (rx * shrink) ** 2) <= 1.0
        return out


def build(checkpoint, cfg_name="configs/sam2.1_hiera_t512.yaml",
          device="cuda", dry_run=False):
    return (DummySegmenter() if dry_run
            else MedSAM2Segmenter(checkpoint, cfg_name, device))
