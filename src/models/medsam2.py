"""
MedSAM2 wrapper. The model is FROZEN -- inference only, no training.

MedSAM2 is built on SAM 2.1 and is volumetric: prompt one slice, propagate the
mask through adjacent slices. Requires the sam2 package from
https://github.com/bowang-lab/MedSAM2 (pip install -e external/MedSAM2).
"""
from pathlib import Path
import numpy as np
import torch


class MedSAM2Segmenter:
    def __init__(self, checkpoint, model_cfg, device="cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.checkpoint = str(checkpoint)
        self.model_cfg = str(model_cfg)
        self._predictor = None

    def load(self):
        """Lazy load so importing this module never needs a GPU."""
        from sam2.build_sam import build_sam2_video_predictor
        self._predictor = build_sam2_video_predictor(
            self.model_cfg, self.checkpoint, device=self.device
        )
        for p in self._predictor.parameters():
            p.requires_grad = False
        self._predictor.eval()
        return self

    @torch.inference_mode()
    def segment_volume(self, volume_rgb, prompt_slice_idx, box):
        """
        volume_rgb : (D, H, W, 3) uint8
        box        : [x0, y0, x1, y1] on slice prompt_slice_idx
        returns    : (D, H, W) boolean mask
        """
        if self._predictor is None:
            self.load()

        state = self._predictor.init_state_from_numpy(volume_rgb)
        self._predictor.add_new_points_or_box(
            inference_state=state,
            frame_idx=prompt_slice_idx,
            obj_id=1,
            box=np.array(box, dtype=np.float32),
        )

        out = np.zeros(volume_rgb.shape[:3], dtype=bool)
        for fwd in (False, True):
            for idx, _ids, logits in self._predictor.propagate_in_video(
                state, reverse=fwd
            ):
                out[idx] = (logits[0] > 0).cpu().numpy().squeeze()
        return out

    @torch.inference_mode()
    def segment_slice(self, slice_rgb, box):
        """Single-slice fallback, for comparison against the volume path."""
        vol = slice_rgb[None, ...]
        return self.segment_volume(vol, 0, box)[0]
