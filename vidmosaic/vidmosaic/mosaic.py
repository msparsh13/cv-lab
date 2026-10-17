"""Panorama compositing.

Phase 4 implements feather blending only. Temporal median, sharpest-frame
selection and robust fallbacks are Phase 5.

Feather blending: every valid pixel of a warped frame gets the weight
``w = distance to the nearest invalid pixel`` (Euclidean distance transform of
the validity mask), so weights fall linearly to ~0 at the frame border. The
output is the weight-averaged colour over all frames covering the pixel:

    out = sum_k w_k * I_k / sum_k w_k

* A pixel covered by one frame equals that frame's pixel exactly.
* A pixel covered by no frame is invalid (black in the image, False in the mask).
* Streaming: only two float32 accumulators (colour sum, weight sum) are kept,
  16 bytes per canvas pixel, regardless of the number of frames.

Known limitation: averaging blurs anything that is not perfectly aligned
(parallax, moving objects, exposure steps). Moving objects appear as faint
ghosts; the background mosaic (Phase 5) is the tool for removing them.
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

from .warping import WarpedFrame


class FeatherBlender:
    def __init__(self, canvas_size: Tuple[int, int]):
        w, h = canvas_size
        self.size = (w, h)
        self.acc = np.zeros((h, w, 3), np.float32)
        self.wsum = np.zeros((h, w), np.float32)
        self.n_frames = 0

    @staticmethod
    def bytes_needed(canvas_size: Tuple[int, int]) -> int:
        return canvas_size[0] * canvas_size[1] * 16

    def add(self, wf: WarpedFrame) -> None:
        h, w = wf.mask.shape
        # Pad with a zero border: if the mask has no zero pixel at all (a frame filling
        # its whole patch), distanceTransform would return +inf-like values.
        padded = cv2.copyMakeBorder(wf.mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        weight = cv2.distanceTransform(padded, cv2.DIST_L2, 3)[1:-1, 1:-1]
        weight[wf.mask == 0] = 0
        ys, xs = slice(wf.y0, wf.y0 + h), slice(wf.x0, wf.x0 + w)
        self.acc[ys, xs] += wf.image.astype(np.float32) * weight[..., None]
        self.wsum[ys, xs] += weight
        self.n_frames += 1

    def result(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(image uint8 BGR, valid bool mask)``."""
        valid = self.wsum > 0
        out = np.zeros(self.acc.shape, np.float32)
        np.divide(self.acc, self.wsum[..., None], out=out, where=valid[..., None])
        return np.clip(np.rint(out), 0, 255).astype(np.uint8), valid
