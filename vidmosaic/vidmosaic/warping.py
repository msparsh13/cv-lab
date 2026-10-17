"""Frame warping and validity masks.

Convention: ``M`` maps frame pixel coordinates -> canvas pixel coordinates
(``M = C @ H_k``, see trajectory.py). Warping is forward-defined like
``cv2.warpPerspective``: destination pixel ``M @ p`` takes the colour of source
pixel ``p`` (OpenCV inverts M internally and samples bilinearly).

Validity rule
-------------
A canvas pixel is valid for a frame iff its centre maps back to a point that
rounds to a pixel inside the source image. It is computed by warping an all-ones
image with nearest-neighbour sampling and a zero border, so black or empty
canvas area is never mistaken for image content. The colour image is warped with
BORDER_REPLICATE so valid edge pixels are not blended with black.

Memory: each frame is warped only into the bounding box of its footprint
(``WarpedFrame``), never into a full-canvas array, and is discarded after use.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass
class WarpedFrame:
    image: np.ndarray      # (h, w, 3) uint8, valid only where mask > 0
    mask: np.ndarray       # (h, w) uint8, 255 = valid, 0 = no image content
    x0: int                # top-left of this patch on the canvas
    y0: int

    @property
    def size(self) -> Tuple[int, int]:
        return (self.image.shape[1], self.image.shape[0])


def warp_to_canvas(image: np.ndarray, M: np.ndarray, canvas_size: Tuple[int, int],
                   pad: int = 1) -> Optional[WarpedFrame]:
    """Warp ``image`` with ``M`` into its footprint on a ``canvas_size`` = (W, H) canvas.

    Returns None if the footprint lies entirely outside the canvas.
    """
    h, w = image.shape[:2]
    W, H = canvas_size
    corners = np.array([[[0, 0]], [[w - 1, 0]], [[w - 1, h - 1]], [[0, h - 1]]], np.float64)
    quad = cv2.perspectiveTransform(corners, M).reshape(-1, 2)
    if not np.all(np.isfinite(quad)):
        raise ValueError("Frame maps to non-finite canvas coordinates.")
    x0 = max(0, int(np.floor(quad[:, 0].min())) - pad)
    y0 = max(0, int(np.floor(quad[:, 1].min())) - pad)
    x1 = min(W, int(np.ceil(quad[:, 0].max())) + 1 + pad)
    y1 = min(H, int(np.ceil(quad[:, 1].max())) + 1 + pad)
    if x1 <= x0 or y1 <= y0:
        return None
    T = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]], np.float64)
    M2 = T @ M
    dsize = (x1 - x0, y1 - y0)
    out = cv2.warpPerspective(image, M2, dsize, flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE)
    mask = cv2.warpPerspective(np.full((h, w), 255, np.uint8), M2, dsize,
                               flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
                               borderValue=0)
    out[mask == 0] = 0
    if not mask.any():
        return None
    return WarpedFrame(out, mask, x0, y0)
