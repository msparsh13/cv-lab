"""Video decoding, frame sampling and timestamp tracking.

Conventions
-----------
* Images are OpenCV-style ``uint8`` arrays of shape (H, W, 3), BGR order.
* ``frame_index`` is the 0-based index of the frame in the *source video*.
  It is never renumbered after sampling, so later stages can always map a
  sampled frame back to the original video.
* ``sample_index`` is the 0-based position within the *sampled* sequence.
* ``timestamp_s`` is ``frame_index / fps`` when the container reports a valid
  FPS. This is exact for constant-frame-rate video. If FPS is missing, the
  decoder's own position (CAP_PROP_POS_MSEC) is used instead.
* ``scale`` is ``resized_size / original_size`` (<= 1). A pixel coordinate
  in the resized frame times ``1/scale`` is the coordinate in the original
  frame. Later phases use this to relate results to full resolution.

Frames are decoded strictly sequentially. Seeking (CAP_PROP_POS_FRAMES) is
unreliable for many codecs and produces off-by-N frame errors, which would
silently corrupt timestamps. Skipped frames use ``grab()`` only, which avoids
the colour-conversion cost of ``retrieve()``.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Iterator, List, Optional, Tuple

import cv2
import numpy as np

from .config import ConfigError, SamplingConfig


class VideoOpenError(RuntimeError):
    """The file does not exist or OpenCV cannot decode it."""


class VideoReadError(RuntimeError):
    """The video opened but yielded no usable frames."""


@dataclass(frozen=True)
class VideoInfo:
    """Metadata of the source video (before sampling/resizing)."""

    path: str
    width: int
    height: int
    fps: Optional[float]          # None if the container reports no valid FPS
    frame_count: Optional[int]    # Container's estimate; None if unknown. May be inexact.

    @property
    def duration_s(self) -> Optional[float]:
        if self.fps and self.frame_count:
            return self.frame_count / self.fps
        return None


@dataclass
class FrameRecord:
    """One sampled frame together with everything needed to trace it back."""

    sample_index: int
    frame_index: int
    timestamp_s: float
    image: np.ndarray             # resized BGR uint8
    scale: float                  # resized / original, in (0, 1]
    original_size: Tuple[int, int]  # (width, height) of the source frame

    @property
    def size(self) -> Tuple[int, int]:
        """(width, height) of ``image``."""
        h, w = self.image.shape[:2]
        return (w, h)


def resize_to_max_dim(image: np.ndarray, max_dim: Optional[int]) -> Tuple[np.ndarray, float]:
    """Shrink ``image`` so its longest side is <= ``max_dim``.

    Returns ``(image, scale)``. Never enlarges; returns the input unchanged
    with ``scale == 1.0`` if it already fits or ``max_dim`` is None.
    Uses INTER_AREA, the appropriate filter for downscaling.
    """
    if max_dim is None:
        return image, 1.0
    h, w = image.shape[:2]
    longest = max(h, w)
    if longest <= max_dim:
        return image, 1.0
    scale = max_dim / float(longest)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    # Report the effective scale actually applied after integer rounding.
    return resized, new_w / float(w)


class VideoReader:
    """Sequential, sampling video reader.

    Usage::

        with VideoReader("pan.mp4", SamplingConfig(target_fps=5, max_dim=960)) as vr:
            print(vr.info)
            for rec in vr:
                ...  # rec is a FrameRecord

    The reader is a context manager and also iterable. Iterating a second
    time re-opens the video from the start.
    """

    def __init__(self, path: str, config: Optional[SamplingConfig] = None):
        self.path = str(path)
        self.config = config or SamplingConfig()
        self.config.validate()
        self._cap: Optional[cv2.VideoCapture] = None
        self.info = self._probe()
        self.step = self.config.resolve_step(self.info.fps)

    # -- lifecycle ---------------------------------------------------------

    def _open(self) -> cv2.VideoCapture:
        if not os.path.isfile(self.path):
            raise VideoOpenError(f"Video file not found: {self.path}")
        cap = cv2.VideoCapture(self.path)
        if not cap.isOpened():
            cap.release()
            raise VideoOpenError(
                f"OpenCV could not open '{self.path}'. The file may be corrupt "
                f"or use a codec this OpenCV build does not support."
            )
        return cap

    def _probe(self) -> VideoInfo:
        cap = self._open()
        try:
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            n = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        finally:
            cap.release()
        fps_valid = fps is not None and math.isfinite(fps) and fps > 0
        n_valid = n is not None and math.isfinite(n) and n > 0
        return VideoInfo(
            path=self.path,
            width=w,
            height=h,
            fps=float(fps) if fps_valid else None,
            frame_count=int(n) if n_valid else None,
        )

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "VideoReader":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- iteration ---------------------------------------------------------

    def __iter__(self) -> Iterator[FrameRecord]:
        self.close()
        self._cap = self._open()
        cfg = self.config
        cap = self._cap
        fps = self.info.fps

        frame_index = 0
        sample_index = 0
        try:
            while True:
                if cfg.end_frame is not None and frame_index >= cfg.end_frame:
                    break
                if cfg.max_frames is not None and sample_index >= cfg.max_frames:
                    break

                if not cap.grab():
                    break  # end of stream (or decode failure; see below)

                # A frame is kept if it lies in range and falls on the step grid
                # anchored at start_frame, so the first kept frame is start_frame.
                in_range = frame_index >= cfg.start_frame
                on_grid = (frame_index - cfg.start_frame) % self.step == 0
                if in_range and on_grid:
                    ok, frame = cap.retrieve()
                    if not ok or frame is None:
                        raise VideoReadError(
                            f"Failed to decode frame {frame_index} of '{self.path}'."
                        )
                    if fps is not None:
                        ts = frame_index / fps
                    else:
                        ts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                    h, w = frame.shape[:2]
                    image, scale = resize_to_max_dim(frame, cfg.max_dim)
                    yield FrameRecord(
                        sample_index=sample_index,
                        frame_index=frame_index,
                        timestamp_s=float(ts),
                        image=image,
                        scale=scale,
                        original_size=(w, h),
                    )
                    sample_index += 1
                frame_index += 1
        finally:
            self.close()

    def read_all(self) -> List[FrameRecord]:
        """Collect all sampled frames, raising if there are none.

        Convenience for small clips and tests. For long videos iterate
        instead, so frames are not all held in memory at once.
        """
        frames = list(self)
        if not frames:
            raise VideoReadError(
                f"No frames were sampled from '{self.path}' "
                f"(start_frame={self.config.start_frame}, "
                f"end_frame={self.config.end_frame}, step={self.step}). "
                f"Check that the range lies within the video."
            )
        return frames
