"""Shared test utilities: deterministic synthetic videos."""

import cv2
import numpy as np


def write_index_video(path, n_frames=60, fps=25.0, size=(200, 100)):
    """Write a lossless (FFV1) video where frame i is filled with blue == i.

    The frame index is therefore recoverable *exactly* from pixel data, which
    lets tests verify that the reader returns the frames it claims to.
    Requires n_frames <= 256. ``size`` is (width, height).
    """
    assert n_frames <= 256
    w, h = size
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"FFV1"), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError("FFV1 writer unavailable in this OpenCV build")
    for i in range(n_frames):
        f = np.zeros((h, w, 3), np.uint8)
        f[..., 0] = i          # B channel encodes the index
        f[..., 1] = 100        # constant G
        f[..., 2] = (i * 3) % 256
        writer.write(f)
    writer.release()


def decode_index(image):
    """Recover the frame index written by ``write_index_video``."""
    return int(round(float(image[..., 0].mean())))
