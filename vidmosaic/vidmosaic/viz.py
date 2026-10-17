"""Lightweight visualisation helpers (no GUI dependency).

Kept separate from ``video_reader`` so the core stays UI-free; the CLI and the
future ``ui.py`` both build on these functions.
"""

from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np

from .video_reader import FrameRecord


def make_contact_sheet(
    records: Sequence[FrameRecord],
    columns: int = 4,
    thumb_width: int = 320,
    max_thumbs: int = 24,
) -> np.ndarray:
    """Tile thumbnails of sampled frames, each labelled with index and time.

    If more than ``max_thumbs`` records are given, they are subsampled evenly
    so the sheet spans the whole clip.
    """
    if not records:
        raise ValueError("make_contact_sheet needs at least one frame.")
    if len(records) > max_thumbs:
        idx = np.linspace(0, len(records) - 1, max_thumbs).round().astype(int)
        records = [records[i] for i in idx]

    thumbs = []
    for r in records:
        h, w = r.image.shape[:2]
        th = max(1, int(round(h * thumb_width / w)))
        t = cv2.resize(r.image, (thumb_width, th), interpolation=cv2.INTER_AREA)
        label = f"#{r.frame_index}  {r.timestamp_s:.2f}s"
        cv2.rectangle(t, (0, 0), (min(thumb_width, 8 * len(label) + 8), 18), (0, 0, 0), -1)
        cv2.putText(t, label, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (255, 255, 255), 1, cv2.LINE_AA)
        thumbs.append(t)

    cell_h = max(t.shape[0] for t in thumbs)
    cols = max(1, min(columns, len(thumbs)))
    rows = (len(thumbs) + cols - 1) // cols
    sheet = np.zeros((rows * cell_h, cols * thumb_width, 3), np.uint8)
    for i, t in enumerate(thumbs):
        r, c = divmod(i, cols)
        sheet[r * cell_h : r * cell_h + t.shape[0], c * thumb_width : (c + 1) * thumb_width] = t
    return sheet


# --------------------------------------------------------------------------
# Phase 2 diagnostics
# --------------------------------------------------------------------------

def _banner(img: np.ndarray, lines: Sequence[str], color=(255, 255, 255)) -> np.ndarray:
    """Return ``img`` with a black text banner on top."""
    h = 8 + 18 * len(lines)
    band = np.zeros((h, img.shape[1], 3), np.uint8)
    for i, line in enumerate(lines):
        cv2.putText(band, line, (6, 18 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    color, 1, cv2.LINE_AA)
    return np.vstack([band, img])


def draw_matches(ref_img: np.ndarray, mov_img: np.ndarray, result,
                 max_lines: int = 150) -> np.ndarray:
    """Side-by-side view: reference (left), moving frame (right).

    Green lines: RANSAC inliers. Red lines: matches that passed the ratio test
    but were rejected by RANSAC (outliers, e.g. on moving objects).
    ``result`` is a ``motion_estimation.PairResult``. At most ``max_lines``
    inliers and ``max_lines`` outliers are drawn (evenly strided).
    """
    h = max(ref_img.shape[0], mov_img.shape[0])
    canvas = np.zeros((h, ref_img.shape[1] + mov_img.shape[1], 3), np.uint8)
    canvas[: ref_img.shape[0], : ref_img.shape[1]] = ref_img
    canvas[: mov_img.shape[0], ref_img.shape[1]:] = mov_img
    ox = ref_img.shape[1]

    def draw(idx, color):
        if len(idx) > max_lines:
            idx = idx[np.linspace(0, len(idx) - 1, max_lines).astype(int)]
        for i in idx:
            p = tuple(int(round(v)) for v in result.pts_ref[i])
            q = (int(round(result.pts_mov[i][0])) + ox, int(round(result.pts_mov[i][1])))
            cv2.line(canvas, p, q, color, 1, cv2.LINE_AA)
            cv2.circle(canvas, p, 3, color, 1, cv2.LINE_AA)
            cv2.circle(canvas, q, 3, color, 1, cv2.LINE_AA)

    mask = result.inlier_mask
    if len(mask):
        draw(np.flatnonzero(~mask), (60, 60, 255))
        draw(np.flatnonzero(mask), (60, 220, 60))

    m = result.metrics
    head = (f"frame {result.ref_frame_index} <- {result.mov_frame_index}   "
            f"[{result.status.upper()}]")
    stats = (f"good={m.n_good} inliers={m.n_inliers} ({m.inlier_ratio:.0%}) "
             f"rms={m.rms_error:.2f}px cov={m.coverage:.0%} "
             f"shift=({m.shift_x:.1f},{m.shift_y:.1f})")
    return _banner(canvas, [head, stats],
                   color=(120, 255, 120) if result.ok else (90, 90, 255))


def alignment_overlay(ref_img: np.ndarray, mov_img: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Absolute difference between the reference and the moving frame warped by ``H``.

    Black = well aligned (and static); bright edges = misalignment or motion.
    Pixels that the warped frame does not cover are shown in dark blue so empty
    area is never mistaken for a good match. ``H`` maps moving -> reference.
    """
    h, w = ref_img.shape[:2]
    warped = cv2.warpPerspective(mov_img, H, (w, h), flags=cv2.INTER_LINEAR)
    valid = cv2.warpPerspective(np.full(mov_img.shape[:2], 255, np.uint8), H, (w, h),
                                flags=cv2.INTER_NEAREST)
    diff = cv2.absdiff(ref_img, warped)
    diff[valid == 0] = (80, 0, 0)
    return diff


def draw_trajectory(traj, max_width: int = 1400, margin: int = 30) -> np.ndarray:
    """Diagnostic picture of the camera trajectory in reference coordinates.

    Keyframe footprints are drawn as coloured quadrilaterals (blue -> red over
    time), the reference keyframe in white, and the path of the frame centres as a
    yellow line. Skewed or fanning footprints indicate perspective stretch; a
    path that curves away from the expected motion indicates drift.
    """
    quads = [traj.frame_quad(e) for e in traj.entries]
    allp = np.vstack(quads)
    lo, hi = allp.min(axis=0), allp.max(axis=0)
    span = np.maximum(hi - lo, 1.0)
    s = (max_width - 2 * margin) / span[0]
    s = min(s, 2.0)
    W = int(span[0] * s) + 2 * margin
    H = int(span[1] * s) + 2 * margin + 40
    img = np.full((H, W, 3), 24, np.uint8)

    def pt(p):
        return (int((p[0] - lo[0]) * s) + margin, int((p[1] - lo[1]) * s) + margin + 40)

    keys = [e for e in traj.entries if e.is_key]
    for i, e in enumerate(keys):
        t = i / max(1, len(keys) - 1)
        col = (int(255 * (1 - t)), 80, int(255 * t))
        if e.frame_index == traj.reference_frame_index:
            col = (255, 255, 255)
        q = np.array([pt(p) for p in traj.frame_quad(e)], np.int32)
        cv2.polylines(img, [q], True, col, 1, cv2.LINE_AA)
    path = np.array([pt(p) for p in traj.camera_path()], np.int32)
    cv2.polylines(img, [path], False, (0, 220, 255), 2, cv2.LINE_AA)
    cv2.circle(img, tuple(path[0]), 5, (0, 255, 0), -1)
    cv2.circle(img, tuple(path[-1]), 5, (0, 0, 255), -1)
    txt = (f"{len(traj.entries)} aligned frames, {len(keys)} keyframes, "
           f"{len(traj.rejected)} rejected | white = reference, green = start, red = end")
    cv2.putText(img, txt, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1,
                cv2.LINE_AA)
    return img
