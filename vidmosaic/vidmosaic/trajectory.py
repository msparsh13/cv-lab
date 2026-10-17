"""Transformation accumulation, reference alignment, drift diagnostics, canvas bounds.

Coordinate conventions
----------------------
* Every accepted frame k gets ``H_k`` (3x3, float64) mapping **frame k pixel
  coordinates -> reference coordinates**. The reference is one keyframe
  (``reference_frame_index``); its own ``H`` is the identity.
* Composition: if ``H_ab`` maps frame b into frame a (what ``estimate_pair``
  returns for reference a and moving b) and ``H_a`` maps frame a into the
  reference, then ``H_b = H_a @ H_ab``. All matrices are normalised to H[2,2] = 1.
* Canvas: ``C`` (``CanvasInfo.C``) maps reference coordinates -> canvas pixels:
  ``C = S @ T`` with ``T`` the translation by ``-origin`` (origin = integer
  floor of the smallest mapped coordinate, so negative coordinates are shifted
  to >= 0) and ``S`` an optional uniform scale. A frame is warped with
  ``M_k = C @ H_k``.

Algorithm (keyframes)
---------------------
New frames are aligned to the last keyframe (not the previous frame). A frame
that moved at least ``keyframe_shift`` of the frame size becomes the next
keyframe. If a frame cannot be aligned to the keyframe, the last frame that
could be aligned is promoted to keyframe and alignment is retried, so fast pans
do not lose tracking just because the keyframe is too far away. After
``max_consecutive_failures`` failures tracking is declared lost.

Drift
-----
Error accumulates along the chain of keyframe-to-keyframe links. For each new
keyframe n we also estimate the transform directly to keyframe n-2 and compare
it with the chained one; the disagreement in pixels is stored in
``Trajectory.consistency`` and warned about above ``drift_warn_px``. This
detects drift; it does not correct it (no loop closure yet).

Limitation: a planar (homography) projection stretches rapidly for pans that
approach 90 degrees of rotation. Cylindrical/spherical projection is not
implemented.
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, Iterable, List, Optional, Tuple

import numpy as np

from .config import CanvasConfig, MotionConfig, TrajectoryConfig
from .motion_estimation import FeatureSet, detect_features, estimate_pair
from .video_reader import FrameRecord


class PanoramaError(RuntimeError):
    """Base class for errors that stop panorama construction."""


class TrackingLostError(PanoramaError):
    """Frames could not be aligned for too long."""


class CanvasError(PanoramaError):
    """The panorama canvas is invalid or exceeds the configured limits."""


# --------------------------------------------------------------------------
# Data containers
# --------------------------------------------------------------------------

@dataclass
class FrameEntry:
    frame_index: int
    sample_index: int
    timestamp_s: float
    H: np.ndarray                 # frame -> reference
    is_key: bool
    anchor_frame_index: int       # keyframe this frame was aligned to
    n_inliers: int
    rms_error: float

    def to_dict(self) -> Dict[str, Any]:
        return {"frame_index": self.frame_index, "sample_index": self.sample_index,
                "timestamp_s": self.timestamp_s, "H": self.H.tolist(),
                "is_key": self.is_key, "anchor_frame_index": self.anchor_frame_index,
                "n_inliers": self.n_inliers, "rms_error": self.rms_error}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FrameEntry":
        d = dict(d)
        d["H"] = np.asarray(d["H"], np.float64)
        return cls(**d)


@dataclass
class Trajectory:
    entries: List[FrameEntry]
    rejected: List[Dict[str, Any]]            # frames that could not be aligned
    consistency: List[Dict[str, Any]]         # keyframe drift checks
    warnings: List[str]
    frame_size: Tuple[int, int]               # (width, height) of aligned frames
    reference_frame_index: int
    lost_at: Optional[int] = None             # frame index where tracking was lost
    meta: Dict[str, Any] = field(default_factory=dict)   # configs, video identity

    @property
    def complete(self) -> bool:
        return self.lost_at is None

    def keyframes(self) -> List[FrameEntry]:
        return [e for e in self.entries if e.is_key]

    def select(self, use: str) -> List[FrameEntry]:
        return self.keyframes() if use == "keyframes" else list(self.entries)

    def frame_quad(self, e: FrameEntry) -> np.ndarray:
        """Corners (pixel centres) of frame ``e`` in reference coordinates, (4, 2)."""
        return _map_points(e.H, _corners(self.frame_size))

    def camera_path(self) -> np.ndarray:
        """Frame centres in reference coordinates, one row per entry."""
        w, h = self.frame_size
        c = np.array([[(w - 1) / 2.0, (h - 1) / 2.0]])
        return np.vstack([_map_points(e.H, c) for e in self.entries])

    def to_dict(self) -> Dict[str, Any]:
        return {"entries": [e.to_dict() for e in self.entries], "rejected": self.rejected,
                "consistency": self.consistency, "warnings": self.warnings,
                "frame_size": list(self.frame_size),
                "reference_frame_index": self.reference_frame_index,
                "lost_at": self.lost_at, "meta": self.meta}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Trajectory":
        return cls(entries=[FrameEntry.from_dict(e) for e in d["entries"]],
                   rejected=d["rejected"], consistency=d["consistency"],
                   warnings=d["warnings"], frame_size=tuple(d["frame_size"]),
                   reference_frame_index=d["reference_frame_index"],
                   lost_at=d.get("lost_at"), meta=d.get("meta", {}))

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(_clean(self.to_dict()), f, indent=1)

    @classmethod
    def load(cls, path: str) -> "Trajectory":
        with open(path) as f:
            return cls.from_dict(json.load(f))


def _clean(o: Any) -> Any:
    """Make nested data JSON-safe (NaN/inf -> None)."""
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return o


# --------------------------------------------------------------------------
# Geometry helpers
# --------------------------------------------------------------------------

def _corners(size: Tuple[int, int]) -> np.ndarray:
    """Pixel-centre corners of a (width, height) frame as (4, 2)."""
    w, h = size
    return np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float64)


def _map_points(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    p = np.hstack([pts, np.ones((len(pts), 1))]) @ H.T
    return p[:, :2] / p[:, 2:3]


def _normalise(H: np.ndarray) -> np.ndarray:
    return H / H[2, 2]


def homography_disagreement(Ha: np.ndarray, Hb: np.ndarray, size: Tuple[int, int]) -> float:
    """Max distance (px) between where ``Ha`` and ``Hb`` send a grid over the frame."""
    w, h = size
    xs, ys = np.linspace(0, w - 1, 5), np.linspace(0, h - 1, 4)
    pts = np.array([[x, y] for x in xs for y in ys], np.float64)
    return float(np.linalg.norm(_map_points(Ha, pts) - _map_points(Hb, pts), axis=1).max())


# --------------------------------------------------------------------------
# Trajectory construction
# --------------------------------------------------------------------------

@dataclass
class _Anchor:
    entry: FrameEntry
    feats: FeatureSet


ProgressFn = Callable[[int, FrameRecord, Optional[FrameEntry]], None]


def build_trajectory(
    records: Iterable[FrameRecord],
    mcfg: Optional[MotionConfig] = None,
    tcfg: Optional[TrajectoryConfig] = None,
    progress: Optional[ProgressFn] = None,
) -> Trajectory:
    """Align a sequence of frames into one coordinate system. See module docstring."""
    mcfg = mcfg or MotionConfig()
    tcfg = tcfg or TrajectoryConfig()
    mcfg.validate()
    tcfg.validate()

    entries: List[FrameEntry] = []
    rejected: List[Dict[str, Any]] = []
    consistency: List[Dict[str, Any]] = []
    warnings: List[str] = []
    keys: Deque[_Anchor] = deque(maxlen=3)     # last keyframes, newest last
    cand: Optional[_Anchor] = None             # last aligned non-key frame
    failures = 0
    lost_at: Optional[int] = None
    size: Optional[Tuple[int, int]] = None

    def promote(a: _Anchor) -> None:
        if tcfg.consistency_check and len(keys) >= 2:
            older = keys[-2]
            direct = estimate_pair(older.feats, a.feats, mcfg)
            if direct.ok:
                chained = _normalise(np.linalg.inv(older.entry.H) @ a.entry.H)
                err = homography_disagreement(chained, direct.H, size)
                consistency.append({"frame_a": older.entry.frame_index,
                                    "frame_b": a.entry.frame_index,
                                    "disagreement_px": err,
                                    "direct_inliers": direct.metrics.n_inliers})
                if err > tcfg.drift_warn_px:
                    warnings.append(
                        f"possible drift: chained and direct alignment of frames "
                        f"{older.entry.frame_index} and {a.entry.frame_index} disagree by "
                        f"{err:.1f}px (> {tcfg.drift_warn_px}px)")
        a.entry.is_key = True
        keys.append(a)

    for n, rec in enumerate(records):
        feats = detect_features(rec.image, mcfg)
        if size is None:
            size = feats.size
            e0 = FrameEntry(rec.frame_index, rec.sample_index, rec.timestamp_s,
                            np.eye(3), True, rec.frame_index, 0, 0.0)
            entries.append(e0)
            keys.append(_Anchor(e0, feats))
            if progress:
                progress(n, rec, e0)
            continue
        if feats.size != size:
            raise PanoramaError(
                f"Frame size changed from {size} to {feats.size} at frame "
                f"{rec.frame_index}; mosaicking needs constant frame size.")

        key = keys[-1]
        res = estimate_pair(key.feats, feats, mcfg)
        if not res.ok and cand is not None:
            promote(cand)                      # candidate is closer to the new frame
            cand = None
            key = keys[-1]
            res = estimate_pair(key.feats, feats, mcfg)

        entry = None
        if res.ok:
            failures = 0
            H = _normalise(key.entry.H @ res.H)
            entry = FrameEntry(rec.frame_index, rec.sample_index, rec.timestamp_s, H,
                               False, key.entry.frame_index, res.metrics.n_inliers,
                               res.metrics.rms_error)
            entries.append(entry)
            anchor = _Anchor(entry, feats)
            shift = max(abs(res.metrics.shift_x) / size[0], abs(res.metrics.shift_y) / size[1])
            if shift >= tcfg.keyframe_shift:
                promote(anchor)
                cand = None
            else:
                cand = anchor
        else:
            failures += 1
            rejected.append({"frame_index": rec.frame_index, "status": res.status,
                             "reasons": res.reasons})
            if failures > tcfg.max_consecutive_failures:
                lost_at = rec.frame_index
                msg = (f"tracking lost at frame {rec.frame_index}: {failures} consecutive "
                       f"frames could not be aligned ({res.status})")
                if tcfg.on_lost == "error":
                    raise TrackingLostError(msg)
                warnings.append(msg + "; later frames are NOT included (panorama is partial)")
                if progress:
                    progress(n, rec, None)
                break
        if progress:
            progress(n, rec, entry)

    if not entries:
        raise PanoramaError("No frames were provided.")
    if cand is not None and not cand.entry.is_key:
        cand.entry.is_key = True               # make sure the final stretch is covered
    if len(entries) < 2:
        raise PanoramaError(
            "Fewer than two frames could be aligned; cannot build a panorama. "
            "Check that consecutive sampled frames overlap and contain texture "
            "(see the 'align' command for diagnostics).")

    ref_idx = _choose_reference(entries, tcfg.reference)
    ref_entry = next(e for e in entries if e.frame_index == ref_idx)
    inv = np.linalg.inv(ref_entry.H)
    for e in entries:
        e.H = _normalise(inv @ e.H)
    return Trajectory(entries=entries, rejected=rejected, consistency=consistency,
                      warnings=warnings, frame_size=size, reference_frame_index=ref_idx,
                      lost_at=lost_at,
                      meta={"motion_config": mcfg.to_dict(),
                            "trajectory_config": tcfg.to_dict()})


def _choose_reference(entries: List[FrameEntry], mode: str) -> int:
    keys = [e for e in entries if e.is_key]
    if mode == "first":
        return entries[0].frame_index
    return keys[len(keys) // 2].frame_index


# --------------------------------------------------------------------------
# Canvas
# --------------------------------------------------------------------------

@dataclass
class CanvasInfo:
    origin: Tuple[int, int]            # integer reference coords of canvas pixel (0, 0) (unscaled)
    unscaled_size: Tuple[int, int]     # (width, height) before fitting to limits
    scale: float                       # uniform scale applied (<= 1)
    size: Tuple[int, int]              # final (width, height)
    C: np.ndarray                      # reference -> canvas pixels (3x3)
    warnings: List[str] = field(default_factory=list)

    @property
    def pixels(self) -> int:
        return self.size[0] * self.size[1]

    def to_dict(self) -> Dict[str, Any]:
        return {"origin": list(self.origin), "unscaled_size": list(self.unscaled_size),
                "scale": self.scale, "size": list(self.size), "pixels": self.pixels,
                "warnings": self.warnings}


def compute_canvas(
    Hs: List[np.ndarray], frame_size: Tuple[int, int], cfg: Optional[CanvasConfig] = None
) -> CanvasInfo:
    """Canvas containing every mapped pixel centre of every frame.

    ``Hs`` are frame -> reference homographies. Negative coordinates are handled
    by the translation ``T``. If the canvas exceeds ``cfg`` limits it is scaled
    down uniformly (``on_exceed="scale"``) or CanvasError is raised.
    """
    cfg = cfg or CanvasConfig()
    cfg.validate()
    if not Hs:
        raise CanvasError("No frames to place on the canvas.")
    pts = np.vstack([_map_points(H, _corners(frame_size)) for H in Hs])
    if not np.all(np.isfinite(pts)):
        raise CanvasError("Non-finite coordinates: a transform diverged or maps a "
                          "frame corner to infinity.")
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    if np.any(hi - lo > cfg.hard_limit):
        raise CanvasError(
            f"Canvas extent {hi[0]-lo[0]:.0f} x {hi[1]-lo[1]:.0f}px exceeds the sanity "
            f"limit {cfg.hard_limit:.0f}px; the camera trajectory has most likely "
            f"diverged. Inspect it with the trajectory diagnostic image.")
    ox, oy = int(np.floor(lo[0])), int(np.floor(lo[1]))
    wu = int(np.ceil(hi[0])) + 1 - ox
    hu = int(np.ceil(hi[1])) + 1 - oy

    warnings: List[str] = []
    s = min(1.0, cfg.max_width / wu, cfg.max_height / hu,
            float(np.sqrt(cfg.max_pixels / float(wu * hu))))
    if s < 1.0:
        if cfg.on_exceed == "error":
            raise CanvasError(
                f"Panorama would be {wu}x{hu}px ({wu*hu/1e6:.1f} Mpx), exceeding the limits "
                f"(max {cfg.max_width}x{cfg.max_height}, {cfg.max_pixels/1e6:.1f} Mpx). "
                f"Lower max_dim / use fewer frames, or allow on_exceed='scale'.")
        if s < cfg.min_scale:
            raise CanvasError(
                f"Fitting the panorama into the limits needs scale {s:.3f} "
                f"(< min_scale {cfg.min_scale}); the result would be uselessly small. "
                f"The trajectory has probably diverged.")
        # shrink slightly more until integer rounding respects every limit
        while True:
            w, h = int(np.ceil(wu * s)), int(np.ceil(hu * s))
            if w <= cfg.max_width and h <= cfg.max_height and w * h <= cfg.max_pixels:
                break
            s *= 0.999
        warnings.append(f"canvas {wu}x{hu}px exceeded limits; scaled by {s:.3f} to {w}x{h}px")
    else:
        w, h = wu, hu
    T = np.array([[1, 0, -ox], [0, 1, -oy], [0, 0, 1]], np.float64)
    S = np.diag([s, s, 1.0])
    return CanvasInfo((ox, oy), (wu, hu), float(s), (w, h), S @ T, warnings)
