"""Central place for configurable parameters and resource limits.

Each pipeline stage owns one dataclass here. Later phases will add
``MotionConfig``, ``CanvasConfig`` etc. next to ``SamplingConfig``.
All configs are plain dataclasses so they can be dumped to JSON for
reproducibility (``to_dict`` / ``from_dict``).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional


class ConfigError(ValueError):
    """Raised when a configuration value is invalid or contradictory."""


@dataclass
class SamplingConfig:
    """How frames are selected from the video and how large they may be.

    Frame selection (choose at most one of ``frame_step`` / ``target_fps``):
        frame_step   Keep every N-th decoded frame (1 = every frame).
        target_fps   Keep roughly this many frames per second of video.
                     Converted to an integer step: round(video_fps / target_fps),
                     never below 1.

    Range (in *source video frame indices*, 0-based):
        start_frame  First frame index considered (inclusive).
        end_frame    Stop before this index (exclusive). None = to the end.
        max_frames   Cap on the number of *sampled* frames. None = no cap.

    Resizing:
        max_dim      Longest side of a returned frame is at most this many
                     pixels. Frames are only ever shrunk, never enlarged.
                     None = keep original resolution.
    """

    frame_step: Optional[int] = None
    target_fps: Optional[float] = None
    start_frame: int = 0
    end_frame: Optional[int] = None
    max_frames: Optional[int] = None
    max_dim: Optional[int] = 1280

    def validate(self) -> None:
        if self.frame_step is not None and self.target_fps is not None:
            raise ConfigError("Specify only one of frame_step or target_fps, not both.")
        if self.frame_step is not None and self.frame_step < 1:
            raise ConfigError(f"frame_step must be >= 1, got {self.frame_step}.")
        if self.target_fps is not None and not self.target_fps > 0:
            raise ConfigError(f"target_fps must be > 0, got {self.target_fps}.")
        if self.start_frame < 0:
            raise ConfigError(f"start_frame must be >= 0, got {self.start_frame}.")
        if self.end_frame is not None and self.end_frame <= self.start_frame:
            raise ConfigError(
                f"end_frame ({self.end_frame}) must be greater than "
                f"start_frame ({self.start_frame})."
            )
        if self.max_frames is not None and self.max_frames < 1:
            raise ConfigError(f"max_frames must be >= 1, got {self.max_frames}.")
        if self.max_dim is not None and self.max_dim < 16:
            raise ConfigError(f"max_dim must be >= 16 pixels, got {self.max_dim}.")

    def resolve_step(self, video_fps: Optional[float]) -> int:
        """Return the integer frame step to use for a video with ``video_fps``."""
        if self.frame_step is not None:
            return self.frame_step
        if self.target_fps is not None:
            if video_fps is None or not video_fps > 0:
                raise ConfigError(
                    "target_fps was requested but the video reports no valid FPS; "
                    "use frame_step instead."
                )
            return max(1, int(round(video_fps / self.target_fps)))
        return 1

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SamplingConfig":
        cfg = cls(**d)
        cfg.validate()
        return cfg


@dataclass
class MotionConfig:
    """Parameters of pairwise camera-motion estimation (Phase 2).

    Features / matching:
        max_features        Keep the N strongest SIFT keypoints per frame (0 = no cap).
        contrast_threshold  SIFT contrast threshold; lower = more (weaker) keypoints.
        min_features        A frame with fewer keypoints is rejected outright.
        matcher             "bf" (exact brute force) or "flann" (approximate, faster
                            for many features).
        ratio               Lowe's ratio test: keep a match only if
                            d(best) < ratio * d(second best). Lower = stricter.
        min_matches         Minimum matches surviving the ratio test.

    RANSAC homography:
        ransac_thresh       Max reprojection error (pixels) for a match to count as an
                            inlier during RANSAC.
        ransac_confidence   RANSAC confidence (0..1).
        ransac_max_iters    RANSAC iteration cap.

    Acceptance gates (a pair failing any gate is rejected, never silently used):
        min_inliers         Minimum RANSAC inliers.
        min_inlier_ratio    Minimum inliers / ratio-test matches.
        max_reproj_error    Maximum RMS reprojection error over inliers (pixels).
        min_inlier_coverage Minimum convex-hull area of the inliers as a fraction of
                            the frame area. Rejects estimates supported only by a small
                            cluster of points, which extrapolate badly.
        max_scale_change    The homography's local scale must lie in
                            [1/max_scale_change, max_scale_change]. A panning camera
                            should not zoom much between consecutive frames.
    """

    max_features: int = 3000
    contrast_threshold: float = 0.04
    min_features: int = 50
    matcher: str = "bf"
    ratio: float = 0.75
    min_matches: int = 30
    ransac_thresh: float = 3.0
    ransac_confidence: float = 0.995
    ransac_max_iters: int = 2000
    min_inliers: int = 25
    min_inlier_ratio: float = 0.25
    max_reproj_error: float = 2.0
    min_inlier_coverage: float = 0.05
    max_scale_change: float = 1.5

    def validate(self) -> None:
        if self.max_features < 0:
            raise ConfigError("max_features must be >= 0.")
        if not self.contrast_threshold > 0:
            raise ConfigError("contrast_threshold must be > 0.")
        if self.min_features < 4:
            raise ConfigError("min_features must be >= 4.")
        if self.matcher not in ("bf", "flann"):
            raise ConfigError(f"matcher must be 'bf' or 'flann', got {self.matcher!r}.")
        if not 0.0 < self.ratio < 1.0:
            raise ConfigError(f"ratio must be in (0, 1), got {self.ratio}.")
        if self.min_matches < 4:
            raise ConfigError("min_matches must be >= 4 (a homography needs 4 points).")
        if not self.ransac_thresh > 0:
            raise ConfigError("ransac_thresh must be > 0.")
        if not 0.0 < self.ransac_confidence < 1.0:
            raise ConfigError("ransac_confidence must be in (0, 1).")
        if self.ransac_max_iters < 1:
            raise ConfigError("ransac_max_iters must be >= 1.")
        if self.min_inliers < 4:
            raise ConfigError("min_inliers must be >= 4.")
        if not 0.0 <= self.min_inlier_ratio <= 1.0:
            raise ConfigError("min_inlier_ratio must be in [0, 1].")
        if not self.max_reproj_error > 0:
            raise ConfigError("max_reproj_error must be > 0.")
        if not 0.0 <= self.min_inlier_coverage <= 1.0:
            raise ConfigError("min_inlier_coverage must be in [0, 1].")
        if not self.max_scale_change >= 1.0:
            raise ConfigError("max_scale_change must be >= 1.")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MotionConfig":
        cfg = cls(**d)
        cfg.validate()
        return cfg


@dataclass
class TrajectoryConfig:
    """How per-pair motions are chained into one coordinate system (Phase 3).

    keyframe_shift            A frame becomes a keyframe (anchor) once it has moved
                              this fraction of the frame size (max of |dx|/width and
                              |dy|/height) relative to the previous keyframe. New frames
                              are aligned to the last keyframe, not the previous frame,
                              so error accumulates over few links instead of many.
    max_consecutive_failures  Frames that cannot be aligned in a row before tracking is
                              declared lost.
    on_lost                   "truncate": keep what was aligned so far, warn loudly.
                              "error": raise TrackingLostError.
    reference                 Which keyframe defines the output coordinate system:
                              "first" or "middle". "middle" keeps perspective stretching
                              symmetric for long pans.
    consistency_check         For each new keyframe also align it directly to the keyframe
                              before the previous one and compare with the chained result.
                              The disagreement (px) measures accumulated drift.
    drift_warn_px             Warn if that disagreement exceeds this many pixels.
    """

    keyframe_shift: float = 0.2
    max_consecutive_failures: int = 5
    on_lost: str = "truncate"
    reference: str = "middle"
    consistency_check: bool = True
    drift_warn_px: float = 3.0

    def validate(self) -> None:
        if not 0.0 < self.keyframe_shift < 1.0:
            raise ConfigError("keyframe_shift must be in (0, 1).")
        if self.max_consecutive_failures < 0:
            raise ConfigError("max_consecutive_failures must be >= 0.")
        if self.on_lost not in ("truncate", "error"):
            raise ConfigError("on_lost must be 'truncate' or 'error'.")
        if self.reference not in ("first", "middle"):
            raise ConfigError("reference must be 'first' or 'middle'.")
        if not self.drift_warn_px > 0:
            raise ConfigError("drift_warn_px must be > 0.")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TrajectoryConfig":
        cfg = cls(**d)
        cfg.validate()
        return cfg


@dataclass
class CanvasConfig:
    """Limits on the output canvas (Phase 3).

    max_width, max_height, max_pixels  Upper bounds on the final panorama size.
        The blender needs ~16 bytes per canvas pixel (float32 colour sum + weight sum),
        so 20 Mpx is ~320 MB.
    on_exceed   "scale": uniformly shrink the whole panorama to fit (a warning is
                recorded). "error": raise CanvasError instead.
    min_scale   If fitting needs a scale below this, raise CanvasError even in
                "scale" mode: such a result would be uselessly tiny and usually
                means the trajectory diverged.
    hard_limit  Sanity cap (px) on the unscaled extent. A larger extent means the
                transforms diverged (not just a big panorama).
    """

    max_width: int = 20000
    max_height: int = 20000
    max_pixels: int = 20_000_000
    on_exceed: str = "scale"
    min_scale: float = 0.1
    hard_limit: float = 1.0e6

    def validate(self) -> None:
        if self.max_width < 16 or self.max_height < 16:
            raise ConfigError("max_width and max_height must be >= 16.")
        if self.max_pixels < 256:
            raise ConfigError("max_pixels must be >= 256.")
        if self.on_exceed not in ("scale", "error"):
            raise ConfigError("on_exceed must be 'scale' or 'error'.")
        if not 0.0 < self.min_scale <= 1.0:
            raise ConfigError("min_scale must be in (0, 1].")
        if not self.hard_limit > 0:
            raise ConfigError("hard_limit must be > 0.")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CanvasConfig":
        cfg = cls(**d)
        cfg.validate()
        return cfg


@dataclass
class RenderConfig:
    """Which aligned frames are composited (Phase 4).

    use   "keyframes": only anchor frames (fewer overlaps, less ghosting from moving
          objects, faster). "all": every aligned frame.
    """

    use: str = "keyframes"

    def validate(self) -> None:
        if self.use not in ("keyframes", "all"):
            raise ConfigError("use must be 'keyframes' or 'all'.")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RenderConfig":
        cfg = cls(**d)
        cfg.validate()
        return cfg
