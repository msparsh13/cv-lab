"""Feature-based camera motion estimation between video frames.

Pipeline for one pair:  SIFT -> descriptor matching -> Lowe ratio test ->
RANSAC homography -> quality metrics -> acceptance gates.

Coordinate and direction conventions
------------------------------------
* Pixel coordinates are (x, y), OpenCV style: x to the right, y downwards, in
  the *resized* frame (``FrameRecord.image``), not the original resolution.
  Divide by ``FrameRecord.scale`` to express them at original resolution.
* The homography ``H`` maps points of the **moving** frame into the
  **reference** frame:        p_ref ~ H @ [x_mov, y_mov, 1]^T
  For consecutive frames the reference is the previous frame, so Phase 3 can
  accumulate ``H_0k = H_01 @ H_12 @ ... @ H_(k-1,k)``, which maps frame k into
  frame 0's coordinates. ``cv2.warpPerspective(mov, H, ref_size)`` therefore
  renders the moving frame in the reference frame.
* The reprojection error of a match is ||H p_mov - p_ref|| in reference pixels.

Model assumptions and known limitations
---------------------------------------
A single homography is exact only for (a) a camera rotating about its optical
centre, or (b) a planar scene. It is violated by:
  * parallax / camera translation (near and far objects shift differently),
  * zoom (changes scale; flagged only if it exceeds ``max_scale_change``),
  * rolling shutter (rows captured at different times; a fast pan skews the image),
  * lens distortion (not modelled),
  * independently moving objects (their matches are outliers; RANSAC discards
    them as long as the static background provides most of the matches),
  * featureless regions (sky, blank walls) and repetitive textures.
The acceptance gates below detect many of these indirectly (few inliers, high
reprojection error, small coverage), but cannot detect them all.

RANSAC in OpenCV uses a fixed internal seed, so results are deterministic.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import cv2
import numpy as np

from .config import MotionConfig
from .video_reader import FrameRecord

# Status codes (PairResult.status). "ok" is the only status safe to use downstream.
OK = "ok"
INSUFFICIENT_FEATURES = "insufficient_features"
INSUFFICIENT_MATCHES = "insufficient_matches"
HOMOGRAPHY_FAILED = "homography_failed"
TOO_FEW_INLIERS = "too_few_inliers"
LOW_INLIER_RATIO = "low_inlier_ratio"
HIGH_REPROJ_ERROR = "high_reproj_error"
LOW_COVERAGE = "low_coverage"
DEGENERATE_GEOMETRY = "degenerate_geometry"

# A frame corner must map to a point with homogeneous weight w above this
# (after normalising H[2,2] to 1); smaller values mean it is thrown far away.
_MIN_CORNER_W = 0.1


# --------------------------------------------------------------------------
# Data containers
# --------------------------------------------------------------------------

@dataclass
class FeatureSet:
    """SIFT keypoints of one frame."""

    pts: np.ndarray                 # (N, 2) float32, (x, y) in frame pixels
    desc: Optional[np.ndarray]      # (N, 128) float32, or None if N == 0
    size: Tuple[int, int]           # (width, height) of the frame
    detect_ms: float = 0.0

    def __len__(self) -> int:
        return int(self.pts.shape[0])


@dataclass
class PairMetrics:
    n_features_ref: int = 0
    n_features_mov: int = 0
    n_candidates: int = 0           # moving keypoints that had two neighbours to compare
    n_good: int = 0                 # matches surviving the ratio test
    n_inliers: int = 0              # RANSAC inliers
    inlier_ratio: float = 0.0       # n_inliers / n_good
    rms_error: float = float("nan")     # reprojection error over inliers (px)
    mean_error: float = float("nan")
    median_error: float = float("nan")
    max_error: float = float("nan")
    coverage: float = 0.0           # inlier convex-hull area / frame area
    shift_x: float = float("nan")   # displacement of the frame centre under H (px)
    shift_y: float = float("nan")
    rotation_deg: float = float("nan")
    scale: float = float("nan")     # sqrt(det) of the affine part of H
    detect_ms: float = 0.0          # SIFT time for the moving frame
    match_ms: float = 0.0
    ransac_ms: float = 0.0


@dataclass
class PairResult:
    """Outcome of aligning one moving frame to one reference frame.

    ``H`` maps moving -> reference coordinates (see module docstring). It may be
    present even when ``ok`` is False (kept for diagnostics); downstream code
    must check ``ok`` before using it.
    """

    ok: bool
    status: str
    reasons: List[str]
    H: Optional[np.ndarray]
    metrics: PairMetrics
    pts_ref: np.ndarray             # (n_good, 2) matched points in the reference frame
    pts_mov: np.ndarray             # (n_good, 2) matched points in the moving frame
    inlier_mask: np.ndarray         # (n_good,) bool
    ref_frame_index: Optional[int] = None
    mov_frame_index: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ref_frame_index": self.ref_frame_index,
            "mov_frame_index": self.mov_frame_index,
            "ok": self.ok,
            "status": self.status,
            "reasons": list(self.reasons),
            "H": None if self.H is None else self.H.tolist(),
            "metrics": _jsonable(asdict(self.metrics)),
        }


def _jsonable(d: Dict[str, Any]) -> Dict[str, Any]:
    """Replace NaN/inf (invalid JSON) with None."""
    out = {}
    for k, v in d.items():
        if isinstance(v, float) and not np.isfinite(v):
            out[k] = None
        else:
            out[k] = v
    return out


# --------------------------------------------------------------------------
# Step 1: features
# --------------------------------------------------------------------------

def detect_features(
    image: np.ndarray, cfg: MotionConfig, mask: Optional[np.ndarray] = None
) -> FeatureSet:
    """Detect SIFT keypoints and descriptors.

    ``mask`` (uint8, nonzero = search here) lets later phases exclude detected
    moving objects from feature detection.
    """
    t0 = time.perf_counter()
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create(nfeatures=cfg.max_features,
                           contrastThreshold=cfg.contrast_threshold)
    kps, desc = sift.detectAndCompute(gray, mask)
    h, w = gray.shape[:2]
    if not kps or desc is None:
        pts = np.zeros((0, 2), np.float32)
        desc = None
    else:
        pts = np.array([k.pt for k in kps], dtype=np.float32)
    return FeatureSet(pts=pts, desc=desc, size=(w, h),
                      detect_ms=(time.perf_counter() - t0) * 1000.0)


# --------------------------------------------------------------------------
# Step 2: matching + Lowe ratio test
# --------------------------------------------------------------------------

def match_features(
    ref: FeatureSet, mov: FeatureSet, cfg: MotionConfig
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Match moving descriptors to reference descriptors with the ratio test.

    Returns ``(idx_mov, idx_ref, n_candidates)``: index arrays of equal length
    for the surviving matches, and the number of moving keypoints that could be
    compared (had two nearest neighbours).
    """
    empty = np.zeros(0, np.int64)
    if ref.desc is None or mov.desc is None or len(ref) < 2 or len(mov) < 1:
        return empty, empty, 0

    if cfg.matcher == "flann":
        index_params = dict(algorithm=1, trees=4)       # FLANN_INDEX_KDTREE
        matcher = cv2.FlannBasedMatcher(index_params, dict(checks=64))
    else:
        matcher = cv2.BFMatcher(cv2.NORM_L2)

    knn = matcher.knnMatch(mov.desc, ref.desc, k=2)
    idx_mov, idx_ref, n_cand = [], [], 0
    for pair in knn:
        if len(pair) < 2:
            continue
        n_cand += 1
        best, second = pair
        if best.distance < cfg.ratio * second.distance:
            idx_mov.append(best.queryIdx)
            idx_ref.append(best.trainIdx)
    return np.asarray(idx_mov, np.int64), np.asarray(idx_ref, np.int64), n_cand


# --------------------------------------------------------------------------
# Step 3: homography + diagnostics
# --------------------------------------------------------------------------

def check_homography_geometry(
    H: np.ndarray, size: Tuple[int, int], max_scale_change: float
) -> List[str]:
    """Return a list of problems that make ``H`` implausible for a camera pan.

    Checks, with ``size = (width, height)`` of the moving frame:
      * H is finite and H[2,2] is not ~0 (needed to normalise);
      * the four frame corners map to finite points with positive weight;
      * the mapped quadrilateral keeps its orientation and is convex (a fold or
        reflection means the transform is not a plausible view change);
      * the local scale (singular values of the 2x2 affine part) lies within
        [1/max_scale_change, max_scale_change];
      * every mapped frame edge keeps its length within the same factor (catches
        strong perspective that the affine part cannot see).
    """
    problems: List[str] = []
    if H is None or not np.all(np.isfinite(H)):
        return ["homography contains non-finite values"]
    if abs(H[2, 2]) < 1e-12:
        return ["homography cannot be normalised (H[2,2] ~ 0)"]
    Hn = H / H[2, 2]
    w, h = size
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float64)
    hom = Hn @ np.vstack([corners.T, np.ones(4)])
    if np.any(hom[2] <= _MIN_CORNER_W):
        return ["a frame corner maps to or beyond the horizon (extreme perspective)"]
    quad = (hom[:2] / hom[2]).T
    crosses = []
    for i in range(4):
        a, b, c = quad[i], quad[(i + 1) % 4], quad[(i + 2) % 4]
        e1, e2 = b - a, c - b
        crosses.append(e1[0] * e2[1] - e1[1] * e2[0])
    if not all(c > 0 for c in crosses):
        problems.append("mapped frame is folded, reflected or non-convex")
    lo, hi = 1.0 / max_scale_change, max_scale_change
    sv = np.linalg.svd(Hn[:2, :2], compute_uv=False)
    if sv[1] < lo or sv[0] > hi:
        problems.append(
            f"scale change out of range (singular values {sv[0]:.2f}, {sv[1]:.2f}; "
            f"allowed [{lo:.2f}, {hi:.2f}])")
    # The affine part alone misses strong perspective (it describes the scale at the
    # origin only), so also bound how much each frame edge is stretched or shrunk.
    orig = np.linalg.norm(np.roll(corners, -1, axis=0) - corners, axis=1)
    mapped = np.linalg.norm(np.roll(quad, -1, axis=0) - quad, axis=1)
    ratios = mapped / orig
    if ratios.min() < lo or ratios.max() > hi:
        problems.append(
            f"frame edges stretched/shrunk by {ratios.min():.2f}x to {ratios.max():.2f}x "
            f"(allowed [{lo:.2f}, {hi:.2f}]); extreme perspective")
    return problems


def _hull_area(pts: np.ndarray) -> float:
    if len(pts) < 3:
        return 0.0
    hull = cv2.convexHull(pts.astype(np.float32))
    return float(cv2.contourArea(hull))


def estimate_pair(ref: FeatureSet, mov: FeatureSet, cfg: MotionConfig) -> PairResult:
    """Estimate the homography mapping the moving frame into the reference frame."""
    m = PairMetrics(n_features_ref=len(ref), n_features_mov=len(mov),
                    detect_ms=mov.detect_ms)
    no_pts = np.zeros((0, 2), np.float32)
    no_mask = np.zeros(0, bool)

    def fail(status: str, reasons: List[str], H=None, pr=no_pts, pm=no_pts, mask=no_mask):
        return PairResult(False, status, reasons, H, m, pr, pm, mask)

    if len(ref) < cfg.min_features or len(mov) < cfg.min_features:
        return fail(INSUFFICIENT_FEATURES, [
            f"{INSUFFICIENT_FEATURES}: reference has {len(ref)}, moving has {len(mov)} "
            f"keypoints (need >= {cfg.min_features}); frame may be blank or blurred"])

    t0 = time.perf_counter()
    idx_mov, idx_ref, m.n_candidates = match_features(ref, mov, cfg)
    m.match_ms = (time.perf_counter() - t0) * 1000.0
    m.n_good = int(len(idx_mov))
    if m.n_good < cfg.min_matches:
        return fail(INSUFFICIENT_MATCHES, [
            f"{INSUFFICIENT_MATCHES}: {m.n_good} matches passed the ratio test "
            f"(need >= {cfg.min_matches}); too little overlap or texture"])

    pts_mov = mov.pts[idx_mov]
    pts_ref = ref.pts[idx_ref]

    t0 = time.perf_counter()
    H, mask = cv2.findHomography(pts_mov, pts_ref, cv2.RANSAC, cfg.ransac_thresh,
                                 maxIters=cfg.ransac_max_iters,
                                 confidence=cfg.ransac_confidence)
    m.ransac_ms = (time.perf_counter() - t0) * 1000.0
    if H is None or mask is None:
        return fail(HOMOGRAPHY_FAILED, [
            f"{HOMOGRAPHY_FAILED}: RANSAC could not fit a homography"],
            pr=pts_ref, pm=pts_mov)

    inl = mask.ravel().astype(bool)
    m.n_inliers = int(inl.sum())
    m.inlier_ratio = m.n_inliers / float(m.n_good)

    # Reprojection error of the inliers under the final (refined) H.
    if m.n_inliers > 0:
        proj = cv2.perspectiveTransform(pts_mov[inl].reshape(-1, 1, 2).astype(np.float64),
                                        H).reshape(-1, 2)
        err = np.linalg.norm(proj - pts_ref[inl].astype(np.float64), axis=1)
        m.rms_error = float(np.sqrt(np.mean(err ** 2)))
        m.mean_error = float(err.mean())
        m.median_error = float(np.median(err))
        m.max_error = float(err.max())
        m.coverage = _hull_area(pts_mov[inl]) / float(mov.size[0] * mov.size[1])

    # Human-readable motion summary from H (only meaningful for a sane H).
    if np.all(np.isfinite(H)) and abs(H[2, 2]) > 1e-12:
        Hn = H / H[2, 2]
        c = np.array([mov.size[0] / 2.0, mov.size[1] / 2.0, 1.0])
        pc = Hn @ c
        if abs(pc[2]) > 1e-12:
            m.shift_x = float(pc[0] / pc[2] - c[0])
            m.shift_y = float(pc[1] / pc[2] - c[1])
        A = Hn[:2, :2]
        m.rotation_deg = float(np.degrees(np.arctan2(A[1, 0], A[0, 0])))
        det = np.linalg.det(A)
        m.scale = float(np.sqrt(det)) if det > 0 else float("nan")

    # Acceptance gates: collect every violated one, report the first as status.
    failures: List[Tuple[str, str]] = []
    if m.n_inliers < cfg.min_inliers:
        failures.append((TOO_FEW_INLIERS,
                         f"{m.n_inliers} inliers (need >= {cfg.min_inliers})"))
    if m.inlier_ratio < cfg.min_inlier_ratio:
        failures.append((LOW_INLIER_RATIO,
                         f"inlier ratio {m.inlier_ratio:.2f} (need >= {cfg.min_inlier_ratio:.2f}); "
                         f"matches are inconsistent with one rigid view change"))
    if not m.rms_error <= cfg.max_reproj_error:      # also catches NaN
        failures.append((HIGH_REPROJ_ERROR,
                         f"RMS reprojection error {m.rms_error:.2f}px (max {cfg.max_reproj_error:.2f}px); "
                         f"possible parallax, rolling shutter or distortion"))
    if m.coverage < cfg.min_inlier_coverage:
        failures.append((LOW_COVERAGE,
                         f"inliers cover {m.coverage:.1%} of the frame "
                         f"(need >= {cfg.min_inlier_coverage:.1%}); estimate would extrapolate"))
    for problem in check_homography_geometry(H, mov.size, cfg.max_scale_change):
        failures.append((DEGENERATE_GEOMETRY, problem))

    if failures:
        return PairResult(False, failures[0][0],
                          [f"{code}: {msg}" for code, msg in failures],
                          H, m, pts_ref, pts_mov, inl)
    return PairResult(True, OK, [], H, m, pts_ref, pts_mov, inl)


# --------------------------------------------------------------------------
# Sequential driver
# --------------------------------------------------------------------------

class PairwiseAligner:
    """Aligns each sampled frame to the previous one.

    Feed frames in order with :meth:`process`; the first call returns None,
    every later call returns the PairResult for (previous frame -> this frame),
    with ``H`` mapping this frame into the previous frame's coordinates.
    Features of the previous frame are cached so each frame is detected once.

    A failed pair is reported, not hidden. What to do about it (skip the frame,
    re-anchor to the last good frame, start a new segment) is a trajectory-level
    decision and belongs to Phase 3.
    """

    def __init__(self, cfg: Optional[MotionConfig] = None):
        self.cfg = cfg or MotionConfig()
        self.cfg.validate()
        self._prev_feats: Optional[FeatureSet] = None
        self._prev_index: Optional[int] = None

    def process(self, rec: FrameRecord) -> Optional[PairResult]:
        feats = detect_features(rec.image, self.cfg)
        result = None
        if self._prev_feats is not None:
            if feats.size != self._prev_feats.size:
                raise ValueError(
                    f"Frame size changed from {self._prev_feats.size} to {feats.size} "
                    f"at frame {rec.frame_index}; mosaicking needs constant frame size.")
            result = estimate_pair(self._prev_feats, feats, self.cfg)
            result.ref_frame_index = self._prev_index
            result.mov_frame_index = rec.frame_index
        self._prev_feats = feats
        self._prev_index = rec.frame_index
        return result

    def run(self, records: Iterable[FrameRecord]) -> Iterator[PairResult]:
        for rec in records:
            res = self.process(rec)
            if res is not None:
                yield res
