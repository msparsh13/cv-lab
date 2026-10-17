"""Orchestration: video -> trajectory -> canvas -> panorama, with caching and reports."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import cv2
import numpy as np

from .config import (CanvasConfig, MotionConfig, RenderConfig, SamplingConfig,
                     TrajectoryConfig)
from .mosaic import FeatherBlender
from .trajectory import (CanvasInfo, PanoramaError, Trajectory, build_trajectory,
                         compute_canvas)
from .video_reader import VideoOpenError, VideoReader
from .viz import draw_trajectory
from .warping import warp_to_canvas

Log = Callable[[str], None]


@dataclass
class PanoramaResult:
    image: np.ndarray            # uint8 BGR
    valid: np.ndarray            # bool, True where any frame contributed
    canvas: CanvasInfo
    n_frames_used: int


def _peak_rss_mb() -> Optional[float]:
    try:
        import resource
        import sys
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return rss / (1024.0 * 1024.0) if sys.platform == "darwin" else rss / 1024.0
    except Exception:
        return None


def _video_identity(path: str) -> Dict[str, Any]:
    return {"path": os.path.abspath(path), "size_bytes": os.path.getsize(path)}


def render_panorama(video_path: str, traj: Trajectory, ccfg: CanvasConfig,
                    rcfg: RenderConfig, log: Optional[Log] = None) -> PanoramaResult:
    """Decode the video again and feather-blend the selected, aligned frames."""
    log = log or (lambda s: None)
    entries = traj.select(rcfg.use)
    if not entries:
        raise PanoramaError("No aligned frames to render.")
    canvas = compute_canvas([e.H for e in entries], traj.frame_size, ccfg)
    for w in canvas.warnings:
        log(f"WARNING: {w}")
    log(f"canvas {canvas.size[0]}x{canvas.size[1]}px "
        f"({canvas.pixels/1e6:.1f} Mpx, ~{FeatherBlender.bytes_needed(canvas.size)/1e6:.0f} MB "
        f"accumulators), {len(entries)} frames")

    by_index = {e.frame_index: e for e in entries}
    last = max(by_index)
    blender = FeatherBlender(canvas.size)
    scfg = SamplingConfig.from_dict(traj.meta["sampling_config"])
    done = 0
    with VideoReader(video_path, scfg) as vr:
        for rec in vr:
            if rec.frame_index > last:
                break
            e = by_index.get(rec.frame_index)
            if e is None:
                continue
            if rec.size != tuple(traj.frame_size):
                raise PanoramaError(
                    f"Frame {rec.frame_index} is {rec.size}, trajectory was built at "
                    f"{tuple(traj.frame_size)}; sampling/resizing settings changed.")
            wf = warp_to_canvas(rec.image, canvas.C @ e.H, canvas.size)
            if wf is not None:
                blender.add(wf)
            done += 1
            if done % 10 == 0 or done == len(by_index):
                log(f"  blended {done}/{len(by_index)} frames")
    if done != len(by_index):
        raise PanoramaError(f"Only {done} of {len(by_index)} frames could be re-read from "
                            f"the video; the file may have changed.")
    image, valid = blender.result()
    return PanoramaResult(image, valid, canvas, blender.n_frames)


def run_panorama(
    video_path: str,
    out_dir: str,
    scfg: SamplingConfig,
    mcfg: MotionConfig,
    tcfg: TrajectoryConfig,
    ccfg: CanvasConfig,
    rcfg: RenderConfig,
    reuse_trajectory: bool = False,
    log: Optional[Log] = None,
    out_name: str = "panorama.png",
) -> Dict[str, Any]:
    """Run the whole pipeline and write panorama, diagnostics and report to ``out_dir``.

    Returns the report dict (also saved as report.json). Raises PanoramaError
    subclasses on unrecoverable problems; a partial panorama (tracking lost)
    is returned with ``report["complete"] == False``.
    """
    log = log or (lambda s: None)
    for c in (scfg, mcfg, tcfg, ccfg, rcfg):
        c.validate()
    if not os.path.isfile(video_path):
        raise VideoOpenError(f"Video file not found: {video_path}")
    os.makedirs(out_dir, exist_ok=True)
    traj_path = os.path.join(out_dir, "trajectory.json")
    identity = _video_identity(video_path)

    t0 = time.perf_counter()
    reused = False
    traj: Optional[Trajectory] = None
    if reuse_trajectory and os.path.isfile(traj_path):
        cached = Trajectory.load(traj_path)
        same = (cached.meta.get("sampling_config") == scfg.to_dict()
                and cached.meta.get("video") == identity
                and cached.meta.get("motion_config") == mcfg.to_dict()
                and cached.meta.get("trajectory_config") == tcfg.to_dict())
        if same:
            traj, reused = cached, True
            log(f"reusing cached trajectory {traj_path}")
        else:
            log("cached trajectory was built with different settings or video; recomputing")

    if traj is None:
        def progress(n, rec, entry):
            if (n + 1) % 20 == 0:
                log(f"  aligned {n + 1} frames (frame {rec.frame_index})")
        with VideoReader(video_path, scfg) as vr:
            log(f"video {vr.info.width}x{vr.info.height} @ {vr.info.fps} fps, step {vr.step}")
            traj = build_trajectory(vr, mcfg, tcfg, progress)
        traj.meta.update({"sampling_config": scfg.to_dict(), "video": identity})
        traj.save(traj_path)
    t_traj = time.perf_counter() - t0
    for w in traj.warnings:
        log(f"WARNING: {w}")
    log(f"trajectory: {len(traj.entries)} aligned frames, {len(traj.keyframes())} keyframes, "
        f"{len(traj.rejected)} rejected")

    t1 = time.perf_counter()
    result = render_panorama(video_path, traj, ccfg, rcfg, log)
    t_render = time.perf_counter() - t1

    pano_path = os.path.join(out_dir, out_name)
    if not cv2.imwrite(pano_path, result.image):
        raise PanoramaError(f"Could not write panorama to '{pano_path}'.")
    diag_path = os.path.join(out_dir, "trajectory.png")
    cv2.imwrite(diag_path, draw_trajectory(traj))

    drift = [c["disagreement_px"] for c in traj.consistency]
    report = {
        "complete": traj.complete,
        "video": identity,
        "panorama": pano_path,
        "configs": {"sampling": scfg.to_dict(), "motion": mcfg.to_dict(),
                    "trajectory": tcfg.to_dict(), "canvas": ccfg.to_dict(),
                    "render": rcfg.to_dict()},
        "trajectory": {
            "aligned_frames": len(traj.entries), "keyframes": len(traj.keyframes()),
            "rejected_frames": len(traj.rejected), "lost_at": traj.lost_at,
            "reference_frame_index": traj.reference_frame_index,
            "median_inliers": float(np.median([e.n_inliers for e in traj.entries[1:]])),
            "consistency_checks": len(drift),
            "drift_mean_px": float(np.mean(drift)) if drift else None,
            "drift_max_px": float(np.max(drift)) if drift else None,
            "reused_cache": reused,
        },
        "canvas": result.canvas.to_dict(),
        "frames_blended": result.n_frames_used,
        "coverage": float(result.valid.mean()),
        "accumulator_bytes": FeatherBlender.bytes_needed(result.canvas.size),
        "peak_rss_mb": _peak_rss_mb(),
        "seconds": {"trajectory": round(t_traj, 3), "render": round(t_render, 3)},
        "warnings": traj.warnings + result.canvas.warnings,
    }
    with open(os.path.join(out_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=1)
    return report
