"""Command-line entry point.

    python -m vidmosaic.cli info    VIDEO
    python -m vidmosaic.cli preview VIDEO [--target-fps 2 | --step 10] \
        [--max-dim 960] [--out sheet.png] [--dump-dir frames/]
    python -m vidmosaic.cli align   VIDEO [--step 2] [--max-dim 640] \
        [--ratio 0.75] [--min-inliers 25] [--out-dir align_out]
    python -m vidmosaic.cli panorama VIDEO [--target-fps 10] [--max-dim 960] \
        [--out-dir pano_out] [--reuse-trajectory]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import List, Optional

import cv2

import numpy as np

from .config import (CanvasConfig, ConfigError, MotionConfig, RenderConfig,
                     SamplingConfig, TrajectoryConfig)
from .motion_estimation import PairwiseAligner
from .pipeline import run_panorama
from .trajectory import PanoramaError
from .video_reader import VideoOpenError, VideoReader, VideoReadError
from .viz import alignment_overlay, draw_matches, make_contact_sheet


def _add_sampling_args(p: argparse.ArgumentParser) -> None:
    g = p.add_mutually_exclusive_group()
    g.add_argument("--step", type=int, help="keep every N-th frame")
    g.add_argument("--target-fps", type=float, help="keep ~this many frames per second")
    p.add_argument("--start-frame", type=int, default=0)
    p.add_argument("--end-frame", type=int, default=None, help="exclusive")
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--max-dim", type=int, default=1280,
                   help="longest side in pixels (frames are only shrunk)")


def _config_from_args(a: argparse.Namespace) -> SamplingConfig:
    cfg = SamplingConfig(
        frame_step=a.step, target_fps=a.target_fps, start_frame=a.start_frame,
        end_frame=a.end_frame, max_frames=a.max_frames, max_dim=a.max_dim,
    )
    cfg.validate()
    return cfg


def cmd_info(a: argparse.Namespace) -> int:
    with VideoReader(a.video) as vr:
        i = vr.info
        print(json.dumps({
            "path": i.path, "width": i.width, "height": i.height, "fps": i.fps,
            "frame_count_estimate": i.frame_count, "duration_s": i.duration_s,
        }, indent=2))
    return 0


def cmd_preview(a: argparse.Namespace) -> int:
    cfg = _config_from_args(a)
    t0 = time.perf_counter()
    records: List = []
    with VideoReader(a.video, cfg) as vr:
        print(f"Video: {vr.info.width}x{vr.info.height} @ {vr.info.fps} fps, "
              f"step={vr.step}", file=sys.stderr)
        for rec in vr:
            records.append(rec)
            if len(records) % 25 == 0:
                print(f"  sampled {len(records)} frames...", file=sys.stderr)
    if not records:
        raise VideoReadError("No frames were sampled; check --start-frame/--end-frame/--step.")
    dt = time.perf_counter() - t0
    first = records[0]
    print(f"Sampled {len(records)} frames in {dt:.2f}s; "
          f"frame size {first.size[0]}x{first.size[1]} (scale {first.scale:.3f}); "
          f"indices {records[0].frame_index}..{records[-1].frame_index}")

    if a.dump_dir:
        os.makedirs(a.dump_dir, exist_ok=True)
        for r in records:
            cv2.imwrite(os.path.join(a.dump_dir, f"frame_{r.frame_index:06d}.png"), r.image)
    sheet = make_contact_sheet(records, columns=a.columns)
    if not cv2.imwrite(a.out, sheet):
        raise RuntimeError(f"Could not write contact sheet to '{a.out}'.")
    print(f"Wrote contact sheet: {a.out}")
    if a.show:
        cv2.imshow("preview", sheet)
        cv2.waitKey(0)
    return 0


def _motion_config_from_args(a: argparse.Namespace) -> MotionConfig:
    cfg = MotionConfig(
        max_features=a.max_features, matcher=a.matcher, ratio=a.ratio,
        min_matches=a.min_matches, ransac_thresh=a.ransac_thresh,
        min_inliers=a.min_inliers, min_inlier_ratio=a.min_inlier_ratio,
        max_reproj_error=a.max_reproj_error,
    )
    cfg.validate()
    return cfg


def cmd_align(a: argparse.Namespace) -> int:
    scfg = _config_from_args(a)
    mcfg = _motion_config_from_args(a)
    os.makedirs(a.out_dir, exist_ok=True)
    viz_dir = os.path.join(a.out_dir, "matches")
    os.makedirs(viz_dir, exist_ok=True)

    aligner = PairwiseAligner(mcfg)
    results = []
    prev_rec = None
    n_failed_saved = 0
    t0 = time.perf_counter()
    print(f"{'ref':>6} {'mov':>6}  {'status':<20} {'good':>5} {'inl':>5} "
          f"{'rms':>6} {'cov':>5} {'dx':>8} {'dy':>8} {'ms':>6}")
    with VideoReader(a.video, scfg) as vr:
        for rec in vr:
            res = aligner.process(rec)
            if res is not None:
                results.append(res)
                m = res.metrics
                ms = m.detect_ms + m.match_ms + m.ransac_ms
                print(f"{res.ref_frame_index:>6} {res.mov_frame_index:>6}  {res.status:<20} "
                      f"{m.n_good:>5} {m.n_inliers:>5} {m.rms_error:>6.2f} "
                      f"{m.coverage:>5.0%} {m.shift_x:>8.2f} {m.shift_y:>8.2f} {ms:>6.0f}")
                k = len(results) - 1
                save = (k == 0 or (a.viz_every and k % a.viz_every == 0)
                        or (not res.ok and n_failed_saved < a.max_failed_viz))
                if save and prev_rec is not None:
                    tag = f"{res.ref_frame_index:06d}_{res.mov_frame_index:06d}_{res.status}"
                    cv2.imwrite(os.path.join(viz_dir, f"matches_{tag}.jpg"),
                                draw_matches(prev_rec.image, rec.image, res))
                    if res.H is not None:
                        cv2.imwrite(os.path.join(viz_dir, f"overlay_{tag}.jpg"),
                                    alignment_overlay(prev_rec.image, rec.image, res.H))
                    if not res.ok:
                        n_failed_saved += 1
            prev_rec = rec
    if not results:
        raise VideoReadError("Need at least two sampled frames to estimate motion.")

    dt = time.perf_counter() - t0
    by_status = {}
    for r in results:
        by_status[r.status] = by_status.get(r.status, 0) + 1
    ok = [r for r in results if r.ok]
    summary = {
        "pairs": len(results), "ok": len(ok), "by_status": by_status,
        "median_inliers_ok": float(np.median([r.metrics.n_inliers for r in ok])) if ok else None,
        "median_rms_ok": float(np.median([r.metrics.rms_error for r in ok])) if ok else None,
        "total_seconds": round(dt, 3),
    }
    report = {
        "video": a.video,
        "sampling_config": scfg.to_dict(),
        "motion_config": mcfg.to_dict(),
        "summary": summary,
        "pairs": [r.to_dict() for r in results],
    }
    path = os.path.join(a.out_dir, "pairs.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=1)
    print(f"\n{len(ok)}/{len(results)} pairs accepted in {dt:.1f}s  {by_status}")
    print(f"Wrote {path} and match/overlay images in {viz_dir}/")
    return 0 if len(ok) == len(results) else 1


def cmd_panorama(a: argparse.Namespace) -> int:
    scfg = _config_from_args(a)
    mcfg = _motion_config_from_args(a)
    tcfg = TrajectoryConfig(keyframe_shift=a.keyframe_shift, reference=a.reference,
                            on_lost="error" if a.strict else "truncate",
                            max_consecutive_failures=a.max_failures)
    ccfg = CanvasConfig(max_pixels=int(a.max_pixels), on_exceed=a.on_exceed)
    rcfg = RenderConfig(use=a.use)
    for c in (tcfg, ccfg, rcfg):
        c.validate()
    log = lambda m: print(m, file=sys.stderr)
    report = run_panorama(a.video, a.out_dir, scfg, mcfg, tcfg, ccfg, rcfg,
                          reuse_trajectory=a.reuse_trajectory, log=log, out_name=a.name)
    t, c = report["trajectory"], report["canvas"]
    print(f"Panorama: {report['panorama']}  ({c['size'][0]}x{c['size'][1]}px, "
          f"{report['coverage']:.0%} covered, {report['frames_blended']} frames blended)")
    if t["drift_max_px"] is not None:
        print(f"Drift check: mean {t['drift_mean_px']:.2f}px, max {t['drift_max_px']:.2f}px "
              f"over {t['consistency_checks']} keyframe pairs")
    print(f"Time: {report['seconds']['trajectory']}s alignment, "
          f"{report['seconds']['render']}s blending")
    for w in report["warnings"]:
        print(f"WARNING: {w}", file=sys.stderr)
    if not report["complete"]:
        print("PARTIAL panorama: tracking was lost; see warnings above.", file=sys.stderr)
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="vidmosaic", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    pi = sub.add_parser("info", help="print video metadata")
    pi.add_argument("video")
    pi.set_defaults(func=cmd_info)

    pp = sub.add_parser("preview", help="sample frames and write a contact sheet")
    pp.add_argument("video")
    _add_sampling_args(pp)
    pp.add_argument("--out", default="preview.png")
    pp.add_argument("--columns", type=int, default=4)
    pp.add_argument("--dump-dir", default=None, help="also save each sampled frame as PNG")
    pp.add_argument("--show", action="store_true", help="open a window (needs a display)")
    pp.set_defaults(func=cmd_preview)

    pa = sub.add_parser("align", help="estimate camera motion between consecutive sampled frames")
    pa.add_argument("video")
    _add_sampling_args(pa)
    pa.add_argument("--out-dir", default="align_out")
    pa.add_argument("--matcher", choices=["bf", "flann"], default="bf")
    pa.add_argument("--max-features", type=int, default=3000)
    pa.add_argument("--ratio", type=float, default=0.75, help="Lowe ratio test threshold")
    pa.add_argument("--min-matches", type=int, default=30)
    pa.add_argument("--ransac-thresh", type=float, default=3.0, help="pixels")
    pa.add_argument("--min-inliers", type=int, default=25)
    pa.add_argument("--min-inlier-ratio", type=float, default=0.25)
    pa.add_argument("--max-reproj-error", type=float, default=2.0, help="RMS pixels")
    pa.add_argument("--viz-every", type=int, default=10,
                    help="save match images for every N-th pair (0 = only first/failed)")
    pa.add_argument("--max-failed-viz", type=int, default=20)
    pa.set_defaults(func=cmd_align)

    pn = sub.add_parser("panorama", help="build a panorama from a panning video")
    pn.add_argument("video")
    _add_sampling_args(pn)
    pn.add_argument("--out-dir", default="pano_out")
    pn.add_argument("--name", default="panorama.png", help="output image file name")
    pn.add_argument("--matcher", choices=["bf", "flann"], default="bf")
    pn.add_argument("--max-features", type=int, default=3000)
    pn.add_argument("--ratio", type=float, default=0.75)
    pn.add_argument("--min-matches", type=int, default=30)
    pn.add_argument("--ransac-thresh", type=float, default=3.0)
    pn.add_argument("--min-inliers", type=int, default=25)
    pn.add_argument("--min-inlier-ratio", type=float, default=0.25)
    pn.add_argument("--max-reproj-error", type=float, default=2.0)
    pn.add_argument("--keyframe-shift", type=float, default=0.2,
                    help="fraction of frame size between keyframes")
    pn.add_argument("--reference", choices=["first", "middle"], default="middle")
    pn.add_argument("--max-failures", type=int, default=5,
                    help="consecutive unalignable frames before tracking is lost")
    pn.add_argument("--strict", action="store_true",
                    help="error out if tracking is lost instead of writing a partial panorama")
    pn.add_argument("--max-pixels", type=float, default=20e6, help="canvas pixel limit")
    pn.add_argument("--on-exceed", choices=["scale", "error"], default="scale")
    pn.add_argument("--use", choices=["keyframes", "all"], default="keyframes")
    pn.add_argument("--reuse-trajectory", action="store_true",
                    help="reuse out-dir/trajectory.json if settings and video are unchanged")
    pn.set_defaults(func=cmd_panorama)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (VideoOpenError, VideoReadError, ConfigError, PanoramaError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
