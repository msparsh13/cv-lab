#!/usr/bin/env python3
"""Make a panorama from a panning video.

    python make_panorama.py my_video.mp4                  # -> pano_out/panorama.png
    python make_panorama.py my_video.mp4 -o beach.png     # choose the output file
    python make_panorama.py my_video.mp4 --fps 5 --max-dim 800

Everything else (diagnostics, trajectory, report.json) is written next to the
output file in a folder named after it. For all options use:
    python -m vidmosaic.cli panorama --help

Tips for good results: pan slowly by rotating on the spot (do not walk), keep the
scene textured, and avoid large moving objects covering most of the frame.
"""
import argparse
import os
import sys

from vidmosaic import cli


def main() -> int:
    ap = argparse.ArgumentParser(description="Make a panorama from a panning video.")
    ap.add_argument("video", help="input video file")
    ap.add_argument("-o", "--output", default="panorama.png", help="output image (png/jpg)")
    ap.add_argument("--fps", type=float, default=10.0,
                    help="frames per second of video to analyse (default 10)")
    ap.add_argument("--max-dim", type=int, default=960,
                    help="longest frame side in pixels used for processing (default 960)")
    a, extra = ap.parse_known_args()

    out_abs = os.path.abspath(a.output)
    out_dir = os.path.join(os.path.dirname(out_abs),
                           os.path.splitext(os.path.basename(out_abs))[0] + "_work")
    argv = ["panorama", a.video, "--target-fps", str(a.fps), "--max-dim", str(a.max_dim),
            "--out-dir", out_dir, "--name", os.path.basename(out_abs)] + extra
    rc = cli.main(argv)
    if rc in (0, 1):
        final = os.path.join(out_dir, os.path.basename(out_abs))
        if os.path.abspath(final) != out_abs and os.path.exists(final):
            os.replace(final, out_abs)
        print(f"\nSaved: {out_abs}\nDiagnostics: {out_dir}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
