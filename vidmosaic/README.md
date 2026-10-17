# vidmosaic — Phases 1–4: video reader, alignment, trajectory, panorama

Part of the Video Panorama Mosaic / Temporal Summarization project.
Phase 1: decoding, sampling, resizing, timestamps.
Phase 2: SIFT + ratio test + RANSAC homography between frames, with quality gates and diagnostics.
Phase 3: keyframe trajectory, drift check, canvas bounds with memory limits.
Phase 4: warping with validity masks and streaming feather blending -> **panorama**.
Not yet implemented: background mosaic / temporal median (Phase 5), temporal mosaic (6), exposure compensation and loop closure (7), UI (8).

## Run

    pip install -r requirements.txt
    python -m unittest discover -s tests -t . -v        # also works with pytest
    python -m vidmosaic.cli info    pan.mp4
    python tools/make_demo_video.py --out demo_pan      # synthetic pan with ground truth
    python -m vidmosaic.cli preview pan.mp4 --target-fps 2 --max-dim 960 --out sheet.png
    python -m vidmosaic.cli align   demo_pan.mp4 --step 3 --out-dir align_out   # Phase 2 diagnostics
    python make_panorama.py demo_pan.mp4 -o pano.png                            # the panorama
    python -m vidmosaic.cli panorama demo_pan.mp4 --target-fps 10 --out-dir pano_out --reuse-trajectory
    python -m vidmosaic.cli preview pan.mp4 --step 10 --start-frame 100 --end-frame 400 --dump-dir frames/

## Conventions (relied on by later phases)

- Images: `uint8`, shape (H, W, 3), BGR.
- `frame_index`: 0-based index in the *source video*; never renumbered.
- `sample_index`: 0-based index in the sampled sequence.
- `timestamp_s = frame_index / fps` (exact for constant-frame-rate video);
  falls back to the decoder's position if the container reports no FPS.
- Range `[start_frame, end_frame)`; the step grid is anchored at `start_frame`.
- `scale = resized / original` in (0, 1]; frames are only shrunk, never enlarged.
  Divide resized pixel coordinates by `scale` to get original-resolution coordinates.
- Decoding is sequential (no seeking), because seeking is inaccurate in many codecs.

## Known limitations

- `frame_count` from the container is an estimate for some formats; the reader
  therefore stops on the first failed `grab()` rather than trusting it.
- Variable-frame-rate video: `frame_index / fps` is approximate there. Phase 1
  does not detect VFR.
- Rotation metadata (phone videos) is handled by OpenCV's backend, which is not
  consistent across builds; check the first preview if orientation looks wrong.

## Phase 2 conventions

- Homography `H` maps the **moving** frame into the **reference** frame:
  `p_ref ~ H @ [x, y, 1]`. For consecutive frames the reference is the previous
  frame, so `H_0k = H_01 @ H_12 @ ... @ H_(k-1,k)` maps frame k into frame 0.
- Coordinates are in the resized frame; divide by `FrameRecord.scale` for original pixels.
- Reprojection error = `||H p_mov - p_ref||` over RANSAC inliers (RMS reported).
- A pair is usable only if `result.ok`. Rejected pairs keep a status code
  (`insufficient_features`, `insufficient_matches`, `homography_failed`,
  `too_few_inliers`, `low_inlier_ratio`, `high_reproj_error`, `low_coverage`,
  `degenerate_geometry`) and human-readable reasons.
- `align` writes `pairs.json` (configs + per-pair metrics, reproducible),
  `matches_*.jpg` (green = inliers, red = RANSAC outliers) and `overlay_*.jpg`
  (|reference - warped moving frame|; dark blue = not covered by the warp).
  Exit code 0 only if every pair was accepted.

## Known limitations of the single-homography model

Exact only for a camera rotating about its optical centre or a planar scene.
Violated by parallax/camera translation, zoom, rolling shutter, lens distortion and
independently moving objects (the latter are rejected as RANSAC outliers while the
static background dominates). The gates catch many of these indirectly but not all:
a smooth, slowly accumulating error (drift) passes every per-pair gate and is a
Phase 3 problem.

## Measured on the synthetic demo (640x360, CPU)

- Per pair: SIFT ~130-160 ms, matching 25-45 ms, RANSAC ~1 ms. Detection dominates;
  matcher choice (bf vs flann) barely matters at this size. No optimisation needed yet.
- Accuracy vs ground truth: ~0.1 px shift error from 97% down to 50% overlap,
  ~0.7 px at 25% overlap, rejected (`low_inlier_ratio`) at ~6% overlap.

## Phases 3-4: panorama

Pipeline: sample frames -> align each to the last *keyframe* (a frame that moved >= 20% of
the frame size) -> chain into one coordinate system (middle keyframe = reference) ->
canvas bounds (+ offset for negative coordinates, size limits) -> warp each keyframe into
its footprint only -> distance-weighted feather blend (two float32 accumulators, 16 B/px).

- `H_k` maps frame k -> reference; `H_b = H_a @ H_ab`. Canvas matrix `C = S @ T` maps
  reference -> canvas; a frame is warped with `C @ H_k`.
- Pixel validity: a canvas pixel is valid iff its centre maps inside the source image
  (all-ones image warped with nearest sampling and a zero border). Black content is
  still valid; empty canvas is not. Uncovered pixels are black in the PNG; `coverage`
  in `report.json` says how much of the canvas is covered.
- Failure policy: an unalignable frame is skipped; after 5 in a row tracking is lost and
  a PARTIAL panorama is written (exit code 1, warning in `report.json`). `--strict` raises instead.
- Canvas limits (default 20 Mpx, 20000 px per side): the panorama is uniformly scaled
  down with a warning (`--on-exceed error` to refuse instead).
- Drift: each new keyframe is also aligned directly to the keyframe two back; the
  disagreement with the chained result (px) is reported (`drift_mean_px`, `drift_max_px`)
  and warned about above 3 px. It is detected, not corrected (no loop closure yet).
- Caching: `trajectory.json` is reused with `--reuse-trajectory` only if video, sampling
  and motion/trajectory settings are unchanged.
- Outputs in `--out-dir`: `panorama.png`, `trajectory.png` (keyframe footprints and camera
  path), `trajectory.json`, `report.json` (configs, canvas, memory, timings, warnings).

Limitations: planar projection stretches for pans approaching 90 degrees (no
cylindrical/spherical warp); moving objects leave faint ghosts under feathering (Phase 5
removes them); exposure changes are only smoothed by the feather, not compensated;
needs a camera that rotates rather than translates.

## Measured on the synthetic demo

100 frames chained: max camera-position error 0.27 px vs ground truth; the panorama
aligns with the true scene to within 0.06 px (phase correlation); drift check 0.6 px mean.
