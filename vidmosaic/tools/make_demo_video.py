"""Generate a synthetic panning video with known ground truth.

Writes:
  <out>.mp4            the panning video (mp4v codec, plays almost anywhere)
  <out>_scene.png      the full static background scene (ground truth for Phase 5)
  <out>_truth.json     per-frame camera offset and moving-object position

Model: a static textured scene; the camera is a WxH window sliding over it, so
frame k sees scene[y0_k : y0_k+H, x0_k : x0_k+W]. Between frames the true
homography is a pure translation by (-dx, -dy). A coloured 'car' moves through
the scene in scene coordinates, so it is a transient object in the background.

    python tools/make_demo_video.py --out demo_pan
"""
import argparse, json
import cv2
import numpy as np


def make_scene(w, h, seed=7):
    rng = np.random.default_rng(seed)
    # sky-to-ground gradient base
    base = np.zeros((h, w, 3), np.float32)
    g = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    base[..., 0] = 200 - 60 * g
    base[..., 1] = 170 - 20 * g
    base[..., 2] = 120 + 40 * g
    scene = base.astype(np.uint8)
    # multi-scale blobs for natural-looking texture (good for SIFT)
    for s, n, a in [(120, 60, 40), (40, 400, 50), (12, 1500, 60)]:
        layer = np.zeros((h, w, 3), np.float32)
        for _ in range(n):
            c = (int(rng.integers(0, w)), int(rng.integers(0, h)))
            col = rng.integers(-a, a, 3).tolist()
            cv2.circle(layer, c, int(rng.integers(s // 2, s)), col, -1)
        layer = cv2.GaussianBlur(layer, (0, 0), s / 6)
        scene = np.clip(scene.astype(np.float32) + layer, 0, 255).astype(np.uint8)
    # buildings / rectangles with edges and corners
    for _ in range(70):
        x, y = int(rng.integers(0, w - 120)), int(rng.integers(h // 3, h - 80))
        bw, bh = int(rng.integers(40, 160)), int(rng.integers(40, 200))
        col = rng.integers(40, 230, 3).tolist()
        cv2.rectangle(scene, (x, y), (x + bw, y + bh), col, -1)
        cv2.rectangle(scene, (x, y), (x + bw, y + bh), (20, 20, 20), 2)
        for wx in range(x + 8, x + bw - 12, 22):
            for wy in range(y + 8, y + bh - 12, 26):
                cv2.rectangle(scene, (wx, wy), (wx + 10, wy + 14), (230, 240, 250), -1)
    # text labels as extra distinctive features
    for i in range(0, w, 300):
        cv2.putText(scene, f"X{i}", (i + 20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.4,
                    (30, 30, 30), 3, cv2.LINE_AA)
    return scene


def draw_car(img, cx, cy, scale=1.0):
    """Draw a simple red car centred at (cx, cy); clipped by OpenCV."""
    w, h = int(120 * scale), int(44 * scale)
    x0, y0 = int(cx - w / 2), int(cy - h / 2)
    cv2.rectangle(img, (x0, y0 + h // 3), (x0 + w, y0 + h), (30, 30, 200), -1)
    cv2.rectangle(img, (x0 + w // 4, y0), (x0 + 3 * w // 4, y0 + h // 3 + 2), (40, 40, 220), -1)
    cv2.rectangle(img, (x0 + w // 3, y0 + 4), (x0 + 2 * w // 3, y0 + h // 3), (230, 220, 180), -1)
    for wx in (x0 + w // 5, x0 + 4 * w // 5):
        cv2.circle(img, (wx, y0 + h), int(9 * scale), (15, 15, 15), -1)
    cv2.rectangle(img, (x0, y0 + h // 3), (x0 + w, y0 + h), (0, 0, 0), 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="demo_pan")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=360)
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--pan-px", type=float, default=4.0, help="horizontal px/frame")
    ap.add_argument("--no-car", action="store_true")
    a = ap.parse_args()

    W, H = a.width, a.height
    scene_w = int(W + a.pan_px * a.frames + 40)
    scene_h = H + 60
    scene = make_scene(scene_w, scene_h)

    writer = cv2.VideoWriter(a.out + ".mp4", cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (W, H))
    if not writer.isOpened():
        raise SystemExit("Could not open mp4v writer in this OpenCV build.")

    truth = {"scene_size": [scene_w, scene_h], "frame_size": [W, H], "fps": a.fps,
             "frames": []}
    for k in range(a.frames):
        x0 = 20 + a.pan_px * k
        y0 = 30 + 12 * np.sin(2 * np.pi * k / 150)     # gentle vertical sway
        # exposure drift: slight brightness change over time
        gain = 1.0 + 0.06 * np.sin(2 * np.pi * k / 200)
        canvas = scene.copy()
        car = None
        if not a.no_car:
            # car crosses the scene faster than the pan, in scene coordinates
            cx = 100 + 7.0 * k
            cy = scene_h - 70
            draw_car(canvas, cx, cy)
            car = [float(cx), float(cy)]
        # sub-pixel window extraction via affine warp (translation only)
        M = np.float32([[1, 0, -x0], [0, 1, -y0]])
        frame = cv2.warpAffine(canvas, M, (W, H), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE)
        frame = np.clip(frame.astype(np.float32) * gain, 0, 255).astype(np.uint8)
        writer.write(frame)
        truth["frames"].append({"frame_index": k, "t": k / a.fps,
                                "x0": float(x0), "y0": float(y0),
                                "gain": float(gain), "car_scene_xy": car})
    writer.release()
    cv2.imwrite(a.out + "_scene.png", scene)   # note: scene is car-free ground truth
    with open(a.out + "_truth.json", "w") as f:
        json.dump(truth, f, indent=1)
    print(f"wrote {a.out}.mp4 ({a.frames} frames, {W}x{H}), "
          f"{a.out}_scene.png ({scene_w}x{scene_h}), {a.out}_truth.json")


if __name__ == "__main__":
    main()
