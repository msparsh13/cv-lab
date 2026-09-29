import cv2
import numpy as np


class Stitcher:

    def _get_corners(self, image):
        h, w = image.shape[:2]

        return np.float32([
            [0, 0],
            [w, 0],
            [w, h],
            [0, h],
        ]).reshape(-1, 1, 2)

    def _compute_canvas(self, image1, image2, H):
        h1, w1 = image1.shape[:2]
        h2, w2 = image2.shape[:2]

        corners1 = self._get_corners(image1)
        corners2 = self._get_corners(image2)

        # H maps image1 -> image2.
        warped_corners1 = cv2.perspectiveTransform(
            corners1,
            H,
        )

        corners2 = corners2.reshape(-1, 2)
        warped_corners1 = warped_corners1.reshape(-1, 2)

        all_corners = np.vstack([
            warped_corners1,
            corners2,
        ])

        min_x = np.floor(
            all_corners[:, 0].min()
        )

        min_y = np.floor(
            all_corners[:, 1].min()
        )

        max_x = np.ceil(
            all_corners[:, 0].max()
        )

        max_y = np.ceil(
            all_corners[:, 1].max()
        )

        return (
            min_x,
            min_y,
            max_x,
            max_y,
        )

    def stitch(self, image1, image2, H):
        if H is None:
            raise ValueError(
                "Homography is None"
            )

        H = H.astype(np.float64)

        if abs(H[2, 2]) < 1e-12:
            raise ValueError(
                "Invalid homography"
            )

        H /= H[2, 2]

        # ----------------------------------------------------
        # Determine complete canvas
        # ----------------------------------------------------

        min_x, min_y, max_x, max_y = (
            self._compute_canvas(
                image1,
                image2,
                H,
            )
        )

        width = int(
            max_x - min_x
        )

        height = int(
            max_y - min_y
        )

        print(
            f"Canvas bounds: "
            f"x={min_x:.1f} -> {max_x:.1f}, "
            f"y={min_y:.1f} -> {max_y:.1f}"
        )

        print(
            f"Canvas size: "
            f"{width} x {height}"
        )

        if width <= 0 or height <= 0:
            raise RuntimeError(
                "Invalid panorama canvas"
            )

        # ----------------------------------------------------
        # Prevent accidental catastrophic allocation
        # ----------------------------------------------------

        max_canvas_dimension = 5000

        if (
            width > max_canvas_dimension
            or height > max_canvas_dimension
        ):
            raise RuntimeError(
                f"Panorama canvas is too large: "
                f"{width} x {height}"
            )

        # ----------------------------------------------------
        # Translation
        #
        # Move negative coordinates into the canvas.
        # ----------------------------------------------------

        translation = np.array([
            [1.0, 0.0, -min_x],
            [0.0, 1.0, -min_y],
            [0.0, 0.0, 1.0],
        ])

        H_translated = (
            translation @ H
        )

        # ----------------------------------------------------
        # Warp image1
        # ----------------------------------------------------

        warped1 = cv2.warpPerspective(
            image1,
            H_translated,
            (width, height),
        )

        # ----------------------------------------------------
        # Place image2 directly on canvas
        # ----------------------------------------------------

        canvas2 = np.zeros_like(
            warped1
        )

        offset_x = int(
            round(-min_x)
        )

        offset_y = int(
            round(-min_y)
        )

        canvas2[
            offset_y:offset_y + image2.shape[0],
            offset_x:offset_x + image2.shape[1],
        ] = image2

        # ----------------------------------------------------
        # Create masks
        # ----------------------------------------------------

        mask1 = np.ones(
            image1.shape[:2],
            dtype=np.uint8,
        ) * 255

        mask1 = cv2.warpPerspective(
            mask1,
            H_translated,
            (width, height),
        )

        mask2 = np.zeros(
            (height, width),
            dtype=np.uint8,
        )

        mask2[
            offset_y:offset_y + image2.shape[0],
            offset_x:offset_x + image2.shape[1],
        ] = 255

        # ----------------------------------------------------
        # Convert masks to float
        # ----------------------------------------------------

        mask1_float = (
            mask1.astype(np.float32)
            / 255.0
        )

        mask2_float = (
            mask2.astype(np.float32)
            / 255.0
        )

        # ----------------------------------------------------
        # Blend
        #
        # Non-overlap:
        #     keep whichever image exists.
        #
        # Overlap:
        #     average both images.
        # ----------------------------------------------------

        weight1 = mask1_float
        weight2 = mask2_float

        total_weight = (
            weight1 + weight2
        )

        total_weight[
            total_weight == 0
        ] = 1.0

        result = (
            warped1.astype(np.float32)
            * weight1[..., None]
            +
            canvas2.astype(np.float32)
            * weight2[..., None]
        ) / total_weight[..., None]

        result = np.clip(
            result,
            0,
            255,
        ).astype(np.uint8)

        return result