import cv2
import numpy as np


def draw_homography(
    image1,
    image2,
    H,
):
    h1, w1 = image1.shape[:2]

    corners = np.float32([
        [0, 0],
        [w1 - 1, 0],
        [w1 - 1, h1 - 1],
        [0, h1 - 1],
    ]).reshape(-1, 1, 2)

    transformed = cv2.perspectiveTransform(
        corners,
        H,
    ).reshape(-1, 2)

    print("Image 1 corners:")
    print(corners.reshape(-1, 2))

    print("\nProjected corners in image 2:")
    print(transformed)

    print("\nProjected width:")
    print(
        transformed[:, 0].max()
        - transformed[:, 0].min()
    )

    print("\nProjected height:")
    print(
        transformed[:, 1].max()
        - transformed[:, 1].min()
    )

    return transformed