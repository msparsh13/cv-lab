import cv2


class FeatureDetector:
    def __init__(self, max_features=5000):
        self.sift = cv2.SIFT_create(
            nfeatures=max_features,
        )

    def detect(self, image):
        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY,
        )

        keypoints, descriptors = (
            self.sift.detectAndCompute(
                gray,
                None,
            )
        )

        return keypoints, descriptors