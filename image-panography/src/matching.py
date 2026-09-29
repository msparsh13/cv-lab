
import cv2


class FeatureMatcher:
    def __init__(self, ratio=0.75):
        self.ratio = ratio

        self.matcher = cv2.BFMatcher(
            cv2.NORM_L2,
        )

    def _ratio_test(self, desc1, desc2):
        raw_matches = self.matcher.knnMatch(
            desc1,
            desc2,
            k=2,
        )

        good_matches = []

        for pair in raw_matches:
            if len(pair) < 2:
                continue

            m, n = pair

            if m.distance < self.ratio * n.distance:
                good_matches.append(m)

        return good_matches

    def match(self, desc1, desc2):
        if desc1 is None or desc2 is None:
            return []

        forward_matches = self._ratio_test(
            desc1,
            desc2,
        )

        backward_matches = self._ratio_test(
            desc2,
            desc1,
        )

        # --------------------------------------------------
        # Build reverse lookup:
        #
        # image2 descriptor -> image1 descriptor
        # --------------------------------------------------

        reverse_map = {
            match.queryIdx: match.trainIdx
            for match in backward_matches
        }

        # --------------------------------------------------
        # Keep only mutual matches.
        #
        # forward:
        #   image1 descriptor -> image2 descriptor
        #
        # backward:
        #   image2 descriptor -> image1 descriptor
        # --------------------------------------------------

        mutual_matches = []

        for match in forward_matches:
            reverse_query = reverse_map.get(
                match.trainIdx
            )

            if reverse_query == match.queryIdx:
                mutual_matches.append(match)

        # --------------------------------------------------
        # Sort by descriptor distance.
        # --------------------------------------------------

        mutual_matches.sort(
            key=lambda match: match.distance
        )

        return mutual_matches
