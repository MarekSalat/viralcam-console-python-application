from enum import Enum
from virtualcam.matting import AlphaMatte

__author__ = 'Marek'

import cv2
import numpy as np


class Trimap (Enum):
    foreground = 1
    background = 0
    unknown = 2


class ImageAligner:
    def __init__(self, first_image, second_image, iterations=100, termination_eps=1e-11):
        self.first_image = first_image
        self.second_image = second_image
        self.warp_mode = cv2.MOTION_HOMOGRAPHY
        self.warp_matrix = np.eye(3, 3, dtype=np.float32)
        self.warp_matrix_calculated = False
        self.number_of_iterations = iterations
        self.termination_eps = termination_eps
        self.criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            self.number_of_iterations,
            self.termination_eps)

    def calculate_warp_matrix(self):
        if self.warp_matrix_calculated:
            return self.warp_matrix

        (cc, self.warp_matrix) = cv2.findTransformECC(
            cv2.cvtColor(self.first_image, cv2.COLOR_BGR2GRAY),
            cv2.cvtColor(self.second_image, cv2.COLOR_BGR2GRAY),
            self.warp_matrix,
            self.warp_mode,
            self.criteria
        )
        self.warp_matrix_calculated = True

        return self.warp_matrix

    def get_second_image(self):
        shape = self.second_image.shape
        return cv2.warpPerspective(
            self.second_image,
            self.calculate_warp_matrix(),
            (shape[1], shape[0]),
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)


class SimpleAlphaMatte (AlphaMatte):
    def __init__(self, image, control_image, input_trimap=None, image_aligner=None):
        self.input_trimap = input_trimap
        self.control_image = control_image
        self.image = image
        self.trimap = None
        self.aligned_image = None

        self.image_aligner = image_aligner if not image_aligner else ImageAligner(image, control_image)

    def get_alpha(self):
        image_converted = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        control_image_converted = cv2.cvtColor(self.control_image, cv2.COLOR_BGR2HSV)
        result = cv2.absdiff(image_converted, control_image_converted)
        return cv2.cvtColor(result, cv2.COLOR_HSV2BGR)
