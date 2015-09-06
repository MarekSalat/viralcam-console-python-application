from enum import Enum

__author__ = 'Marek'

import cv2
import numpy as np


class Trimap (Enum):
    foreground = 1
    background = 0
    unknown = 2


class ImageAligner:
    def __init__(self, first_image, second_image):
        self.first_image = first_image
        self.second_image = second_image

    def get_first_image(self, cropped=True):
        return self.first_image

    def get_second_image(self, cropped=True):
        return self.second_image


class AlphaMate:
    def __init__(self, image, control_image, input_trimap, image_aligner=None):
        self.input_trimap = input_trimap
        self.control_image = control_image
        self.image = image
        self.trimap = None
        self.aligned_image = None

        self.image_aligner = image_aligner if not image_aligner else ImageAligner(image, control_image)

    def get_background(self):
        pass

    def get_foreground(self):
        pass

    def get_trimap(self):
        return self.trimap

    def _get_trimap(self):
        if self.trimap:
            return self.trimap
        aligned_image = self.image_aligner.get_first_image()
        aligned_control_image = self.image_aligner.get_second_image()

        image_diff = aligned_image - aligned_control_image
