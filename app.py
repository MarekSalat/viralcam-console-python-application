from virtualcam.simple_matting.matting import AlphaMatte, ImageAligner
import cv2
import numpy as np
from matplotlib import pyplot as plt

__author__ = 'Marek'


def plot(image_, index):
    plt.subplot(2, 2, index)
    plt.imshow(cv2.cvtColor(image_, cv2.COLOR_BGR2RGB), interpolation="quadric")
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis


def blend_images(_image, __image, alpha):
    return cv2.addWeighted(_image, alpha, __image, 1 - alpha, 0)


image = cv2.imread(r"dataset\IMAG0001.jpg", cv2.IMREAD_COLOR)
control_image = cv2.imread(r"dataset\IMAG0002.jpg", cv2.IMREAD_COLOR)

matting = AlphaMatte(image, control_image)
trimap = matting.get_trimap()

plot(image, 1)
plot(control_image, 2)
# plot(trimap, 3)
control_image_transformed = ImageAligner(image, control_image).get_second_image()
plot(blend_images(image, control_image_transformed, 0.5), 3)


plt.subplot(2, 2, 4)
plt.imshow(cv2.threshold(cv2.cvtColor(trimap, cv2.COLOR_BGR2GRAY), 32, 255, cv2.THRESH_BINARY)[1], 'gray')

plt.show()
