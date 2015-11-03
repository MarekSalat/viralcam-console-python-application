from virtualcam.simple_matting.matting import SimpleAlphaMatte, ImageAligner
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


image = cv2.imread(r"dataset\IMAG0005_aligned.png", cv2.IMREAD_COLOR)
control_image = cv2.imread(r"dataset\IMAG0006_aligned.png", cv2.IMREAD_COLOR)
# control_image_transformed = ImageAligner(image, control_image).get_second_image()

kernel = (13, 13)
blurred_image = cv2.GaussianBlur(image, kernel, 0)
blurred_control_image = cv2.GaussianBlur(control_image, kernel, 0)

matting = SimpleAlphaMatte(blurred_image, blurred_control_image)
trimap = matting.get_alpha()

plot(image, 1)
plot(control_image, 2)
plot(blend_images(image, control_image, 0.5), 3)


plt.subplot(2, 2, 4)

result_trimap = cv2.threshold(cv2.cvtColor(trimap, cv2.COLOR_BGR2GRAY), 32, 255, cv2.THRESH_BINARY)[1]

# result_trimap = cv2.dilate(result_trimap, (255, 255), iterations=64)
# result_trimap = cv2.erode(result_trimap, (255, 255), iterations=64)

# result_trimap = cv2.morphologyEx(result_trimap, cv2.MORPH_OPEN, (255, 255))
# result_trimap = cv2.morphologyEx(result_trimap, cv2.MORPH_CLOSE, (255, 255))

plt.imshow(result_trimap, 'gray')

plt.show()
