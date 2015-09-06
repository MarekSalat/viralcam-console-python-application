__author__ = 'Marek'


import cv2
from matplotlib import pyplot as plt


image = cv2.imread(r"dataset\IMAG0002.jpg", cv2.IMREAD_COLOR)

plt.imshow(image)
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()

