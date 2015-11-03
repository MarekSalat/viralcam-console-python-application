import cv2
from virtualcam.learning_based_matting.matting import LearningBasedMatting
from matplotlib import pyplot as plt
import numpy as np
from virtualcam.matting import AlphaMatte

__author__ = 'Marek'

image = cv2.imread(r"..\..\dataset\troll.png", cv2.IMREAD_COLOR)
image = cv2.resize(image, (0, 0), fx=0.125, fy=0.125, interpolation=cv2.INTER_LINEAR)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
trimap = cv2.imread(r"..\..\dataset\trimap_troll.png", cv2.IMREAD_GRAYSCALE)
trimap = cv2.resize(trimap, (0, 0), fx=0.125, fy=0.125, interpolation=cv2.INTER_LINEAR)

foreground = trimap == 255
background = trimap == 0
trimap_mask = np.zeros(trimap.shape)
trimap_mask[foreground] = AlphaMatte.TRIMAP_FOREGROUND
trimap_mask[background] = AlphaMatte.TRIMAP_BACKGROUND

matting = LearningBasedMatting(image, trimap_mask)
alpha = matting.get_alpha()

plt.imshow(alpha * 255, 'gray')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis

plt.show()
