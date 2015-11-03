from itertools import product
import cv2
from virtualcam.matting import AlphaMatte

import numpy as np
import scipy as sp
import scipy.sparse as sparse
import scipy.sparse.linalg

from scipy.linalg.matfuncs import inv

__author__ = 'Marek'


class LearningBasedMatting(AlphaMatte):
    def __init__(self, image, trimap, window_size=3, regularization_factor=800, lamb=0.0000001):
        """

        :param image:
        :param trimap: foreground should be set to AlphaMatte.TRIMAP_FOREGROUND, background to AlphaMatte.TRIMAP_BACKGROUND
                     and unknown region to AlphaMatte.TRIMAP_UNKNOWN
        :param window_size:
        :param regularization_factor: c contstat
        :param lamb: lambda value
        """

        super().__init__()
        self.image = image
        self.trimap = trimap
        self.window_size = window_size
        self.regularization_factor = regularization_factor
        self.lamb = lamb

    def get_alpha(self, *args, **kwargs):
        mask = cv2.erode(abs(self.trimap), np.ones((self.window_size, self.window_size), np.uint8), iterations=1)
        laplacian = self.get_laplacian(
            self.image,
            mask,
            self.window_size,
            self.lamb)

        regularization_matrix = self.regularization_matrix(
            self.trimap,
            self.regularization_factor)

        alpha_star = self.trimap.reshape(-1)
        identity = sparse.identity(self.trimap.shape[0] * self.trimap.shape[1])

        # solve quadratic cost function
        # aplha = (L + C + D)^-1 * (C*aplha^*)  => Ax = B => (L + C + D)^-1 * alpha = C*aplha^*
        alpha = sparse.linalg.spsolve(laplacian + regularization_matrix + identity * self.lamb,
                                      regularization_matrix * alpha_star)

        # reshape vector of an alphas to the image shape
        alpha = alpha.reshape(self.trimap.shape)

        # truncate to large or to small value
        alpha[alpha > AlphaMatte.TRIMAP_FOREGROUND] = AlphaMatte.TRIMAP_FOREGROUND
        alpha[alpha < AlphaMatte.TRIMAP_BACKGROUND] = AlphaMatte.TRIMAP_BACKGROUND

        # remap alpha from range [-1, 1] into the range [0, 1]
        return alpha * 0.5 + 0.5

    def get_laplacian(self, image, mask, window_size, lamb):
        """
            Return (m x n) x (m x n) sparse laplacian matrix based on image data and local window size.
            See equations (6) and (11) in the iccv2009 paper.
        :param image:
        :param mask:
        :param window_size: vec with 2-components showing size of local window for training
                            local linear models. Values will be obliged to be odd.
        :param lamb:        param of the regularization in training local linear model
        """

        # normalize image data to be in within [0, 1]
        normalized_image = image / 255.0
        feature_dimension = image.shape[2]  # 3

        image_shape = image.shape[0:2]
        image_pixels = image_shape[0] * image_shape[1]
        pixels_in_window = window_size * window_size
        half_window_size = round((window_size - 1) / 2)
        image_pixel_indices = np.array(range(image_pixels)).reshape(image_shape)

        kernel = LinearKernel()
        laplacian = sparse.lil_matrix((image_pixels, image_pixels))

        for y in range(half_window_size, image_shape[0] - half_window_size):
            print("\r", y, end="")
            for x in range(half_window_size, image_shape[1] - half_window_size):
                if mask[y, x]:
                    continue

                image_values_in_window = normalized_image[y - half_window_size:y + half_window_size + 1,
                                         x - half_window_size:x + half_window_size + 1, :]

                laplacian_coefficients = kernel.get_coefficients(
                    image_values_in_window.reshape(pixels_in_window, feature_dimension), lamb)

                indices = image_pixel_indices[y - half_window_size:y + half_window_size + 1,
                          x - half_window_size:x + half_window_size + 1].reshape(-1)

                coefs = laplacian_coefficients.shape[0]
                for l_coord, f_coord in zip(product(range(coefs), range(coefs)), product(indices, indices)):
                    laplacian[f_coord[0], f_coord[1]] = laplacian[f_coord[0], f_coord[1]] + laplacian_coefficients[
                        l_coord[0], l_coord[1]]
            # end x
        # end y

        return laplacian

    def regularization_matrix(self, trimap, regularization_factor):
        """
            Returns sparse diagonal matrix (m x n) x (m x n) where diagonal elements are set to regularization_factor
            if element belongs to the labeled set.
        :param trimap:
        :param regularization_factor:
        """
        # reshape trimap to array, take only those pixels with labels and set them to regularization matrix
        diagonal = abs(trimap.reshape(-1)) * regularization_factor

        return sparse.diags(diagonal, 0)


class Kernel:
    def get_coefficients(self, window_values, lamb):
        """
        :param window_values: (n x c) matrix where n is the number of pixel, c number of features
        :param lamb: regularization factor
        """
        raise NotImplemented("Method should be overridden.")


class LinearKernel(Kernel):
    def get_coefficients(self, window_values, lamb):
        """
        Calculates laplacian coefficients, i.e. (I-F)'*(I-F). And let Xi be n x (d+1) matrix where last dimensions
        is set to one. Then fi = (Xi*Xi' + lamb*I)^-1 * Xi*Xi'

        :param window_values: (n x c) matrix where n is the number of pixel, c number of features
        :param lamb: regularization factor
        """
        # create (n) x (d+1) matrix where last dimensions is set to one
        feature_xi = np.hstack((window_values, [[1] for i in range(window_values.shape[0])]))
        # identity n x n, last element is set to zero
        identity = np.eye(feature_xi.shape[0], feature_xi.shape[0])
        identity[identity.shape[0] - 2, identity.shape[1] - 1] = 0

        xi_times_xi_t = feature_xi.dot(feature_xi.T)
        fi = np.linalg.solve((xi_times_xi_t + identity * lamb).T, xi_times_xi_t.T).T

        identity_minus_f = np.eye(fi.shape[0], fi.shape[0]) - fi

        return identity_minus_f.T.dot(identity_minus_f)
