import numpy as np

"""A Data Object which encodes the intrinsic state of the camera.

Attributes:
  K: the matrix which encodes the intrinsic camera parameters: focal length and principal offset 3x4 Matrix
  K_inv: the inverse matrix 3x3 Matrix
"""


class Intrinsic:
    def __init__(self, fx,fy,cx,cy):
        self.K = np.array([[fx,0,cx,0],[0,fy,cy,0],[0.,0.,1.,0.]])
        self.K_inv = Intrinsic.invert(self.K)

    def extract_fx(self):
        return self.K[0, 0]

    def extract_fy(self):
        return self.K[1, 1]

    def extract_cx(self):
        return self.K[0, 2]

    def extract_cy(self):
        return self.K[1, 2]

    @staticmethod
    def static_extract_fx(K):
        return K[0, 0]

    @staticmethod
    def static_extract_fy(K):
        return K[1, 1]

    @staticmethod
    def static_extract_cx(K):
        return K[0, 2]

    @staticmethod
    def static_extract_cy(K):
        return K[1, 2]

    # Returns 3x3 Intrinsic Camera Matrix
    @staticmethod
    def invert(K):
        fx = Intrinsic.static_extract_fx(K)
        fy = Intrinsic.static_extract_fy(K)
        fx_inv = 1 / fx
        fy_inv = 1 / fy
        cx_inv = -1 * Intrinsic.static_extract_cx(K) / fx
        cy_inv = -1 * Intrinsic.static_extract_cy(K) / fy

        K_inv = np.array([[fx_inv, 0, cx_inv],
                          [0, fy_inv, cy_inv],
                          [0, 0, 1]])

        return K_inv

    def scaly_by(self,scale_factor):
        self.K[0,0] /= scale_factor # Halving the image double focal length
        self.K[1,1] /= scale_factor
        self.K[0,2] *= scale_factor
        self.K[1,2] *= scale_factor

        self.K_inv = Intrinsic.invert(self.K)
