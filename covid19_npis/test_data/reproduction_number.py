import numpy as np


def fsigmoid(x, a, b):
    b = np.expand_dims(b, axis=-1)
    return 1.0 / (1.0 + np.exp(np.multiply(-a, (x - b))))


class Change_point(object):
    def __init__(self, alpha, gamma_max, length, begin):
        self.alpha = alpha
        self.gamma_max = gamma_max
        self.length = length
        self.begin = begin

    def get_gamma(self, t):
        return fsigmoid(t, 4.0 / (self.length), self.begin) * self.gamma_max


def gamma_from_delta_t(t, begin, delta_t):
    sigmoid = 1.0 / (1.0 + np.exp(((t - begin) / delta_t * 4)))

    return sigmoid / np.linalg.norm(sigmoid, 1)
