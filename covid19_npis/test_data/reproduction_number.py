import numpy as np


def fsigmoid(x, a, b):
    return 1.0 / (1.0 + np.exp(-a * (x - b)))


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


def get_R_t(times, R_0, cps):
    R_t = np.array([R_0])
    R_0 = np.array(R_0)
    for t in times:
        _sum = 0
        for cp in cps:
            _sum += cp.alpha * cp.get_gamma(t)
        R_t = np.append(R_t, [R_0 * np.exp(-_sum)], axis=0)
    return R_t
