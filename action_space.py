
import numpy as np
import pyflann

"""
This class represents a n-dimensional unit cube with a specific number of points embedded.
Points are distributed uniformly in the initialization. A search can be made using the
search_point function that returns the k (given) nearest neighbors of the input point.

Reference: Modified code from https://github.com/jimkon/Deep-Reinforcement-Learning-in-Large-Discrete-Action-Spaces
"""


class Space:

    def __init__(self, low, high, points):

        self._low = np.array(low)
        self._high = np.array(high)
        self._range = self._high - self._low
        self._dimensions = len(low)
        self._space_low = -1
        self._space_high = 1
        self._k = (self._space_high - self._space_low) / self._range
        self.__space = init_uniform_space([self._space_low] * self._dimensions,
                                          [self._space_high] * self._dimensions,
                                          points)
        self._flann = pyflann.FLANN()
        self.rebuild_flann()

    def rebuild_flann(self):
        self._index = self._flann.build_index(self.__space, algorithm='kdtree')

    def search_point(self, point, k):
        p_in = point
        if not isinstance(point, np.ndarray):
            p_in = np.array([p_in]).astype(np.float64)
        search_res, _ = self._flann.nn_index(p_in, k)
        knns = self.__space[search_res]
        return knns.squeeze(), search_res

    def import_point(self, point):
        return self._space_low + self._k * (point - self._low)

    def export_point(self, point):
        return self._low + (point - self._space_low) / self._k

    def get_space(self):
        return self.__space

    def shape(self):
        return self.__space.shape

    def get_number_of_actions(self):
        return self.shape()[0]


class Discrete_space(Space):
    """
    Discrete action space with n actions (the integers in the range [0, n))
    1, 2, ..., n-1, n

    Note: In gym, 'Discrete' object has no attribute 'high'
    """

    def __init__(self, n):
        super().__init__([0], [n-1], n)

    def export_point(self, point):
        return np.round(super().export_point(point)).astype(int)

def init_uniform_space(low, high, points):
    axis = np.linspace(low, high, points)
    return axis
