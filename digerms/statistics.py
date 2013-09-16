# coding: utf-8
import numpy as np
from genome import ZeroObjectArray

class Statistics(ZeroObjectArray):
    _fields = (
        ('energy', np.float),
        ('total_health', np.float),
        ('avg_health', np.float),
        ('max_health', np.float),
        ('total_eaten', np.float),
        ('avg_age', np.float),
        ('max_age', np.float),
        ('food', np.float),
        ('food_eaten', np.float),
        ('step', np.uint64),
        ('min_generation', np.uint64),
        ('avg_generation', np.uint64),
        ('max_generation', np.uint64),
        ('born', np.uint),
        ('random', np.uint),
        ('deaths', np.uint),
        ('attacks', np.uint),
        ('kills', np.uint),
    )

    def __init__(self, shape):
        super(Statistics, self).__init__(shape)
        self._pointer = 0

    def current(self):
        return self[self._pointer]

    def advance_frame(self):
        self._pointer = (self._pointer + 1) % self.shape[0]
