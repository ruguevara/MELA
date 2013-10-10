# coding: utf-8
import numpy as np
import pandas as pd

class Statistics(pd.DataFrame):
    _fields = (
        ('birth_health_mean', np.float),
        ('birth_health_amin', np.float),
        ('birth_health_amax', np.float),
        ('herbivore_sum', np.float),
        ('total_eaten_mean', np.float),
        ('total_eaten_sum', np.float),
        ('age_median', np.float),
        ('age_mean', np.float),
        ('age_amax', np.float),
        ('gencount_amin', np.uint64),
        ('gencount_amax', np.uint64),
        ('gencount_mean', np.uint64),

        ('health_sum', np.float),
        ('health_mean', np.float),
        ('fitness_max', np.float),
        ('fitness_mean', np.float),

        ('step', np.uint64),
        ('born', np.uint16),
        ('random', np.uint16),
        ('deaths', np.uint16),
        ('attacking_sum', np.uint16),
        ('attacked_ok_sum', np.uint16),
        ('eating_sum', np.uint16),
        ('ready_to_born', np.uint16),

        ('primary_color_histogram_0', np.uint16),
        ('primary_color_histogram_1', np.uint16),
        ('primary_color_histogram_2', np.uint16),
        ('primary_color_histogram_3', np.uint16),
        ('primary_color_histogram_4', np.uint16),
        ('primary_color_histogram_5', np.uint16),
        ('primary_color_histogram_6', np.uint16),
        ('primary_color_histogram_7', np.uint16),
    )

    def __init__(self, shape):
        dtype = [(n,t) for n, t in self._fields]
        data = np.recarray(shape, dtype=dtype)
        data[:] = 0
        super(Statistics, self).__init__(data=data)
        self._pointer = 0

    def _get_current(self):
        return self.ix[self._pointer]

    def history(self):
        # TODO все равно есть копирование, может тогда лучше добавлять в конец и стирать начало?
        history = self.ix[self._pointer+1:]
        history = history.append(self.ix[0:self._pointer+1])
        return history

    def _set_current(self, values):
        if not isinstance(values, pd.Series):
            values = pd.Series(values)
        self.ix[self._pointer] = values

    current = property(_get_current, _set_current)

    def advance_frame(self):
        self._pointer = (self._pointer + 1) % self.shape[0]
        self.current = 0

    def log(self, **data):
        self.current = data
        # print self.current
