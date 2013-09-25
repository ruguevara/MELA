# coding: utf-8
from StringIO import StringIO
import numpy as np

from matplotlib import pyplot as plt

import pyglet
from pyglet import graphics
from pyglet import gl

from .base import Group, GraphicsObject
from settings import POPULATION_SIZE


class DynamicPlot(GraphicsObject):
    def __init__(self, x, y, width, height, color, max_y=None):
        super(DynamicPlot, self).__init__(x, y)
        self.width = width
        self.height = height
        self.color = color
        self.max_y = max_y
        # prepopulate points
        self.data = np.zeros((self.width, 2), dtype=np.int)
        self.data[:, 0] = np.arange(0, self.width)

    def add_to_batch(self, batch=None, parent=None):
        colors = list(self.color) * self.width
        print parent
        self.vertex_list = batch.add(self.width, gl.GL_LINE_STRIP,
                                     graphics.OrderedGroup(0, parent),
                                     'v2i/stream', ('c3B/static', colors))

    def update(self, Y):
        # todo inefficient operations, memory usage, use numexpr
        max_y = self.max_y or Y.max()
        Y = (Y * self.height / max_y)
        Y = (Y + self.pos.y).astype(np.int)
        self.data[:, 1] = Y
        self.vertex_list.vertices = self.data.flatten()


class StatsView(Group):
    def __init__(self, stats, x, y, width, height,
                 parameter="fitness_mean",
                 mode="full", debug=False):
        super(StatsView, self).__init__(x, y)
        self.stats = stats
        self.width = width
        self.height = height
        self.mode = mode
        self.debug = debug
        self.plots = {
            "fitness_mean":  ((255, 255, 255), None),
            "herbivore_sum": ((  0, 255,   0), POPULATION_SIZE),
            "age_median":    ((  0,   0,  255), None),
        }
        self.vertex_lists = {}

    def add_to_batch(self, batch=None, parent=None):
        super(StatsView, self).add_to_batch(batch, parent)
        self.add_objects([
            DynamicPlot(self.pos.x, self.pos.y, self.width, self.height, *args)
            for plot_code, args in self.plots.iteritems()
        ])

    def update(self):
        stats_history = self.stats.history()[:self.width]
        for plot, obj in zip(self.plots, self.objects):
            Y = stats_history[plot]
            obj.update(Y)
