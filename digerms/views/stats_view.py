# coding: utf-8
from StringIO import StringIO
import numpy as np

from matplotlib import pyplot as plt

import pyglet
from pyglet import graphics
from pyglet import gl
from pyglet import text

from .base import Group, GraphicsObject
from settings import POPULATION_SIZE


class DynamicPlot(GraphicsObject):
    def __init__(self, x, y, width, height, label_text, label_pos, color, max_y=None):
        super(DynamicPlot, self).__init__(x, y)
        self.label_text = label_text
        self.label_pos = label_pos
        self.width = width
        self.height = height
        self.color = color
        self.max_y = max_y
        # prepopulate points
        self.data = np.zeros((self.width, 2), dtype=np.int)
        self.data[:, 0] = np.arange(0, self.width)

    def add_to_batch(self, batch=None, parent=None):
        colors = list(self.color) * self.width
        self.vertex_list = batch.add(self.width, gl.GL_LINE_STRIP,
                                     graphics.OrderedGroup(0, parent),
                                     'v2i/stream', ('c3B/static', colors))
        self.label = text.Label(self.label_text, x=self.pos.x + 8, y=self.label_pos,
                                font_size = 9,
                                color = list(self.color) + [255],
                                batch=batch, group=graphics.OrderedGroup(1, parent))

    def update(self, Y):
        # todo inefficient operations, memory usage, use numexpr
        max_y = self.max_y or Y.max()
        Y = np.array(Y)
        cur_y = Y[-1]
        Y = (Y * self.height / max_y)
        Y = (Y + self.pos.y).astype(np.int)
        self.data[:, 1] = Y
        self.vertex_list.vertices = self.data.flatten()
        self.label.text = "%s = %0.00f (of %0.00f)" % (self.label_text, cur_y, max_y)


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
            # "age_mean":      ((  0, 128,  255), None),
            "age_median":    ((  0,   0,  255), None),
            "age_amax":      ((  0, 255,  255), None),
            "gencount_amax":  ((  255, 0,  255), None),
        }
        self.vertex_lists = {}

    def add_to_batch(self, batch=None, parent=None):
        super(StatsView, self).add_to_batch(batch, parent)
        plots = []
        label_y_step = self.height // len(self.plots)
        for i, (plot_code, args) in enumerate(self.plots.iteritems()):
            plots.append(DynamicPlot(self.pos.x, self.pos.y, self.width, self.height,
                        plot_code, self.pos.y + self.height - i*18 - 18,
                        *args))
        self.add_objects(plots)

    def update(self):
        stats_history = self.stats.history()[:self.width]
        for plot, obj in zip(self.plots, self.objects):
            Y = stats_history[plot]
            obj.update(Y)
