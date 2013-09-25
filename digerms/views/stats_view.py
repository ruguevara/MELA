# coding: utf-8
import numpy as np

from pyglet import graphics
from pyglet.gl import GL_QUADS, GL_LINES, GL_LINE_LOOP

from .base import Group

class StatsView(Group):
    def __init__(self, stats, width, height,
                 color=(255, 255, 255), parameter="fitness_mean",
                 mode="full", debug=False):
        super(StatsView, self).__init__()
        self.stats = stats
        self.width = width
        self.height = height
        self.mode = mode
        self.debug = debug
        self.parameter = parameter
        self.color = color
        self.colors = list(self.color) * self.width
        self.data = np.zeros((self.width, 2), np.int)
        self.data[:, 0] = np.arange(0, self.width)

    def add_to_batch(self, batch=None, parent=None):
        # TODO сделать через спрайт, так как рисовать графики самому — это велосипедизм
        self.vertex_list = batch.add(self.width, GL_LINE_LOOP,
                                     graphics.OrderedGroup(0, parent),
                                     'v2i/stream', ('c3B/static', self.colors)
        )

    def update(self):
        Y = np.array(self.stats[self.parameter][:self.width])
        Y = (Y * self.height / Y.max()).astype(np.int)
        Y = self.pos.y + Y
        self.data[:, 1] = Y
        self.vertex_list.vertices = self.data.flatten()
