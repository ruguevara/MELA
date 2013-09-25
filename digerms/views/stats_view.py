# coding: utf-8
import numpy as np

from matplotlib import pyplot as plt

from pyglet import graphics
from pyglet import gl

from .base import Group, BitMapObject


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
        self.vertex_list = batch.add(self.width, gl.GL_LINE_STRIP,
                                     graphics.OrderedGroup(0, parent),
                                     'v2i/stream', ('c3B/static', self.colors)
        )

    def update(self):
        Y = np.array(self.stats.history()[self.parameter][:self.width])
        Y = (Y * self.height / Y.max()).astype(np.int)
        Y = self.pos.y + Y
        self.data[:, 1] = Y
        self.vertex_list.vertices = self.data.flatten()


class StatsViewSprite(BitMapObject):
    def __init__(self, stats, parameter="fitness_mean", **kwargs):
        super(StatsViewSprite, self).__init__(0, 0, scale=1)
        self.stats = stats
        self.parameter = parameter

    def get_data(self):
        walls = self.walls.astype(np.uint8) * 255
        f = np.zeros_like(walls)[..., None].repeat(3, 2)
        f[:] = 255
        return np.dstack([f, walls])

