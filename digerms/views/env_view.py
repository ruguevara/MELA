# coding: utf-8
import ctypes
from scipy import ndimage

import numpy as np
from pyglet import graphics, sprite, image
from pyglet.gl import  GL_TRIANGLES, GLuint
from pyglet.graphics import Batch
from pygarrayimage.arrayimage import ArrayInterfaceImage

from settings import GRID_SCALE

from views.agent_view import  PopulationRender

FOOD_MAX = 500

class WallsRender(object):
    def __init__(self, batch, env, group, color=(64, 64, 0)):
        self.color = color
        walls = env.walls.astype(np.uint8) * 255
        f = np.zeros_like(walls)[..., None].repeat(3, 2)
        f[:] = 255
        walls = np.dstack([f, walls])
        walls_ii = ArrayInterfaceImage(walls, allow_copy=True)
        texture = walls_ii.get_texture()
        self.walls_sprite = sprite.Sprite(texture, 0, 0, batch=batch, group=graphics.OrderedGroup(group))
        self.walls_sprite.scale = GRID_SCALE

    def update(self):
        pass


class GrassRender(object):
    def __init__(self, batch, width, height, env, group, mode="full", color=(0, 255, 0, 128)):
        self.color = color
        pps = 3
        self.pps = pps
        self.env = env
        self.width = width
        self.height = height
        self.scale_w = width / self.env.cells_w
        self.scale_h = height / self.env.cells_h
        X = np.arange(0, self.width, self.scale_w)
        X += self.scale_w / 2 - 1.5
        Y = np.arange(0, self.height, self.scale_h)
        self.numboxes = self.env.cells_w * self.env.cells_h
        self.points = np.empty((self.numboxes, pps, 2), dtype=np.uint32)
        self.points[:] = np.array(list(np.broadcast(*np.ix_(X, Y)))).reshape(self.numboxes, 1, 2)
        self.points[:, 1, 0] = self.points[:, 0, 0] + 2
        self.points[:, 2, 0] = self.points[:, 0, 0] + 3
        self.batch = batch
        self.mode = mode
        self.colors = list(self.color) * (self.numboxes * pps)
        self.vertex_list = self.batch.add(self.numboxes * self.pps,
                GL_TRIANGLES, graphics.OrderedGroup(group+1),
                'v2i/dynamic', ('c4B/static', self.colors)
            )
        smell = self.env.smell.all_colors_display()
        smell_ii = ArrayInterfaceImage(smell, allow_copy=False)
        texture = smell_ii.get_texture()
        self.smell_sprite = sprite.Sprite(texture, 0, 0, batch=batch, group=graphics.OrderedGroup(group))
        self.smell_sprite.scale = GRID_SCALE

    def update(self):
        food_h = (self.env.food * (float(self.scale_h) / FOOD_MAX)).astype(np.uint32).T
        grass = self.points.copy()
        grass[:, 1, 1] += food_h.flat
        # fast array copy
        # TODO map array instead of copy
        ctypes.memmove(self.vertex_list.vertices, grass.ctypes.data, len(self.vertex_list.vertices) * 4)
        smell = self.env.smell
        smell_colors = smell.all_colors_display(scales=0.01)
        smell_ii = ArrayInterfaceImage(smell_colors, allow_copy=False)
        self.smell_sprite._set_texture(smell_ii.get_texture())

# Implements the view
class EnvironmentRender(object):
    BG_COLOR = 30, 10, 5
    # Initialize the view
    def __init__(self, width, height, env, mode="full", debug=False, group=0):
#        super(EnvironmentRender,self).__init__(self)
        self.env = env
        self.width = width
        self.height = height
        # Create render objects
        self.mode = mode
        self.env.update()
        self.batch = Batch()
        self.grass = GrassRender(self.batch, self.width, self.height, self.env, group, mode)
        self.grass.update()
        self.walls = WallsRender(self.batch, self.env, group + 2)
        self.agents = PopulationRender(self.batch, self.env, mode, debug=True, group=group+3)

    def get_batch(self):
        self.grass.update()
        self.agents.update()
        return self.batch

    def update(self, dt):
#        for i in xrange(2):
        self.env.update()
