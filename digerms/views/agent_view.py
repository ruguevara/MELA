# coding: utf-8
import math
from pyglet import graphics
from pyglet.gl import GL_QUADS, GL_LINES
import numpy as np

PI4 = math.pi / 4
PI_minus_PI4 = math.pi *2 - PI4
_2PI = math.pi * 2

class PopulationRender(object):
    def __init__(self, batch, env, mode="full", debug=False, group=0):
        self.env = env
        self.batch = batch
        self.mode = mode
        self.scale = 8
        self.debug = debug
        self.shape = np.array([[-1,-1],[-1,1],[1,1],[1,-1]])
        agents_g = graphics.OrderedGroup(group)
        agents_links_g = graphics.OrderedGroup(group+1)
        self.vertex_list = self.batch.add(len(self.env.population) * 4, GL_QUADS, agents_g,
            'v2i/stream', 'c3B/stream'
        )
        self.links_list = self.batch.add(len(self.env.population) * 2, GL_LINES, agents_links_g,
            'v2i/stream', 'c4B/stream'
        )
        self.attack_colors = np.array([[255, 200, 128, 255],
                                       [255, 0, 0, 0]])

    def update(self):
        X = self.env.X.astype(np.int16)
        healthes = self.env.population.get_health().astype(np.float32)
        radiuses = (np.sqrt(healthes) / 10).clip(1, 50).astype(int)
        shapes = self.shape[np.newaxis,...] * radiuses[:,np.newaxis,np.newaxis]
        points = X[:,np.newaxis,...] + shapes
        colors = self.env.population.get_colors()
        colors = colors.reshape(colors.shape[0], 1, 3).repeat(4, 1)
        self.vertex_list.vertices = points.flatten()
        self.vertex_list.colors = colors.flatten()

        if self.debug:
            # draw links
            near_to = self.env.population._agents.near_to
            X_to = X[near_to]
            link_X_1 = X[:, None, :]
            link_X_2 = X_to[:, None, :]
            link_X = np.hstack([link_X_1, link_X_2])
            self.links_list.vertices = link_X.flatten()
            attacking_ok = self.env.population._agents.attacked_ok.astype(np.uint8)
            line_colors = attacking_ok[:, None, None] * self.attack_colors
            self.links_list.colors = line_colors.flatten()
