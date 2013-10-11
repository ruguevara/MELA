# coding: utf-8
import math
import random
from pyglet import graphics
from pyglet.gl import GL_QUADS, GL_LINES
from pyglet.window import key
import numpy as np
from messia import WolfChromosome, RabbitChromosome

from views.base import GraphicsObject

class PopulationView(GraphicsObject):
    def __init__(self, env, mode="full", debug=False):
        super(PopulationView, self).__init__()
        self.env = env
        self.mode = mode
        self.scale = 8
        self.debug = debug
        self.shape = np.array([[-1,-1],[-1,1],[1,1],[1,-1]])
        self.attack_colors = np.array([[255, 200, 128, 255],
                                       [255, 0, 0, 0]])

    def add_to_batch(self, batch=None, parent=0):
        self.vertex_list = batch.add(len(self.env.population) * 4, GL_QUADS,
                                     graphics.OrderedGroup(0, parent),
                                     'v2i/stream', 'c3B/stream'
        )
        self.links_list = batch.add(len(self.env.population) * 2, GL_LINES,
                                    graphics.OrderedGroup(1, parent),
                                    'v2i/stream', 'c4B/stream'
        )

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

    def launch_messia(self, messia_cls, n=1):
        population = self.env.population
        start = random.randint(0, len(self.env.population)-n)
        sel = range(start, start+n)
        messia = messia_cls(n)
        population._agents.develop_from_chromosomes(messia, sel)
        agent = population._agents[sel]
        agent.health = agent.birth_health * 5
        self.env.add_new_agents(sel)

    def on_key_press(self, symbol, modifiers):
        if symbol == key.W:
            # запустить волка
            self.launch_messia(WolfChromosome, len(self.env.population)//10)
        elif symbol == key.R:
            self.launch_messia(RabbitChromosome, len(self.env.population)//10)
