# coding: utf-8
import ctypes

import numpy as np
from pyglet import app
from pyglet.gl import  GL_TRIANGLES
from pyglet.window import key
from environment import Environment, Walls

from settings import GRID_SCALE, FOOD_INIT_PROB

from .agent_view import  PopulationView
from .base import GraphicsObject, Mode, Group, BatchGroup, Vec, BitMapObject
from statistics import Statistics
from .stats_view import StatsView

FOOD_MAX = 500


class WallsView(BitMapObject):
    def __init__(self, env, color=(64, 64, 0), **kwargs):
        self.color = color
        self.walls = env.walls
        super(WallsView, self).__init__(0, 0, scale=GRID_SCALE)

    def get_data(self):
        walls = self.walls.astype(np.uint8) * 255
        f = np.zeros_like(walls)[..., None].repeat(3, 2)
        f[:] = 255
        return np.dstack([f, walls])

    def _to_cell(self, x, y):
        return int(round(float(x)/GRID_SCALE)), int(round(float(y)/GRID_SCALE))

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        r, c = self._to_cell(x, y)
        dr, dc = self._to_cell(dx, dy)
        value = 1 if buttons==1 else 0
        steps = max(dr, dc, 1)
        inc_c = float(dc) / steps
        inc_r = float(dr) / steps
        cur_c = c
        cur_r = r
        # рисуем линию
        for i in range(steps):
            self.walls[int(round((cur_c))), int(round((cur_r)))] = value
            cur_c += inc_c
            cur_r += inc_r

    def on_mouse_press(self, x, y, buttons, modifiers):
        r, c = self._to_cell(x, y)
        value = 1 if buttons==1 else 0
        self.walls[c, r] = value



class SmellsView(BitMapObject):
    def __init__(self, env, **kwargs):
        self.env = env
        super(SmellsView, self).__init__(0, 0, scale=GRID_SCALE)

    def get_data(self):
        return self.env.smell.all_colors_display(scales=0.01)


class GrassView(GraphicsObject):
    def __init__(self, width, height, env, color=(0, 255, 0, 128), **kwargs):
        super(GrassView, self).__init__()
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
        self.colors = list(self.color) * (self.numboxes * pps)

    def add_to_batch(self, batch=None, parent=None):
        # должно всегда вызываться после созданя объекта,
        # при добавлении объекта в группу или нет
        self.vertex_list = batch.add(self.numboxes * self.pps,
                GL_TRIANGLES, parent,
                'v2i/dynamic', ('c4B/static', self.colors)
            )

    def draw(self):
        self.vertex_list.draw(GL_TRIANGLES)

    def update(self):
        food_h = (self.env.food * (float(self.scale_h) / FOOD_MAX)).astype(np.uint32).T
        grass = self.points.copy()
        grass[:, 1, 1] += food_h.flat
        # fast array copy
        # TODO map array instead of copy
        ctypes.memmove(self.vertex_list.vertices, grass.ctypes.data, len(self.vertex_list.vertices) * 4)


class EnvironmentView(Group):
    BG_COLOR = 30, 10, 5
    # Initialize the view
    def __init__(self, env, width, height, **kwargs):
        super(EnvironmentView, self).__init__()
        self.env = env
        self.env.update()
        self.width = width
        self.height = height

    def add_to_batch(self, batch=None, parent=None):
        # должно всегда вызываться после созданя объекта,
        # при добавлении объекта в группу или нет
        # Create view objects
        # TODO somehow when adding objects we must know about parent batch
        super(EnvironmentView, self).add_to_batch(batch, parent)
        kwargs = {}
        self.add_objects([
            SmellsView(self.env, **kwargs),
            GrassView(self.width, self.height, self.env,  **kwargs),
            WallsView(self.env,  **kwargs),
            PopulationView(self.env,  **kwargs),
        ])

    def simulate(self, dt=None, multiplier=1):
        for i in range(multiplier):
            self.env.update()


class ExperimentMode(Mode):
    def __init__(self, window, population, maze, stats_height=160, **kwagrs):
        # TODO ability to handle hiding statistics
        super(ExperimentMode, self).__init__()
        self.paused = False
        self.window = window
        self.width = window.width
        self.height = window.height
        self.groups = BatchGroup()

        grid_size = (self.height-stats_height) // GRID_SCALE, self.width // GRID_SCALE
        maze = maze if maze is not None else Walls(grid_size)
        env = Environment(maze, GRID_SCALE, FOOD_INIT_PROB)
        env.set_stats(Statistics.for_shape(self.width))
        env.set_population(population)
        # split window
        self.env_view = EnvironmentView(env, self.width, self.height-stats_height, **kwagrs)
        self.stats_view = StatsView(env.stats, 0, self.height-stats_height,
                                    self.width, stats_height, **kwagrs)
        self.groups.add_objects([self.env_view, self.stats_view])

    def simulate(self, delta):
        # may be called with faster rate than screen update at fps rate
        if not self.paused:
            self._do_simulate(delta)

    def _do_simulate(self, delta):
        # may be called with faster rate than screen update at fps rate
        self.groups.simulate(delta)

    def draw(self):
        self.groups.draw()

    def toggle_pause(self):
        self.paused = not self.paused

    def on_key_press(self, symbol, modifiers):
        if symbol == key.ESCAPE:
            app.exit()
        elif symbol == key.RIGHT:
            if self.paused:
                self._do_simulate(1)
            else:
                self.toggle_pause()
        elif symbol == key.SPACE:
            self.toggle_pause()
        else:
            self.groups.on_key_press(symbol, modifiers)

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        self.groups.on_mouse_drag(x, y, dx, dy, buttons, modifiers)

    def on_mouse_press(self, x, y, buttons, modifiers):
        self.groups.on_mouse_press(x, y, buttons, modifiers)


