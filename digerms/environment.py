# coding: utf-8
import math
import numpy as np
import scipy
from scipy.ndimage import _ni_support
import scipy.spatial
from scipy import ndimage

# food
from settings import *

PI4 = math.pi / 4
PI_minus_PI4 = math.pi *2 - PI4
_2PI = math.pi * 2

SMELLS = ['s_food', 's_red', 's_green', 's_blue', 's_bricks']
SENSES = (['food', 'clock1', 'clock2', 'rnd']
         + SMELLS
         + ['%s_%s' % (s,d) for s in SMELLS for d in ('dx', 'dy')]
         )

class Walls(np.ndarray):
    def __new__(cls, shape):
        assert len(shape) == 2
        return np.ndarray.__new__(cls, shape, np.bool)

    def __init__(self, shape):
        np.ndarray.__init__(self, shape, np.bool)
        self[:] = False

    @classmethod
    def load(cls, file):
        return np.load(file).astype(np.bool).view(cls)


class Gaussinator(object):
    def __init__(self, sigma, mode="wrap"):
        self.sigma = sigma
        self.mode = mode
        size = 1 + 2 * int(round(3 * self.sigma)) # шесть сигм )
        self.kernel = np.zeros(size, np.float64)
        self.kernel[size // 2] = 1.
        ndimage.gaussian_filter(self.kernel, self.sigma, mode="constant", output=self.kernel)

    def gauss(self, a):
        a = np.asarray(a)
        axes = range(a.ndim)
        for axis in axes:
            ndimage.correlate1d(a, self.kernel, axis, a, self.mode)

class SmellField(np.ndarray):
    def __new__(cls, shape, alpha=0.01, sigma=1):
        if len(shape) == 2:
            shape = shape + (1, )
        shape = shape + (3, )
        return np.ndarray.__new__(cls, shape, np.float32)

    def __init__(self, shape, alpha=0.1, sigma=1):
        # на каждый канал еще два канала с градиентом
        np.ndarray.__init__(self, shape, np.float32)
        self[:] = 0
        self.alpha = alpha
        self.sigma = sigma
        self.gaussinator = Gaussinator(sigma)

    def update(self, values):
        values_s = values
        if self.shape[:-1] != values.shape:
            values_s = values.reshape(self.shape[:-1])
        self[..., 0] = self[..., 0] * (1-self.alpha) + values_s * self.alpha
        for i in range(self.shape[2]):
            self.gaussinator.gauss(self[..., i, 0])
        return self

    def clear(self, where, component):
        self[..., component, 0][where] = 0

    def calc_gradient(self):
        for i in range(self.shape[2]):
            self[..., i, 1], self[..., i, 2] = np.gradient(self[..., i, 0])
        return self

    def gradient_display(self, component=0):
        colors = self[..., component, :].copy()
        colors[..., 0] *= 0.2
        colors[..., 1:] = 128 + colors[..., 1:] * 0.1
        return colors.clip(0, 255).astype(np.uint8)

    def all_colors_display(self, components=(1,2,3), scales=0.2):
        colors = self[..., components, 0] * scales
        return colors.clip(0, 255).astype(np.uint8)

    def color_display(self, component=0):
        colors = self[..., component, 0] * 0.2
        return colors.clip(0, 255).astype(np.uint8)


class FoodField(object):
    def __init__(self, cells_w, cells_h, init_food_appear_prob=0.1):
        self.cells_w = cells_w
        self.cells_h = cells_h
        self._food = np.zeros((cells_h, cells_w), np.float32)
        self.grow(init_food_appear_prob)

    def __mul__(self, other):
        return self._food * other

    def __getitem__(self, coords):
        return self._food[coords]

    def __setitem__(self, coords, value):
        self._food[coords] = value

    def grow(self, food_appear_prob=FOOD_APPEAR_PROB):
#        self._food[self._food > 0] += FOOD_GROWTH
#        grow = ndimage.uniform_filter(self._food, size=2)
        grow = ndimage.median_filter(self._food, size=3, mode="wrap") * FOOD_GROWTH
#        grow = ndimage.sobel(self._food, mode="wrap") * FOOD_GROWTH
#        grow = ndimage.prewitt(self._food, mode="wrap") * FOOD_GROWTH
#        grow = ndimage.laplace(self._food, mode="wrap") * FOOD_GROWTH
#        grow = ndimage.gaussian_laplace(self._food, 1, mode="wrap") * FOOD_GROWTH
        self._food +=  grow - FOOD_GROWTH

        num_appear = int(round(self._food.size * food_appear_prob))
        appear_idxs = np.random.randint(0, self._food.size, num_appear)
        new_food_ammounts = np.random.normal(FOOD_APPEAR_AMM, FOOD_APPEAR_AMM*0.1, num_appear)
        self._food.flat[appear_idxs] += new_food_ammounts.astype(np.int)
        self._food.clip(0, FOOD_MAX, out=self._food)

    def update(self):
        self.grow()

    def consume_at(self, cells):
        # FIXME несколько агентов могут съесть одну и ту же травинку, если едят одновременно
        food = self._food[ cells[:,0], cells[:,1] ]
        eaten = food.clip(0, FOOD_INTAKE)
        remains = (food - eaten).clip(0)
        assert np.all(remains >= 0)
        self._food[ cells[:,0], cells[:,1] ] = remains
        return eaten


class Environment(object):
    def __init__(self, walls, cell_size, init_food_appear_prob=0.1):
        self.walls = walls
        cells_h, cells_w = walls.shape
        self.cells_w = cells_w
        self.cells_h = cells_h
        self.cell_size = cell_size
        self.width = self.cells_w * cell_size
        self.height = self.cells_h * cell_size
        self.time = 0
        self.food = FoodField(cells_w, cells_h, init_food_appear_prob)
        self.smell = SmellField((cells_h, cells_w, len(SMELLS)))
        self.stats = None

    def set_stats(self, stats):
        self.stats = stats

    def consume_food(self, agent_vector):
        cells = self.cell_X[agent_vector]
        return self.food.consume_at(cells)

    def set_population(self, population):
        self.population = population
        population.set_env(self)
        self.add_new_agents(range(len(population)))

    def add_new_agents(self, agents):
        X = np.random.random((len(agents), 2))
        X[:, 0] *= self.width
        X[:, 1] *= self.height
        self.population.set_X(X, agents)

#    # spatial
    def get_nearest_agent_in(self, agents, radius_sq):
        nears = self.near_lists[agents, 0].copy()
        in_radius = self.near_dist[agents, 0] <= radius_sq
        return nears, in_radius

    def fetch_positions(self):
        self.X  = self.population.get_X()

    def calc_distances(self):
        # TODO can it be be faster if we do not need distances to too far agents?
        # maybe maintain agent lists for grid of size=MAX_DISTANCE/2?
        # maybe scipy.spatial.KDTree
        # Using triangle of matrix dont make any better
        self.dist = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(self.X, metric='sqeuclidean'))
        dist_tri = self.dist.copy()
        r = np.arange(dist_tri.shape[0])
        dist_tri[r, r] = np.inf

        self.near_lists = dist_tri.argmin(axis=1)[:, None]
        # argsort is slow.
        # self.near_lists = self.dist.argsort(axis=1)[:,1:5] # only 4 nearest
        row_idxs = np.arange(self.near_lists.shape[0])[:, None]
        self.near_dist = self.dist[row_idxs, self.near_lists]

    def calc_cell_pos(self):
        # FIXME если это нужно только один раз при вычислении кто сколько съел, то вынести это в траву.
        self.cell_X = (self.X / self.cell_size).astype(np.int)
        # self.cell_X = (self.X / self.cell_size).round().astype(np.int)
        self.cell_X[:, 0] %= self.cells_w
        self.cell_X[:, 1] %= self.cells_h
        self.cell_X = self.cell_X[:, ::-1] # наоборот: сначала строка (y), потом колонка (x)

    def check_walls(self):
        cells = self.cell_X
        in_walls = self.walls[cells[:, 0], cells[:, 1]]
        self.population._agents.reverse(in_walls)
        self.population._agents.move_by(in_walls, GRID_SCALE) # двигаем назад
        self.fetch_positions()
        self.calc_cell_pos()

    def set_smells(self):
        food = self.food._food[..., None]
        colors = self.population.get_colors()
        health = self.population.get_health()[:, None]
        walls = self.walls
        shape = list(food.shape)
        shape[-1] = 3
        color_smells = np.zeros(shape)
        color_smells[self.cell_X[:, 0], self.cell_X[:, 1], :] = colors * health
        smell_induct = np.dstack([food, color_smells, walls])
        self.smell.update(smell_induct)
        self.smell.clear(self.walls, [0,1,2,3])
        self.smell.calc_gradient()

    def update(self):
        self.time += 1
        if self.time % 2 == 0:
            self.stats.advance_frame()

        self.population.select_and_reproduce()
        self.food.update()
        self.fetch_positions()
        self.calc_cell_pos()
        self.check_walls()
        self.calc_distances()
        self.set_smells()
        self.set_senses()
        self.population.update()

    def get_smells_for_agents(self):
        smells = self.smell[self.cell_X[:, 0], self.cell_X[:, 1]].copy()
        dx = smells[..., 1]
        dy = smells[..., 2]
        ax, ay = self.population.get_angles()
        ax, ay = ax[..., None], ay[..., None]
        smells[..., 1] =   dx * ax - dy * ay
        smells[..., 2] = + dx * ay + dy * ax
        return smells

    def set_senses(self):
        senses = self.population.senses
        senses['rnd'] = np.random.random(len(self.population))
        senses['food'] = self.food[self.cell_X[:, 0], self.cell_X[:, 1]]
        senses['clock1'] = self.population.get_clock(int(1))
        senses['clock2'] = self.population.get_clock(int(2))
        smells = self.get_smells_for_agents()
        for i, s in enumerate(SMELLS):
            senses[s] = smells[..., i, 0]
            senses[s+'_dy'] = smells[..., i, 1]
            senses[s+'_dx'] = smells[..., i, 2]
