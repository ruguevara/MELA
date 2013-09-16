import numpy as np
from nose.tools import eq_, ok_
from population import Population
from tools import eq_f

from ..environment import Environment, FOOD_MAX
from ..agent import VectorAgent
#

def setup():
    n = 100
    population = Population.random(n)
    assert len(population) == n

    env = Environment(10, 20, 8)
    env.set_population(population)
    population.set_X(np.hstack( [np.linspace(0, 20*8-1, 100)[:, np.newaxis],
                                 np.linspace(0, 10*8-1, 100)[:, np.newaxis]]))
    a = population._agents
    a.ax = 1
    a.ay = 0
    a.speed = 1
    return population

def test_population():
    population = setup()
    env = population.env
    a = population._agents

    env.fetch_positions()
    assert env.X.shape == (100, 2)
    np.testing.assert_array_equal(a.x, env.X[:, 0])
    np.testing.assert_array_equal(a.y, env.X[:, 1])

    a.move()
    env.fetch_positions()
    assert (a[0].x, a[0].y) == (1., 0.)
    assert (a[-1].x, a[-1].y) == (0., 79.)

    env.calc_distances()
    dist = env.dist
    np.testing.assert_array_equal(dist.diagonal(), 0)
    a0a1_dist_sq = (a[0].x - a[1].x)**2 + (a[0].y - a[1].y)**2
    eq_f(dist[0, 1], a0a1_dist_sq)

    assert env.near_lists[0, 0] == 1
    eq_f(env.near_dist[0, 0], a0a1_dist_sq)
    eq_f(env.near_dist[1, 0], a0a1_dist_sq)
    eq_f(env.near_dist[0, 1], a0a1_dist_sq * 4)

    env.calc_cell_pos()
    env.food.update()

    env.food._food[:] = FOOD_MAX
    env.food._food[0,0] = 0

    a.health = 10
    a[:50].eating = True
    a.eat_grass()
    assert np.all(a[:2].health == 10)
    assert np.all( a[2:50].health == 210)
    assert np.all( a[50:].health == 10)

    a.update()

def test_senses():
    population = setup()
    env = population.env
    a = population._agents
    a.age = np.arange(100)
    a.clockf1[:] = 5

    env.food._food[:] = FOOD_MAX / 2

    env.fetch_positions()
    env.calc_cell_pos()
    env.set_senses()

    assert np.all(a.senses.food == FOOD_MAX / 2)
    assert np.all(a.senses.clock1[::5] == 0)
    assert np.all(a.senses.clock1[3::5] == 0.6)

    assert abs(a.senses.rnd.mean() - 0.5) < 0.1
    assert np.all(a.senses.clock1[3::5] == 0.6)

def test_colors():
    population = setup()
    env = population.env
    a = population._agents
    assert population.get_colors().shape == (100,3)

def test_hunt():
    population = setup()
    env = population.env
    env.fetch_positions()
    env.calc_distances()
    a = population._agents
    a.herbivore = True

    a[0].attacking = True
    a[0].herbivore = False
    a[0].health = 10
    a[1].health = 7

    a[3].attacking = True
    a[3].herbivore = False
    a[3].health = 10
    a[4].health = 15

    a[6].attacking = True
    a[6].herbivore = False
    a[6].health = 10
    a[7].attacking = True
    a[7].herbivore = False
    a[7].health = 7

    a.check_attack()
    np.testing.assert_array_equal(a[[0,1,3,4,6,7]].health, [14, 0, 21, 0, 13, 0])

def test_selection():
    population = setup()
    env = population.env
    env.fetch_positions()
    env.calc_distances()
    a = population._agents
    # TODO
#    assert 1==2
