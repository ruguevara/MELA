# coding: utf-8
from ..agent import VectorAgent
import numpy as np
import math

#def agent_vec_setup():
#    "set up test fixtures"
#
#def teardown_func():
#    "tear down test fixtures"
#
#@with_setup(setup_func, teardown_func)
#def test():
#    "test ..."
from tools import eq_f

def test_creation():
    av = VectorAgent((10,))
    assert type(av) == VectorAgent
    assert av.x.shape[0] == 10

def test_move():
    av = VectorAgent((10,))
    av.x = np.arange(0, 10)
    av.y = np.arange(0, 100, 10)
    av.speed = 10
    av.ax = 0.5
    av.ay = -0.5

    assert av.speed.shape[0] == 10

    assert av[0].x == 0
    assert av[9].x == 9
    assert av.x[5] == 5 # так тоже можно
    assert av[9].y == 90, "%d != %d" % (av[9].y, 90)

    class FakeEnv(object):
        width = 100
        height = 100
    av.env = FakeEnv()
    av.move()

    assert av[0].x == 5
    assert av[9].y == -5 + 90, "%d != %d" % (av[9].y, -5 + 90)

def test_ops():
    av = VectorAgent((10,))
    av.speed = np.arange(0, 1, 0.1)

    av.speed[av.speed < 0.5] = 0
    av.speed = av.speed.clip(0, 0.8)

    eq_f(av[1].speed, 0.)
    eq_f(av[4].speed, 0.)
    eq_f(av[5].speed, 0.5)
    eq_f(av[8].speed, 0.8)
    eq_f(av[9].speed, 0.8)

def test_rotate():
    av = VectorAgent((8,))
    av.ax = 0.0
    av.ay = -1.0
    av.rotate(np.arange(0, 2 * math.pi, 2 * math.pi / 8))
    eq_f(av[1].ax, math.sqrt(0.5))
    eq_f(av[1].ay, -math.sqrt(0.5))
    eq_f(av[2].ax, 1)
    eq_f(av[2].ay, 0)
    eq_f(av[3].ax, math.sqrt(0.5))
    eq_f(av[3].ay, math.sqrt(0.5))
    eq_f(av[4].ax, 0)
    eq_f(av[4].ay, 1)
    eq_f(av[6].ax, -1)
    eq_f(av[6].ay, 0)

def test_health():
    a = VectorAgent(20)
    a.health = np.arange(0, 20)
    assert type(a[:10]) == VectorAgent
    e1 = a[:10].health.copy()
    a.add_energy(10, slice(None,10))
    e2 = a[:10].health
    print e1
    print e2
    assert np.all(e1 + 10 == e2)

    a.health = np.arange(10, 30)
    a.consume_energy(15)
    np.testing.assert_equal(a[:6].health, 0)
    assert np.all(a[6:].health > 0)

    a.kill(a.health > 10)
    assert np.all(a[-4:].is_dead())


def test_birth_ability():
    a = VectorAgent(10)
    a.health = np.linspace(100, 1000, 10)
    a.birth_health = np.linspace(10, 100, 10)
    assert np.all(a.can_give_birth() == 9)

def test_clock():
    a = VectorAgent(10)
    a.age = np.arange(10)
    a.clockf1[:] = 2
    a.clockf2[:] = 3
    assert np.all(a.get_clock(1)[::2] == 0)
    assert np.all(a.get_clock(1)[1::2] == 0.5)
    assert np.all(a.get_clock(2)[::3] == 0)
    assert np.all(a.get_clock(2)[1::3] == 1./3)


