import numpy as np
from nose.tools import eq_, ok_
from tools import eq_f

from ..agent import VectorAgent
from ..genome import VectorChromosome, VectorBrainChromosome, VectorAgentChromosome
from ..brain import Brain, VectorBrain

def test_creation():
    vc = VectorChromosome(10)
    assert vc.mut_rate.shape == (10,)
    assert vc.mut_std.shape == (10,)
    assert vc._ranges.shape == (2, 2)

def test_random():
    vc = VectorChromosome(10)
    vc.randomize()
    for genes, range in zip(vc._genes, vc._ranges):
        assert np.all(range[0] <= genes) and np.all(genes <= range[1])

def test_mutation():
    class TestChromosome(VectorChromosome):
        GENES = VectorChromosome.GENES + (
            ('test', 1,   0, 1),
        )

    vc = TestChromosome(11)
    vc = vc.copy()
    def init_and_mutate():
        vc.test = 0.5
        vc.mut_rate = np.arange(0., 1.1, 0.1)
        vc.mut_std = np.arange(0.1, 1.2, 0.1)
        vc.mutate()

    num_iter = 1000
    totals = np.ndarray(vc.test.shape + tuple([num_iter]))
    for i in xrange(num_iter):
        init_and_mutate()
        totals[..., i] = vc.test

    stds = np.std(totals, 1)
    prev = stds[0]
    eq_f(prev, 0)
    for std in stds[1:]:
        assert std - prev > -0.1
        prev = std

def test_vec_brain_chromosome():
    c = VectorBrainChromosome.random(10)
    assert c.get_biases().shape == (10, 20)
    assert c.get_adj_matrix().shape == (10,20,30)

def test_vector_brain():
    c = VectorBrainChromosome.random(100)
    b = VectorBrain.from_chromosomes(c)
    for i in xrange(10):
        o = b.tick(np.random.random((100, c.input_size)))
        assert o.shape == (100, c.output_size)

def test_vector_agent_dev():
    c = VectorAgentChromosome.random(100)
    a = VectorAgent.from_chromosomes(c)
    assert np.any(a.ay > 0)
    assert np.any(a.ay < 0)
    assert np.any(a.ax < 0)
    assert np.any(a.color_r > 128)
    assert a.color_r.shape == (100,)
    assert np.all(a.health > 0)
