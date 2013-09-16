# coding: utf-8
from scipy import ndimage
import numpy as np
from environment import FoodField

from genome import VectorAgentChromosome, ACTUATORS, SENSES
from population import Population
from settings import SELECT_RATIO


class BaseMessiaChromosome(VectorAgentChromosome):
    GENES = VectorAgentChromosome.GENES

    def make_neuron(self, i, from_, bias):
        if from_:
            from_links, from_weights = zip(*from_)
            self.connections_reshaped[..., i, range(len(from_links))] = from_links
            self.weights_reshaped[..., i, range(len(from_links))] = from_weights
        self.biases[..., i] = bias

    def __init__(self, shape):
        super(BaseMessiaChromosome, self).__init__(shape)
        self.S = dict( (n, i) for i, n in enumerate(SENSES))
        self.A = dict( (n, self.hidden_size+i) for i, n in enumerate(ACTUATORS))
        self.connections_reshaped = self.connections.reshape(self.shape + (self.brain_size, self.num_connections))
        self.connections_reshaped[:] = self.overall_size # no connections
        self.weights_and_biases = self.weights.reshape(self.shape + (self.brain_size, self.num_connections+1))
        self.weights_and_biases[:] = 0
        self.weights_reshaped = self.weights_and_biases[..., 1:]
        self.biases = self.weights_and_biases[..., 0]
        self.mut_rate = 0.1
        self.mut_std = 0.1


class RabbitChromosome(BaseMessiaChromosome):
    GENES = BaseMessiaChromosome.GENES

    def __init__(self, shape):
        super(RabbitChromosome, self).__init__(shape)
        self.clockf1 = 620
        self.clockf2 = 100
        self.birth_health = 300
        self.herbivore = 1
        self.color = [0, 255, 0]
        A = self.A
        S = self.S
        self.make_neuron(A["eat"], [(S["food"], 1),
                                    ], -100)

        # self.make_neuron(A["walk"], [], 100)
        self.make_neuron(A["walk"], [
                                     # (S["food"], -1),
                                     (S["s_food_dx"], 10),
                                     # (S["s_red_dx"], -10),
                                     # (S["s_blue_dx"],  -10),
                                     ], 0)

        self.make_neuron(A["left"],  [(S["s_food_dy"],  100),
                                      (S["s_red_dy"], -5),
                                      (S["s_blue_dy"], -5),
                                      ], 0)
        self.make_neuron(A["right"], [(S["s_food_dy"], -100),
                                      (S["s_red_dy"], 5),
                                      (S["s_blue_dy"], 5),
                                      ], 0)


        self.make_neuron(A["attack"], [], -100)


class WolfChromosome(BaseMessiaChromosome):
    GENES = BaseMessiaChromosome.GENES

    def __init__(self, shape):
        super(WolfChromosome, self).__init__(shape)
        self.clockf1 = 620
        self.clockf2 = 100
        self.birth_health = 3000
        self.herbivore = 0
        self.color = [255, 0, 0]
        A = self.A
        S = self.S
        self.make_neuron(A["attack"], [(S["s_green"], 1),], -100)

        self.make_neuron(A["walk"], [], 100)
        # self.make_neuron(A["walk"], [(S["s_green_dx"], 1000000),
        #                              ], -100)

        self.make_neuron(A["left"],  [(S["s_green_dy"],  0.1),
                                      ], 0)
        self.make_neuron(A["right"], [(S["s_green_dy"], -0.1),
                                      ], 0)

#        self.make_neuron(A["eat"], [], -100)
#        self.make_neuron(A["attack"], [], 100)


class ModelFood(FoodField):
    def __init__(self, cells_w, cells_h, init_food_appear_prob=0.1):
        super(ModelFood, self).__init__(cells_w, cells_h, 0)
        self.grow()

    def grow(self, init_food_appear_prob=0):
        self._food[:] = 0
        self._food[self.cells_h/2, self.cells_w/2-10] = 100000
        ndimage.gaussian_filter(self._food, 3, output=self._food)


class PolulationWithMessia(Population):
    # def select(self, fitness, select_ratio=SELECT_RATIO):
    #     fitness_rank, dead_idxs = super(PolulationWithMessia, self).select(fitness, select_ratio)
        # # save messia from death
        # if not 0 in fitness_rank:
        #     dead_idxs[dead_idxs==0] = fitness_rank[-1]
        #     fitness_rank[-1] = 0
        # return fitness_rank, dead_idxs

    @classmethod
    def random(cls, number):
        population = super(PolulationWithMessia, cls).random(number)
        rabbits_n = int(round(len(population) * 0.3))
        rabbit_c = RabbitChromosome(rabbits_n)
        sel = slice(0, rabbits_n)
        population._agents.develop_from_chromosomes(rabbit_c, sel)
        rabbits = population._agents[sel]
        rabbits.health = rabbits.birth_health * 5

        wolves_n = len(population) - rabbits_n
        wolves_c = WolfChromosome(wolves_n)
        sel = slice(rabbits_n, None)
        population._agents.develop_from_chromosomes(wolves_c, sel)
        wolves = population._agents[sel]
        wolves.health = wolves.birth_health * 5
        return population
