# coding: utf-8
import math

import numpy as np

from brain import VectorBrain
from genome import ACTUATORS, ObjectArray, \
                   SENSES, VectorAgentChromosome, FloatObjectArray
from settings import MAX_SPEED, ROT_SPEED, ATTACK_RADIUS, ATTACK_RADIUS_SQ, MAX_HEALTH, ENERGY_PER_TURN, ATTACK_ENERGY, HUNT_KPD, BIRTH_KPD, CARN_SPEED_HANDICAP

class VectorSenses(FloatObjectArray):
    _fields = SENSES

class VectorActuators(FloatObjectArray):
    _fields = ACTUATORS

class VectorAgent(ObjectArray):
    _fields = (
        ('color_r', np.uint8, 0),
        ('color_g', np.uint8, 0),
        ('color_b', np.uint8, 0),
        ('clockf1', np.float32, 0),
        ('clockf2', np.float32, 0),
        ('birth_health', np.int32, 0),
        ('gencount', np.uint32, 1),
        # calculated
        ('herbivore', bool, 0),
        ('primary_color', np.int8, 0),
        # volatile
        ('x', np.float32, 0),
        ('y', np.float32, 0),
        ('ax', np.float32, 0), # direction
        ('ay', np.float32, 0),
        ('speed', np.float32, 0),
        ('age', np.uint32, 0),
        ('health', np.int32, 0),
        ('eating', np.bool, False),
        ('total_eaten', np.float32, 0),
        ('near_to', np.uint32, 0),
        ('attacking', bool, False),
        ('attacked_ok', bool, False),
    )

    _ind_stats_fields = (
        'color_r',
        'color_g',
        'color_b',
        'clockf1',
        'clockf2',
        'birth_health',
        'gencount',
        # calculated
        'herbivore',
        'primary_color',
        # volatile
        'speed',
        'age',
        'health',
        'eating',
        'total_eaten',
        'attacking',
        'attacked_ok',
    )

    def __init__(self, shape, *args, **kwargs):
        super(VectorAgent, self).__init__(shape, *args, **kwargs)
        self.env = None
        # TODO обойтись без копирования этого туда-сюда, а залинковать в мозг
        self.senses = VectorSenses(shape)
        self.actuators = VectorActuators(shape)
        self.chromosomes = VectorAgentChromosome(shape)
        self.brains = VectorBrain(shape,
            self.chromosomes.input_size,
            self.chromosomes.output_size,
            self.chromosomes.hidden_size)

    # def get_individual_stats(self):
    #     [self[:, f] for f in self._ind_stats_fields]

    def primary_color_histogram(a):
        return np.histogram(a, bins=range(9))[0]
    primary_color_histogram.return_len = 8
    primary_color_histogram.__name__ = 'histogram'

    _step_stats_fields = (
        ('birth_health', (np.min, np.max, np.mean)),
        ('gencount', (np.min, np.max, np.mean)),
        ('herbivore', (np.sum,)),
        ('primary_color', (primary_color_histogram,)),
        ('age', (np.mean, np.max, np.median)),
        ('health', (np.sum, np.mean, np.median)),
        ('eating', (np.sum,)),
        ('total_eaten', (np.sum, np.mean)),
        ('attacking', (np.sum,)),
        ('attacked_ok', (np.sum,)),
    )

    @classmethod
    def get_step_stats_field_names(cls):
        for field, aggregators in cls._step_stats_fields:
            for aggregator in aggregators:
                f_name = '%s_%s' % (field, aggregator.__name__)
                return_len = getattr(aggregator, 'return_len', None)
                if not return_len:
                    yield f_name
                else:
                    for i in range(return_len):
                        f_name = '%s_%d' % (f_name, i)
                        yield f_name

    def get_step_stats(self):
        stats = dict()
        for field, aggregators in self._step_stats_fields:
            values = getattr(self, field)
            for aggregator in aggregators:
                val = aggregator(values.astype(np.float32))
                f_name = '%s_%s' % (field, aggregator.__name__)
                if not hasattr(val, '__len__'):
                    stats[f_name] = val
                else:
                    for i, one_val in enumerate(val):
                        f_name_i = '%s_%d' % (f_name, i)
                        stats[f_name_i] = one_val
        return stats

    def can_give_birth(self):
        num_children_can = ((self.health - self.birth_health) / (self.birth_health * BIRTH_KPD))
        num_children_can = num_children_can.clip(0)
        # print num_children_can.max(), num_children_can.min(), num_children_can.mean()
        return num_children_can

    @classmethod
    def from_chromosomes(cls, chromosomes):
        babies = cls(chromosomes.shape)
        babies.develop_from_chromosomes(chromosomes)
        return babies

    def calc_primary_colors(self, sel=slice(None)):
        colors = ((self.color_r[sel] / 255.).round().astype(np.int8)
                + ((self.color_g[sel] / 255.).round().astype(np.int8) << 1)
                + ((self.color_b[sel] / 255.).round().astype(np.int8) << 2))
        self.primary_color[sel] = colors

    def develop_from_chromosomes(self, chromosomes, sel=slice(None)):
        # имеем вектор хромосом, надо из них получить
        # вектор развитых агентов
        if not isinstance(sel, slice) and len(sel)==0:
            return
        self.brains.develop_from_chromosomes(chromosomes, sel)
        self.color_r[sel] = chromosomes.color[:, 0].astype(np.uint8)
        self.color_g[sel] = chromosomes.color[:, 1].astype(np.uint8)
        self.color_b[sel] = chromosomes.color[:, 2].astype(np.uint8)
        self.calc_primary_colors(sel)
        self.ax[sel] = 1
        self.ay[sel] = 0
        self.x[sel] = 0
        self.y[sel] = 0
        self.age[sel] = 0
        self.attacked_ok[sel] = False
        self.total_eaten[sel] = 0
        self.rotate(np.random.random(chromosomes.shape) * (2*math.pi), sel)
        self.clockf1[sel] = chromosomes.clockf1
        self.clockf2[sel] = chromosomes.clockf2
        self.birth_health[sel] = chromosomes.birth_health.astype(np.int16)
        self.health[sel] = chromosomes.birth_health.astype(np.int16).copy()
        self.herbivore[sel] = (self.color_g[sel] / 255.).round().astype(bool)
        self.chromosomes[sel] = chromosomes

    def add_energy(self, energy, sel=slice(None)):
        energy_array = np.array(energy)
        self.health[sel] = (self.health[sel] + energy_array).clip(0, MAX_HEALTH).astype(np.int32)
        self.total_eaten[sel] += energy_array.clip(0)

    def consume_energy(self, ammount, sel=slice(None)):
        self.add_energy(-ammount, sel)

    def get_colors(self):
        return self.chromosomes.color.astype(np.uint8)

    def kill(self, sel=slice(None)):
        self.health[sel] = 0
        self.total_eaten[sel] = 0

    def is_dead(self):
        return self.health <= 0

    def eat_grass(self):
        amm = self.env.consume_food(self.eating)
        self.add_energy(amm, self.eating)

    def check_attack(self):
        all_nearest_i, all_in_radius = self.env.get_nearest_agent_in(slice(None), ATTACK_RADIUS_SQ)
        self.near_to = all_nearest_i
        self.consume_energy(ATTACK_ENERGY, self.attacking)
        nearest_i = all_nearest_i[self.attacking]
        in_radius = all_in_radius[self.attacking]
        if not np.count_nonzero(in_radius):
            return
        # remove not in radius
        attacking_i = np.where(self.attacking)[0][in_radius]
        nearest_i = nearest_i[in_radius]
        who_color = self[attacking_i].primary_color
        whom_color = self[nearest_i].primary_color
        # if whom_color beats who_color swap who and whom
        who_whom = np.hstack([attacking_i[:,None], nearest_i[:,None]])
        dist = (whom_color - who_color) % 7
        loosers = ~(dist % 2).astype(bool)
        # # color==7 always wins
        # loosers = loosers | (whom_color==7)
        # loosers = loosers & ~(who_color==7)
        # color==0 always loose
        loosers = loosers | (who_color==0)
        loosers = loosers & ~(whom_color==0)

        draw = (dist == 0)
        loosers = loosers & ~draw
        who_whom[loosers, :] = who_whom[loosers,::-1] # swap them
        who_whom = np.delete(who_whom, np.where(draw), axis=0)
        if who_whom.size == 0:
            return
        # remove duplicates after swapping
        _, uniq_i = np.unique(who_whom[:, 0], True)
        who_whom = who_whom[uniq_i]
        # и кого едят тоже должен быть один
        _, uniq_i = np.unique(who_whom[:, 1], True)
        who_whom = who_whom[uniq_i]
        self.attacked_ok[:] = False
        self.attacked_ok[who_whom[:, 0]] = True
        self.hunt(who_whom[:, 0], who_whom[:, 1])

    def hunt(self, predators, prays):
        # before = self.health[predators].copy()
        self.add_energy(self.health[prays] * HUNT_KPD, predators)
        # after = self.health[predators]
        # print (after - before).sum()
        self.kill(prays)

    def get_clock(self, num):
        # return vector with values n interval [0. .. 1.]
        freqs = getattr(self, "clockf%d" % num)
        return (self.age % freqs / freqs).astype(np.float32)

    def process_actuators(self):
        self.eating = self.actuators.eat.round().astype(np.bool)
        self.attacking = self.actuators.attack.round().astype(np.bool)

        max_speeds = MAX_SPEED + CARN_SPEED_HANDICAP * (~self.herbivore)
        self.speed = self.actuators.walk * max_speeds
        self.speed = self.speed.clip(0, max_speeds)
        rot_left = self.actuators.left.clip(-np.pi, np.pi)
        rot_right = self.actuators.right.clip(-np.pi, np.pi)
        self.rotate((rot_left - rot_right) * ROT_SPEED)

    def tick_brains(self):
        res = self.brains.tick(self.senses.get_values())
        self.actuators.set_values(res)

    def update(self):
        self.tick_brains()
        self.process_actuators()
#        confused = (self.speed.astype(bool) | self.attacking) & self.eating
#        # если одновременно есть и заниматься чем-то еще, то и не поешь и не позанимаешься
#        self.speed[confused] = 0
#        self.eating[confused] = False
        self.eating[~self.herbivore] = False
        self.eat_grass()
        self.check_attack()
        self.age += 1
        self.move()
        self.consume_energy(ENERGY_PER_TURN * self.speed)

    def rotate(self, angles, sel=slice(None)):
        cos, sin = np.cos(angles), np.sin(angles)
        ax, ay = self.ax[sel].copy(), self.ay[sel].copy()
        self.ax[sel] = (ax*cos - ay*sin).astype(np.float32)
        self.ay[sel] = (ax*sin + ay*cos).astype(np.float32)

    def move(self):
        self.x += self.ax * self.speed
        self.y += self.ay * self.speed
        self.wrap()

    def reverse(self, idxs):
        self.ax[idxs] *= -1
        self.ay[idxs] *= -1

    def wrap(self):
        self.x %= self.env.width
        self.y %= self.env.height

    def move_by(self, idxs, step):
        self.x[idxs] += self.ax[idxs] * step
        self.y[idxs] += self.ay[idxs] * step
        self.wrap()

    def reproduce(self, idxs, to_idxs, consume_health=True):
        unique_idxs = np.unique(idxs)
        to_idxs = to_idxs[:len(unique_idxs)]
        if len(unique_idxs) == 0:
            return 0
        parents = self[unique_idxs]
        new_chromosomes = VectorAgentChromosome(len(unique_idxs))
        new_chromosomes[:] = self.chromosomes[unique_idxs]
        new_chromosomes.mutate(to_idxs)
        self.develop_from_chromosomes(new_chromosomes, to_idxs)
        self.x[to_idxs] = parents.x
        self.y[to_idxs] = parents.y
        self.rotate(np.random.random(len(unique_idxs)) * (math.pi*2), to_idxs)
        self.x[to_idxs] += self.ax[to_idxs] * ATTACK_RADIUS
        self.y[to_idxs] += self.ay[to_idxs] * ATTACK_RADIUS
        self.gencount[to_idxs] = parents.gencount + 1
        self.health[to_idxs] = parents.birth_health
        if consume_health:
            self.consume_energy(parents.birth_health / BIRTH_KPD, unique_idxs)
        return len(unique_idxs)


