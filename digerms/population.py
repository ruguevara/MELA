# coding: utf-8
import random
import numpy as np
from agent import VectorAgent
from genome import VectorAgentChromosome
from settings import SELECT_RATIO, RANDOM_RATIO, MIN_HEALTH

__author__ = 'rugu'


class Population(object):
    def __init__(self, count):
        self._count = count
        self._agents = None
        self.env = None
        self.stat_logger = None
        self.total_born = 0
        self.total_random = 0
        self.total_deaths = 0

    def __iter__(self):
        return iter(self._agents)

    def __len__(self):
        return len(self._agents)

    def __getitem__(self, item):
        return self._agents[item]

    @classmethod
    def random(cls, number):
        population = cls(number)
        cromosomes = VectorAgentChromosome.random(number)
        agents = VectorAgent.from_chromosomes(cromosomes)
        population._agents = agents
        return population

    def set_env(self, env):
        self.stat_logger = env.stats
        self._agents.env = env
        self.env = env

    @property
    def senses(self):
        return self._agents.senses

    def get_colors(self):
        return self._agents.get_colors()

    def get_health(self):
        return self._agents.health

    def get_angles(self):
        return self._agents.ax, self._agents.ay

    def set_X(self, X, sel=slice(None)):
        self._agents.x[sel] = X[:, 0].astype(np.float32)
        self._agents.y[sel] = X[:, 1].astype(np.float32)

    def get_X(self):
        l = self._count
        x, y = self._agents.x, self._agents.y
        # FIXME избавиться от копирования, можно манипулировать смещением массива в том же буфере
        return np.hstack([x.reshape(l, 1), y.reshape(l,1)])

    def get_clock(self, i):
        return self._agents.get_clock(i)

    def fitness(self):
        """
        массив с вычисленной фитнесс-функцией для всей популяции
        """
#        варианты:
#        * health
#        * total_eaten
#        * age
#        * комбинация age и total_eaten
        age = self._agents.age + 1
        eaten = self._agents.total_eaten
        # fitness = eaten / age
        fitness = eaten / age + self._agents.health / 3
        self._fitness = fitness
        return fitness

    def select(self, fitness, select_ratio=SELECT_RATIO):
        """
        отбор
        select_ratio -- сколько лучших оставлять
        returns массив индексов выбранных и не выбранных
        """
        # TODO выбирать для отстваления в живых по одному критерию,
        # а ранжировать по другому.
        dead = self._agents.is_dead() # dead already
        age = self._agents.age + 1
        to_live_rating = fitness
        #to_live_rating = np.where(age > 500, fitness, 1000000 / age)
        #print to_live_rating
        to_live_rating[dead] = -1
        liveness_rank = np.argsort(to_live_rating)
        sel_num = int(round(self._count * select_ratio))
        # exclude dead from selection
        sel_num = min(np.count_nonzero(~dead), sel_num)
        # хотя бы одного всегда убивать
        sel_num = min(len(self._agents)-1, sel_num)
        selected_idxs = liveness_rank[-sel_num:]
        dead_idxs = liveness_rank[:-sel_num]
        fitness_rank = selected_idxs[np.argsort(fitness[selected_idxs])[::-1]]
        return fitness_rank, dead_idxs

    def can_give_birth(self):
        return self._agents.can_give_birth()

    def make_new_random(self, to_random_num, idxs):
        assert to_random_num >= 0
        new_chromosomes = VectorAgentChromosome.random(to_random_num)
        self.replace_from_chromosomes(idxs, new_chromosomes)
        self._agents.health[idxs] = MIN_HEALTH
        self.env.add_new_agents(idxs)

    def select_and_reproduce(self):
        # естественный отбор и репродукция
        # select_ratio -- сколько лучших оставлять
        # random_ratio -- сколько еще добавить случайных агентов
        select_ratio=SELECT_RATIO
        random_ratio=RANDOM_RATIO
        select_ratio=min(select_ratio, 1-random_ratio)
        fitness = self.fitness()
        selected_idxs, dead_idxs = self.select(fitness, select_ratio)
        self.total_deaths = len(dead_idxs)

        assert len(selected_idxs) + len(dead_idxs) == self._count,\
            "%d + %d!= %d" % (len(selected_idxs), len(dead_idxs), self._count)

        # сколько случайных вставлять
        to_random_num = random_ratio * self._count
        if to_random_num < 1:
            to_random_num = 1 if random.random() < to_random_num else 0
        # сколько потомков получить при репродукции
        # надо учесть что не все могут быть готовы к репродукции
        # тогда пусть рожают больше положенного
        to_born_num = max(0, self.total_deaths - to_random_num)
        self.total_born = self.reproduce(selected_idxs, dead_idxs, fitness, to_born_num)
        self.total_random = self.total_deaths - self.total_born
        self.make_new_random(self.total_random, dead_idxs[self.total_born:])

    def reproduce(self, sel_idxs, to_idxs, fitness, to_born_num):
        """
        reproduce @num_to_reproduce individs from <sel_idxs> selection
        """
        ready_to_born = self.can_give_birth()
        self.ready_to_born = np.count_nonzero(ready_to_born)
        # каждый индивид готовый размножаться размножается пропорционально фитнесу
        # TODO crossover
        only_ready = fitness[sel_idxs] * ready_to_born[sel_idxs].astype(bool)
        reranked_idxs = np.argsort(only_ready)[::-1]
        combined_sorted = only_ready[reranked_idxs]
        roulette = combined_sorted.cumsum()
        if roulette[-1] <= 1:
            # некому рожать
            return 0
        #num_to_reproduce = min(num_to_reproduce, int(ready_to_born.count()))
        selectors = np.random.randint(0, roulette[-1], to_born_num)
        idxs = np.searchsorted(roulette, selectors)
        idxs = sel_idxs[reranked_idxs[idxs]]
        num_children = self._agents.reproduce(idxs, to_idxs[:len(idxs)])
        while num_children < to_born_num:
            # обычных детей не хватает чтобы восполнить убыль
            forsed_parents = idxs[:to_born_num - num_children]
            num_new = len(forsed_parents)
            num_new = self._agents.reproduce(forsed_parents, to_idxs[num_children:num_children+num_new], consume_health=False)

            num_children += num_new
        return num_children

    def replace_from_chromosomes(self, idxs, chromosomes):
        assert len(idxs) == len(chromosomes), "%d != %d" % (len(idxs), len(chromosomes))
        self._agents.develop_from_chromosomes(chromosomes, idxs)
        # разбросать новых вокруг?
        self.env.add_new_agents(idxs)

    def update(self):
        self._agents.update()
        self.log_stats()

    def log_stats(self):
        if self.stat_logger is not None:
            stats = self._agents.get_step_stats()
            stats.update(
                born=self.total_born,
                random=self.total_random,
                deaths=self.total_deaths,
                fitness_max=self._fitness.max(),
                fitness_mean=self._fitness.mean(),
                step=self.env.time,
                ready_to_born=self.ready_to_born,
            )
            self.stat_logger.log(**stats)
