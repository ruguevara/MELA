# coding: utf-8
from random import randint, sample, random, gauss, uniform
import numpy as np
import sys
from prettytable import PrettyTable
from environment import SENSES

INPUTSIZE = len(SENSES)

ACTUATORS = ['eat', 'walk', 'left', 'right', 'attack']
OUTPUTSIZE = len(ACTUATORS)
#HIDDENSIZE = 0
HIDDENSIZE = OUTPUTSIZE
#HIDDENSIZE = OUTPUTSIZE + INPUTSIZE

CONNS = (INPUTSIZE + HIDDENSIZE + OUTPUTSIZE) / 2

class ObjectArray(np.recarray):
    def __new__(subtype, shape, *args, **kwargs):
        fields = subtype._fields
        dtype = [(n,t) for n, t, d in fields]
        return np.recarray.__new__(subtype, shape, dtype, *args, **kwargs)

    def __init__(self, shape, *args, **kwargs):
        np.recarray.__init__(self, shape, *args, **kwargs)
        self[:] = tuple(d for n, t, d in self._fields)

class ZeroObjectArray(np.recarray):
    def __new__(subtype, shape, *args, **kwargs):
        fields = subtype._fields
        dtype = [(n,t) for n, t in fields]
        return np.recarray.__new__(subtype, shape, dtype, *args, **kwargs)

    def __init__(self, shape, *args, **kwargs):
        np.recarray.__init__(self, shape, *args, **kwargs)
        self[:] = 0


class FloatObjectArray(np.recarray):
    def __new__(subtype, shape, *args, **kwargs):
        fields = subtype._fields
        dtype = [(n,np.float32) for n in fields]
        return np.recarray.__new__(subtype, shape, dtype, *args, **kwargs)

    def __init__(self, shape, *args, **kwargs):
        np.recarray.__init__(self, shape, *args, **kwargs)

    def as_array(self):
        all_shape = self.shape + (len(self._fields),)
        return np.ndarray(shape=all_shape, buffer=self,
            dtype = np.float32)

    def get_values(self):
        return self.as_array()

    def set_values(self, values):
        view = self.as_array()
        view[:] = values

    def tabular_dump(self, sel=slice(None)):
        table = PrettyTable(self._fields)
        for row in self[sel]:
            table.add_row(row)
        return table.get_string()



class VectorChromosomeMetaclass(type):
    def __new__(cls, cls_name, bases, dct):
        genes = dct.get('GENES', None)
        assert genes is not None, "Provide GENES class attribute in %s" % cls_name
        _GENE_MAP = dict()
        genome_len = sum(vars for name, vars, _min, _max in genes)
        _ranges = np.ndarray((genome_len, 2), np.float32)
        _stds = np.ndarray(genome_len, np.float32)
        _means = np.ndarray(genome_len, np.float32)
        i = 0
        for name, vars, _min, _max in genes:
            assert vars > 0
            _GENE_MAP[name] = (i, i+vars)
            _ranges[i:i+vars] = (_min, _max)
            _stds[i:i+vars] = float(_max - _min) / 2
            _means[i:i+vars] = float(_min + _max) / 2
            i += vars

        dct.update(
            genome_len=genome_len,
            _GENE_MAP=_GENE_MAP,
            _ranges=_ranges,
            _stds=_stds,
            _means=_means,
        )
        return super(VectorChromosomeMetaclass, cls).__new__(cls, cls_name, bases, dct)

class VectorChromosome(object):
    """
    двумерный массив float-ов, по строкам экземпляры из популяции,
    а в колонках гены
    гены и наборы генов имеют названия
    """
    __metaclass__ = VectorChromosomeMetaclass
    # TODO вместо числа переменных на именованный параметр
    # можно закодировать размерность этого параметра, чтобы потом не решейпить каждый раз
    GENES = (
        # name           vars  range
        ('mut_rate', 1,   0.01, 1),
        ('mut_std',  1,   0.001, 1), # when mutation occurs,
        # std is scaled to range of mutated gene
        )

    def __init__(self, shape):
        if isinstance(shape, int):
            shape = tuple([shape])
        shape = shape + tuple([self.genome_len])
        self._genes = np.ndarray(shape, np.float32)

    @classmethod
    def random(cls, shape):
        c = cls(shape)
        c.randomize()
        return c

    def __len__(self):
        return self._genes.shape[0]

    def copy(self):
        other = type(self)(self.shape)
        other._genes = self._genes.copy()
        return other

    @property
    def shape(self):
        return self._genes.shape[:-1]

    def __setitem__(self, key, value):
        self._genes[key] = value

    def __getitem__(self, key):
        return self._genes[key]

    def __getattr__(self, item):
        return self.__get_parameter(item)

    def __get_parameter(self, item):
        start, stop = self._GENE_MAP.get(item, (None, None))
        if start is not None:
            out = self._genes[:, start:stop]
            if out.shape[-1] == 1:
                # reduction
                out = out.reshape(out.shape[:-1])
            return out
        return super(VectorChromosome, self).__getattribute__(item)

    def __setattr__(self, key, value):
        start, stop = self._GENE_MAP.get(key, (None, None))
        if start is not None:
            out = self.__getattr__(key)
            out[:] = value
        else:
            super(VectorChromosome, self).__setattr__(key, value)

    def randomize(self, mask = None):
        starts = self._ranges[:, 0]
        ranges_lens = self._ranges[:, 1] - starts
        self._genes[:] = starts + np.random.random(self._genes.shape) * ranges_lens

    def mutate(self, mask = None):
        # если мутаций мало, наверное нужно гены по одному мутировать в цикле
        # иначе будем оперировать большими объемами с немутировавшими генами зазря
        mutators = np.random.random((self.shape[0], self.genome_len))
        mut_rate = self.__get_parameter("mut_rate")
        mut_std = self.__get_parameter("mut_std")
        mutated = mut_rate[..., np.newaxis] >= mutators
        # получить индексы мутирующих генов
        n = len(np.flatnonzero(mutated))
        pop_i, gene_i = np.where(mutated)
        total_var = self._stds[gene_i] * mut_std[pop_i]
        self._genes[mutated] += np.random.normal(0, total_var, n)
        self._genes[mutated] = self._genes[mutated].clip(
            self._ranges[gene_i,0], self._ranges[gene_i,1])


class VectorBrainChromosome(VectorChromosome):
    input_size = INPUTSIZE
    hidden_size = HIDDENSIZE
    output_size = OUTPUTSIZE
    brain_size = hidden_size + output_size
    overall_size = input_size + brain_size
    num_connections = CONNS

    GENES = VectorChromosome.GENES + (
        # name           vars  range
        ('clockf1',     1, 2, 100),
        ('clockf2',     1, 2, 100),
        ('connections', brain_size * num_connections ,    0, overall_size), # от каких узлов связи
        ('weights',     brain_size * (num_connections+1), -100, 100),    # веса этих связей + bias
        )

    def get_adj_matrix(self):
        """
        декодирование генома в матрицу смежности-весов + биас (0 строка)
        рамернойсть каждой матрицы для индивидов 20 (невходных клеток) на 30 (всех клеток)
        а в хромосоме для каждой невходной клетки закодированы от 0 до <num_connections> индексов связей с другими клетками
        если индекс==<num_connections>, то нет связи
        """
        shape = self.shape
        conns = self.connections.reshape(shape + (self.brain_size, self.num_connections)).astype(int)
        new_shape = shape + (self.brain_size, self.num_connections+1)
        weights = self.weights.reshape(new_shape)
        adjacency = np.zeros(shape + (self.brain_size, self.overall_size+1), np.float32 )
        i_popul, i_to, i_from = np.indices(conns.shape)
        # TODO ravel idxs?
        # Вот в следующей строке вся магия
        adjacency[i_popul, i_to, conns] = weights[i_popul, i_to, i_from+1]
        # zero connections from indxs of nonexistent cells (idx==num_connections)
        adjacency = np.delete(adjacency, -1, -1)
        return adjacency

    def get_biases(self):
        new_shape = self.shape + (self.brain_size, self.num_connections+1)
        weights = self.weights.reshape(new_shape)
        return weights[..., :, 0]

    @classmethod
    def get_inputs_outputs(cls):
        input_size = cls.input_size
        output_size = cls.output_size
        overall_size = cls.overall_size
        inputs = range(input_size)
        outputs = range(overall_size - output_size, overall_size)
        return inputs, outputs


class VectorAgentChromosome(VectorBrainChromosome):
    GENES = VectorBrainChromosome.GENES + (
        # name           vars  range
        ('birth_health', 1,  100,   10000),
        # ('herbivore',    1,  0,   1),
        ('color',        3,  10,  255),
        )
