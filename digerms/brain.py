# coding: utf-8

import numpy as np

class VectorBrain(object):
    def __init__(self, shape, input_size, output_size, hidden_size):
        if isinstance(shape, int):
            shape = tuple([shape])
        brain_size = hidden_size + output_size
        overall_size = input_size + brain_size
        self.input_size = input_size
        self.output_size = output_size
        self._W = np.zeros(shape + (brain_size, overall_size), np.float32)
        self._biases = np.zeros(shape + (brain_size,), np.float32)
        ashape = self._W.shape
        ashape = ashape[:-2] + ashape[-1:]
        self._A = np.zeros(ashape)

    @classmethod
    def from_chromosomes(cls, chromosomes):
        brains = cls(chromosomes.shape, chromosomes.input_size, chromosomes.output_size, chromosomes.hidden_size)
        brains.develop_from_chromosomes(chromosomes)
        return brains

    def develop_from_chromosomes(self, chromosomes, sel=slice(None)):
        self._W[sel] = chromosomes.get_adj_matrix()
        self._biases[sel] = chromosomes.get_biases()
        self._A[sel] = 0

    def tick(self, input_values):
        self._A[..., :self.input_size] = input_values
        # 100x30 * 100x30x20 = 100x20
        # TODO поэлементное умножение и сумма требуют лишней памяти и обращений к ней
        # делаем поэлементное умножение
        prod = self._A[..., np.newaxis] * self._W.swapaxes(-2, -1)
        # и потом сумму
        Z = prod.sum(axis = -2)
        Z += self._biases
        new_A = 1 / (1 + np.exp(-Z))
        self._A[..., self.input_size:] = new_A
        outputs = self._A[..., -self.output_size:]
        return outputs
