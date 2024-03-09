import numpy as np

from components.initializaers.initializer_abstract import InitializerAbstract
from components.layers.layers_model import LayersModel


class InitializerRandom(InitializerAbstract):

    def initialize(self, weight_shape: np.shape, bias_shape: np.shape, layers: list[LayersModel]) -> (list[np.array],
                                                                                                      list[np.array]):
        W = []
        b = []
        prev_weight = weight_shape[0]
        for layer in layers:
            # TODO(OD): research and remove .astype if it`s possible
            W.append((np.random.randn(layer.neurons, prev_weight) * 0.01).astype(float))
            b.append((np.zeros((layer.neurons, 1))).astype(float))
            prev_weight = layer.neurons
        return W, b
