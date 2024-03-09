import numpy as np

from abc import abstractmethod
from components.layers.layers_model import LayersModel


class InitializerAbstract:

    @abstractmethod
    def initialize(self, weight_shape: np.shape, bias_shape: np.shape, layers: list[LayersModel]) -> np.array:
        return np.zeros(weight_shape), np.zeros(bias_shape)
