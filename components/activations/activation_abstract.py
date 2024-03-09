import numpy as np

from abc import abstractmethod


class ActivationAbstract:
    Z: np.array
    dA: np.array

    def update_forward(self, Z: np.array):
        self.Z = Z

    def update_backward(self, dA: np.array):
        self.dA = dA

    @abstractmethod
    def forward(self) -> np.array:
        return self.Z

    @abstractmethod
    def backward(self) -> np.array:
        return self.dA
