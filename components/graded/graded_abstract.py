import numpy as np

from abc import abstractmethod


class GradedAbstract:
    A_prev: np.array
    W: np.array
    b: np.array

    dZ: np.array

    dW: np.array
    db: np.array

    def update_forward(self, A_prev: np.array, W: np.array, b: np.array):
        self.A_prev, self.W, self.b = A_prev, W, b

    def update_backward(self, dZ: np.array):
        self.dZ = dZ

    @abstractmethod
    def forward(self) -> np.array:
        print('')

    @abstractmethod
    def backward(self) -> np.array:
        print('')
