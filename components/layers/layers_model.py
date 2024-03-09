import numpy as np

from components.activations.activation_abstract import ActivationAbstract
from components.graded.graded_abstract import GradedAbstract


class LayersModel:
    activation: ActivationAbstract
    graded: GradedAbstract
    neurons: int

    dA: np.array
    dW: np.array
    db: np.array

    def __init__(self, activation: ActivationAbstract, graded: GradedAbstract, neurons: int):
        self.activation, self.graded, self.neurons = activation, graded, neurons

    def forward_activation_propagation(self, A_prev: np.array, W: np.array, b: np.array) -> object:
        self.graded.update_forward(A_prev=A_prev, W=W, b=b)
        Z = self.graded.forward()
        self.activation.update_forward(Z=Z)
        A = self.activation.forward()
        return A

    def backward_activation_propagation(self, dA: np.array) -> object:
        self.activation.update_backward(dA=dA)
        dZ = self.activation.backward()
        self.graded.update_backward(dZ=dZ)
        dA_prev, dW, db = self.graded.backward()
        self.dA, self.dW, self.db = dA, dW, db
        return dA_prev, dW, db
