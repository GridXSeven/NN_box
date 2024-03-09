import numpy as np

from components.activations.activation_abstract import ActivationAbstract


class ActivationSigmoid(ActivationAbstract):

    @staticmethod
    def sigmoid(x):
        return 1/(1+(np.exp(-x)))

    def forward(self):
        return self.sigmoid(self.Z)

    def backward(self):
        Z = self.Z
        sig_Z = self.sigmoid(Z)
        dZ = self.dA * sig_Z * (1 - sig_Z)
        return dZ
