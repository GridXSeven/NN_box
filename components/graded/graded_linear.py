import numpy as np

from components.graded.graded_abstract import GradedAbstract


class GradedLinear(GradedAbstract):

    def forward(self):
        Z = np.dot(self.W, self.A_prev) + self.b
        return Z

    def backward(self):
        m = self.A_prev.shape[1]
        dW = (1 / m) * np.dot(self.dZ, self.A_prev.T)
        db = (1 / m) * np.sum(self.dZ, axis=1, keepdims=True)
        dA_prev = np.dot(self.W.T, self.dZ)
        return dA_prev, dW, db
