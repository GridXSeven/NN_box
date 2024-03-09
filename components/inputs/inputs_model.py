import numpy as np


class InputsModel:
    weights: np.array
    bias: np.array

    def __init__(self, weights: np.array, bias: np.array, ):
        self.weights, self.bias = weights, bias
