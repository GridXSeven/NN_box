import numpy as np

from components.initializaers.initializer_abstract import InitializerAbstract
from components.inputs.inputs_model import InputsModel
from components.layers.layers_model import LayersModel


class NNModel:
    inputs: InputsModel
    layers: list[LayersModel]
    initializer: InitializerAbstract

    learning_rate: float

    num_iterations: int
    W: np.array
    b: np.array
    AL: np.array

    costs: list[np.ndarray]
    cost: np.ndarray

    def __init__(self, inputs: InputsModel, layers: list[LayersModel], initializer: InitializerAbstract,
                 num_iterations: int, learning_rate: float):
        self.inputs, self.layers, self.initializer, self.num_iterations, self.learning_rate = inputs, layers, \
                                                                                              initializer, \
                                                                                              num_iterations, \
                                                                                              learning_rate
        self.costs = []

    def learn(self):
        self.W, self.b = self.initializer.initialize(bias_shape=self.inputs.bias.shape,
                                                     weight_shape=self.inputs.weights.shape,
                                                     layers=self.layers)

        for i in range(self.num_iterations):
            self.l_forward(self.inputs.weights)
            self.compute_cost()
            self.l_backward()
            self.update_parameters()
            if i % 1000 == 0:
                print("Cost after iteration %i: %f" % (i, self.cost))
        # plt.plot(self.costs)
        # plt.ylabel('cost')
        # plt.xlabel('iterations (per hundreds)')
        # plt.title("Learning rate =" + str(self.learning_rate))
        # plt.show()

    def l_forward(self, dataSet: np.array):
        A = dataSet
        for layer_number in range(len(self.layers)):
            A_prev = A
            layer = self.layers[layer_number]
            A = layer.forward_activation_propagation(A_prev=A_prev, W=self.W[layer_number], b=self.b[layer_number])
        self.AL = A.astype(float)

    def l_backward(self):
        # TODO(OD): research and remove .astype if it`s possible
        bias = self.inputs.bias.astype(float)
        AL = self.AL.astype(float)

        dAL = - (np.divide(bias, AL) - np.divide(1 - bias, 1 - AL))
        dA = dAL
        for layer in reversed(range(len(self.layers))):
            layer = self.layers[layer]
            dA, dW, db = layer.backward_activation_propagation(dA=dA)

    def compute_cost(self):
        # TODO(OD): research and remove .astype if it`s possible
        Y = self.inputs.bias.astype(float)
        AL = self.AL.astype(float)
        m = Y.shape[1]
        cost = (-1 / m) * np.sum((Y * np.log(AL)) + (1 - Y) * (np.log(1 - AL)))
        cost = np.squeeze(cost)

        self.cost = cost
        self.costs.append(cost)

    def update_parameters(self):
        for layer in range(len(self.layers)):
            self.W[layer] = self.W[layer] - (self.learning_rate * self.layers[layer].dW)
            self.b[layer] = self.b[layer] - (self.learning_rate * self.layers[layer].db)

    def predict(self, test_set: np.array):
        self.l_forward(test_set)
        predictions = self.AL > 0.5
        return predictions
