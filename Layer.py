import numpy as np


class Layer:
    def __init__(self, number_nodes, number_out):
        self.number_nodes = number_nodes

        self.weights = 0.1 * np.random.randn(number_out, number_nodes)
        self.biases = np.zeros(number_nodes)
        self.deltas = np.zeros(number_nodes)

        self.input = None
        self.output = None
        self.activation = None

        self.previous_layer = None
        self.next_layer = None

    def forward(self, input):
        self.input = input
        self.output = self.activation.activate(np.dot(self.weights, input) + self.next_layer.biases)

        if self.next_layer is not None:
            self.next_layer.forward(self.output)

    def backward(self, learning_rate):
        if self.biases is not None:
            self.biases = self.biases - (self.deltas * learning_rate)

        if self.weights is not None:
            number_weights = self.weights.shape[0]
            for n in range(self.number_nodes):
                for w in range(number_weights):
                    self.weights[w][n] -= learning_rate * self.next_layer.deltas[w] * self.input[n]

        if self.previous_layer is not None:
            self.previous_layer.backward(learning_rate)


class Input_Layer(Layer):
    def __init__(self, number_nodes, number_out):
        super().__init__(number_nodes, number_out)
        self.biases = None

    def forward(self, input):
        self.input = input
        self.output = self.activation.activate(np.dot(self.weights, input) + self.next_layer.biases)

        if self.next_layer is not None:
            self.next_layer.forward(self.output)


class Output_Layer(Layer):
    def __init__(self, number_nodes, number_out):
        super().__init__(number_nodes, number_out)
        self.cost = None
        self.weights = None

    def forward(self, input):
        self.input = input
        self.output = input

    def loss(self, target_output):
        network_loss = self.cost.loss(self.output, target_output)
        print("Loss: ", network_loss)
        return network_loss

