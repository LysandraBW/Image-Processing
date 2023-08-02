from Layer import *
from File import *
from Activation import *
from Loss import *


class Network:
    def __init__(self, breakdown=None, in_file=None):
        if breakdown is None and in_file is None:
            return

        self.layers = None
        self.breakdown = None
        self.number_layers = None

        if breakdown is not None:
            self.init_network(breakdown)
        elif in_file is not None:
            self.load_network(in_file)

    def save_network(self, out_file):
        with open(out_file, 'w') as file:
            file.write(arr_to_string(self.breakdown) + "\n")
            for layer_index in range(len(self.layers)):
                file.write(arr_to_string(self.layers[layer_index].weights) + "\n" if self.layers[layer_index].weights is not None else "None\n")
                file.write(arr_to_string(self.layers[layer_index].biases) + "\n" if self.layers[layer_index].biases is not None else "None\n")
                file.write(self.layers[layer_index].activation.name+"\n" if self.layers[layer_index].activation.name is not None else "None\n")
                if layer_index == len(self.layers) - 1:
                    file.write(self.layers[layer_index].cost.name+"\n" if self.layers[layer_index].cost.name is not None else "None\n")

    def load_network(self, in_file):
        with open(in_file, 'r') as file:
            lines = file.readlines()
            number_lines = len(lines)

            # The breakdown is stored in the first line
            breakdown = string_to_arr(lines[0][0:-1], False, True)

            self.init_network(breakdown)

            layer_index = 0
            for index in range(1, number_lines - 3, 3):
                # The string contains a newline character at the end, so I'm removing that
                weight = lines[index][0:-1]
                bias = lines[index + 1][0:-1]
                act = lines[index + 2][0:-1]

                # Getting Weights, Biases, and Activation Function
                self.layers[layer_index].weights = None if weight == "None" else np.array(string_to_arr(weight, True))
                self.layers[layer_index].biases = None if bias == "None" else np.array(string_to_arr(bias, True))
                self.layers[layer_index].activation = None if act == "None" else activation[act]

                if layer_index == len(breakdown) - 1:
                    cost = lines[index + 3][0:-1]
                    self.layers[layer_index].cost = None if act == "None" else loss[cost]

                layer_index += 1

    def init_network(self, breakdown):
        self.layers = []
        self.breakdown = breakdown
        self.number_layers = len(breakdown)

        for l in range(0, self.number_layers):
            if l == 0:
                self.layers.append(Input_Layer(breakdown[l], breakdown[l + 1]))
            elif l == self.number_layers - 1:
                self.layers.append(Output_Layer(breakdown[l], 0))
            else:
                self.layers.append(Layer(breakdown[l], breakdown[l + 1]))

        for l in range(0, self.number_layers):
            self.layers[l].next_layer = None if l == self.number_layers - 1 else self.layers[l + 1]
            self.layers[l].previous_layer = None if l == 0 else self.layers[l - 1]

    def set_activation(self, **kwargs):
        if "all" in kwargs:
            for layer in self.layers:
                layer.activation = activation[kwargs["all"]]

        if "output" in kwargs:
            self.layers[-2].activation = activation[kwargs["output"]]
            # The output layer doesn't actually require this, but it helps me a lot with readability. The -1 and -2 makes a difference
            # (for me) when trying to reason with myself in my head.
            self.layers[-1].activation = activation[kwargs["output"]]

    def set_cost(self, cost):
        self.layers[-1].cost = loss[cost]

    def train(self, train, extract_in, extract_out, epochs, learning_rate):
        epoch = 0

        while epoch < epochs:
            for e in range(len(train)):
                # Extracting the Input and Target Data
                input = extract_in(train[e])
                output = extract_out(train[e])

                # Forward Pass
                self.layers[0].forward(input)

                # Loss
                self.layers[-1].loss(output)

                # Calculating Delta(s) of Output
                network_output = self.layers[-1].output
                loss_derivative = self.layers[-1].cost.derive(network_output, output)
                activation_derivative = self.layers[-1].activation.derive(network_output)
                self.layers[-1].deltas = np.multiply(loss_derivative, activation_derivative)

                # Calculating Delta(s) of Remaining
                for l in range(self.number_layers - 2, -1, -1):
                    layer = self.layers[l]
                    layer.deltas = np.sum(np.multiply(layer.weights.T, layer.next_layer.deltas),
                                          axis=1) * layer.activation.derive(layer.input)

                # Backward Pass
                self.layers[-1].backward(learning_rate)

            epoch += 1

    def test(self, test, extract_in, extract_out, accurate):
        number_wrong = 0
        number_tests = len(test)

        print("\nTesting...\n")
        for t in range(number_tests):
            inp = extract_in(test[t])
            out = extract_out(test[t])
            self.layers[0].forward(inp)

            print(f"Predicted Output: {self.layers[-1].output}\nExpected Output: {out}\n")
            if not accurate(self.layers[-1].output, out):
                number_wrong += 1

        print(f"Score: {((number_tests - number_wrong) / number_tests) * 100}%\n\tNumber of Samples: {number_tests}\n\tNumber Correct: {number_tests - number_wrong}\n\tNumber Wrong: {number_wrong}\n")

        save = input("Save Network\n[Y] [N]\n")
        if save == "Y" or save == "y":
            file_name = input("Enter File Name: ")
            file_name = "Tuner.txt" if file_name == "" else file_name
            self.save_network(file_name)

    def get_output(self, input):
        self.layers[0].forward(input)
        return self.layers[-1].output