from Activation import _Sigmoid, _Soft_Max
from Loss import _CCE
from Network import *
from File import *
import numpy as np


# One-Hot Encoding
def encode(number_labels, target_label):
    out = [0] * number_labels
    out[int(target_label)] = 1
    return out


decision = input("Train and Test New Network [A] or Test Pre-Loaded [B] Network: ")

if decision == "A":
    train_data = load("mnist_train.csv", 100, True)
    test_data = load("mnist_test.csv", 100, True)

    network = Network([784, 16, 16, 10])
    network.set_activation(all=_Sigmoid, output=_Soft_Max)
    network.set_cost(_CCE)
    network.train(train_data, lambda i: np.divide(i[1:785], 255), lambda o: encode(10, o[0]), 10, 0.1)
    network.test(test_data, lambda i: np.divide(i[1:785], 255), lambda o: encode(10, o[0]), lambda i, o: np.array(i).argmax() == np.array(o).argmax())

elif decision == "B":
    test_data = load("mnist_test.csv", None, True)

    network = Network(None, "MNISTuner.txt")
    network.test(test_data, lambda i: np.divide(i[1:785], 255), lambda o: encode(10, o[0]), lambda i, o: np.array(i).argmax() == np.array(o).argmax())