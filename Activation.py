import math
import numpy as np

E = math.e

_Sigmoid = "Sigmoid"
_ReLU = "ReLU"
_Soft_Max = "Soft_Max"


def for_each(array, callback):
    out = np.zeros(np.array(array).shape)

    if len(out.shape) == 1:
        for index in range(len(out)):
            out[index] = callback(array[index])
    else:
        for i in range(len(out)):
            for j in range(len(out[i])):
                out[i][j] = callback(array[i][j])

    return out


class Activation:
    def __init__(self):
        self.name = None

    def activate(self, output):
        pass

    def derive(self, output):
        pass


class Sigmoid(Activation):
    def __init__(self):
        super().__init__()
        self.name = _Sigmoid

    def activate(self, output):
        return for_each(output, lambda z: 1.0 / (1.0 + E**-z))

    def derive(self, output):
        return for_each(output, lambda a: a * (1 - a))


class ReLU(Activation):
    def __init__(self):
        super().__init__()
        self.name = _ReLU

    def activate(self, output):
        return for_each(output, lambda z: max(0, z))

    def derive(self, output):
        return for_each(output, lambda a: 1 if a > 0 else 0)


class Soft_Max(Activation):
    def __init__(self):
        super().__init__()
        self.name = _Soft_Max

    def activate(self, output):
        exp_output = np.exp(output)
        return exp_output / np.sum(exp_output)

    def derive(self, output):
        return for_each(output, lambda a: a * (1 - a))


activation = {
    _Sigmoid: Sigmoid(),
    _ReLU: ReLU(),
    _Soft_Max: Soft_Max()
}