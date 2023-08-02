import numpy as np

_MSE = "MSE"
_CCE = "Categorical_Cross_Entropy"


class Loss:
    def __init__(self):
        self.name = None

    def loss(self, output, target):
        pass

    def derive(self, output, target):
        pass


class MSE(Loss):
    def __init__(self):
        super().__init__()
        self.name = _MSE

    def loss(self, output, target):
        return np.mean(np.square(np.subtract(output, target)))

    def derive(self, output, target):
        return 2 / len(output) * np.subtract(output, target)


class Categorical_Cross_Entropy(Loss):
    def __init__(self):
        super().__init__()
        self.name = _CCE

    def loss(self, output, target):
        output_clipped = np.clip(output, 1e-7, 1 - 1e-7)
        confidence = np.sum(output_clipped * target)
        return -np.log(confidence)

    def derive(self, output, target):
        output_clipped = np.clip(output, 1e-7, 1 - 1e-7)
        LH = -1 * np.divide(target, output_clipped)
        NUM = np.add(np.multiply(-1, target), 1)
        DEN = np.add(np.multiply(-1, output_clipped), 1)
        return np.add(LH, NUM / DEN)


loss = {
    _MSE: MSE(),
    _CCE: Categorical_Cross_Entropy()
}
