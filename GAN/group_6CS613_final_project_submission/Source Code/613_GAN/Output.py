import numpy as np


class LeastSquares:
    def __init__(self, target):
        self.target = target
        self.dataIn = None

    def forwardPropagate(self, dataIn):
        self.dataIn = dataIn
        return dataIn

    def eval(self, estimated):
        observationCount = estimated.shape[0]  # TODO: Confirm this
        # return (self.target - estimated) ** 2  # single observation math
        return (self.target - estimated) * (self.target - estimated)/observationCount  # batch math

    def gradient(self, estimated):
        # return -2 * (self.target - estimated)  # single observation math
        return 2 * (estimated - self.target)  # batch math


class LogLoss:
    def __init__(self, target):
        self.target = target
        self.dataIn = None
        self.epsilon = 10 ** -7

    def forwardPropagate(self, dataIn):
        self.dataIn = dataIn
        return dataIn

    def eval(self, estimated):
        observationCount = estimated.shape[0]  # TODO: Confirm this
        # return -(self.target * np.log(estimated + self.epsilon) + (1 - self.target) * np.log(
        #     1 - estimated + self.epsilon))  # single observation math
        return -((self.target * np.log(estimated + self.epsilon)) + ((1 - self.target) * np.log(
            1 - estimated + self.epsilon)))/observationCount  # batch math

    def gradient(self, estimated):
        # return -(self.target - estimated) / (estimated * (1 - estimated) + self.epsilon)  # single observation math
        return np.divide((1 - self.target), (1 - estimated + self.epsilon)) - np.divide(
                self.target, (estimated + self.epsilon))  # batch math


class CrossEntropy:
    def __init__(self, target):
        self.target = target
        self.dataIn = None
        self.epsilon = 10 ** -7

    def forwardPropagate(self, dataIn):
        self.dataIn = dataIn
        return dataIn

    def eval(self, estimated):
        observationCount = estimated.shape[0]  # TODO: Confirm this
        # return np.sum(-self.target * np.log(np.transpose(estimated) + self.epsilon))  # single observation math
        return -self.target * np.log(estimated + self.epsilon)/observationCount  # batch math

    def gradient(self, estimated):
        # return -self.target / (estimated + self.epsilon)  # single observation math
        return np.divide(-self.target, (estimated + self.epsilon))  # batch math (same as single)


class Generator:
    def __init__(self):
        self.dataIn = None
        self.epsilon = 10 ** -7

    def forwardPropagate(self, dataIn):
        self.dataIn = dataIn
        return dataIn

    def eval(self, estimated):
        observationCount = estimated.shape[0]  # TODO: Confirm this
        return -np.log(estimated + self.epsilon)

    def gradient(self, estimated):
        return -1/(estimated + self.epsilon)
