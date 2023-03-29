import numpy as np


class ReLu:
    def __init__(self):
        self.dataIn = None

    def forwardPropagate(self, dataIn):
        self.dataIn = dataIn
        output = np.maximum(dataIn, 0)
        return output

    def gradient(self):
        output = np.where(self.dataIn > 0, 1, 0)
        return output

    def backwardPropagate(self, gradIn):
        gradOut = np.multiply(gradIn, self.gradient())# gradIn * self.gradient()
        return gradOut

class ReLuTest:
    def __init__(self):
        self.dataIn = None

    def forwardPropagate(self, dataIn):
        self.dataIn = dataIn
        output = np.where(self.dataIn > 0, 3, 0)
        return output

    def gradient(self):
        output = np.where(self.dataIn > 0, 1, 0)
        return output

    def backwardPropagate(self, gradIn):
        gradOut = np.multiply(gradIn, self.gradient())# gradIn * self.gradient()
        return gradOut


class Sigmoid:
    def __init__(self):
        self.dataIn = None
        self.__dataOut = None

    def forwardPropagate(self, dataIn):
        self.dataIn = dataIn
        output = np.array(1 / (1 + np.exp(-dataIn)))

        self.dataIn = dataIn
        self.__dataOut = output

        return output

    def gradient(self):
        arr1 = self.__dataOut  # arr1 = self.forwardPropagate(self.dataIn)
        arr2 = np.array(1 - arr1)
        output = np.multiply(arr1, arr2)
        return output

    def backwardPropagate(self, gradIn):
        gradOut = gradIn * self.gradient()
        return gradOut


class Softmax:
    def __init__(self):
        self.dataIn = None
        self.__dataOut = None

    def forwardPropagate(self, dataIn):
        if dataIn.ndim == 1:
            output = np.exp(dataIn - np.max(dataIn))
            output = output / np.sum(output)
        else:
            output = np.exp(dataIn - np.max(dataIn, axis=1, keepdims=True))
            output = output / np.sum(output, axis=1, keepdims=True)

        self.dataIn = dataIn
        self.__dataOut = output

        return output

    def gradient(self):
        arr1 = self.__dataOut  # arr1 = self.forwardPropagate(self.dataIn)
        arr2 = np.array(1 - arr1)
        output = np.multiply(arr1, arr2)
        return output

    def backwardPropagate(self, gradIn):
        gradOut = gradIn * self.gradient()
        return gradOut


class HyperbolicTangent:
    def __init__(self):
        self.dataIn = None
        self.__dataOut = None

    def forwardPropagate(self, dataIn):
        if dataIn.ndim == 1:
            output = np.exp(dataIn - np.max(dataIn))
            output = output / np.sum(output)
        else:
            # print(np.max(dataIn, axis=1, keepdims=True)[1])
            # print(dataIn[1])
            num = np.exp(dataIn) - \
                  np.exp(-(dataIn))
            den = np.exp(dataIn) + \
                  np.exp(-(dataIn))
            output = np.divide(num, den)
            # print("\nmine")
            # print(output)
            # print("\ntheirs")
            # output = np.tanh(dataIn)
            # print(output)

        self.dataIn = dataIn
        self.__dataOut = output

        return output

    def gradient(self):
        arr1 = self.__dataOut
        output = np.array(1 - np.square(arr1))
        return output

    def backwardPropagate(self, gradIn):
        gradOut = gradIn * self.gradient()
        return gradOut