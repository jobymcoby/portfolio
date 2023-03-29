import math

import numpy as np


class FullyConnected:
    def __init__(self, sizein, sizeout, learningRate):
        max= math.sqrt(6) / math.sqrt(sizein+sizeout)
        min = -max
        self.__weights = 0.0001 * (np.random.rand(sizein, sizeout) - 0.5)
        self.__biases = 0.0001 * (np.random.rand(1, sizeout) - 0.5)
        # #scale- XAVIER
        # self.__weights = self.__weights + min * (max - min)
        # self.__biases = self.__biases + min * (max - min)

        self.dataIn = None

        #ADAM vars
        self.delta = 10**-8
        self.learningRate = learningRate #.1#10

        #Adaptive Learning Rate - Weights
        self.rho1_w = 0.9
        self.rho2_w = 0.999
        self.r_w = 0  # accumulator
        self.s_w = 0  # momentum

        # Adaptive Learning Rate - Biases
        self.rho1_b = 0.9
        self.rho2_b = 0.999
        self.r_b = 0  # accumulator
        self.s_b = 0  # momentum


    def forwardPropagate(self, dataIn):
        self.dataIn = dataIn
        return (dataIn @ self.__weights) + self.__biases

    def backwardPropagate(self, gradIn, epoch):
        observationCount = gradIn.shape[0]  # TODO: Confirm this

        #Cache gradient before updating weights
        gradOut = gradIn @ self.gradient()

        # # update weights
        # dW = np.transpose(self.dataIn)
        # dJdW = dW @ gradIn
        #
        # # Adaptive Learning Rate
        # self.s_w = (self.rho1_w * self.s_w) + ((1 - self.rho1_w) * dJdW)
        # self.r_w = (self.rho2_w * self.r_w) + ((1 - self.rho2_w) * (dJdW * dJdW))
        # _num = self.s_w / (1 - pow(self.rho1_w, epoch))
        # _denom = (np.sqrt(self.r_w / (1 - pow(self.rho2_w, epoch)))) + self.delta
        # Adam = _num / _denom
        # self.__weights = self.__weights + self.learningRate * (-Adam) / observationCount
        #
        # # update biases
        # db = np.transpose(np.ones((observationCount, 1)))
        # dJdb = db @ gradIn
        #
        # # Adaptive Learning Rate
        # self.s_b = (self.rho1_b * self.s_b) + ((1 - self.rho1_b) * dJdb)
        # self.r_b = (self.rho2_b * self.r_b) + ((1 - self.rho2_b) * (dJdb * dJdb))
        # _num = self.s_b / (1 - pow(self.rho1_b, epoch))
        # _denom = (np.sqrt(self.r_b / (1 - pow(self.rho2_b, epoch)))) + self.delta
        # Adam = _num / _denom
        # self.__biases = self.__biases + self.learningRate * (-Adam) / observationCount

        # update weights
        dW = np.transpose(self.dataIn)
        dJdW = dW @ gradIn

        self.__weights = self.__weights + self.learningRate/observationCount*(-dJdW)

        # update biases
        db = np.transpose(np.ones((observationCount, 1)))
        dJdb = db @ gradIn
        self.__biases = self.__biases + self.learningRate/observationCount*(-dJdb)

        return gradOut

    def backwardPropagateNoUpdate(self, gradIn):
        gradOut = gradIn @ self.gradient()
        return gradOut

    def gradient(self):
        # print(self.__weights.shape[0])
        # print(self.__weights.shape[1])
        dh = np.transpose(self.__weights)
        # print(dh.shape[0])
        # print(dh.shape[1])
        return dh

    def printParameters(self):
        print('\nEnding Weights')
        print(self.__weights)
        print('\nEnding Bias')
        print(self.__biases)
        return 0
