import pandas as pd
import numpy as np

class InputLayer:
    def __init__(self, dataIn):
        self.__meanX = np.mean(dataIn, 0) #dataIn.mean(0)
        self.__stdX = np.std(dataIn, 0) #dataIn.std(0)

        # clean up and std values that are 0 and convert them to 1
        with np.nditer(self.__stdX, op_flags=['readwrite']) as std:
            for x in std:
                if x[...] == 0:
                    x[...] = 1

        # print(self.__stdX)

    def forwardPropagate(self, X):
        _x = pd.DataFrame(X)
        _x = _x.apply(lambda x: (x - self.__meanX)/self.__stdX, axis=1)
        _x = _x.to_numpy()

        return _x

    def backwardPropagate(self, gradIn):
        pass
