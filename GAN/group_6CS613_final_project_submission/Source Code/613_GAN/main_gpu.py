import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
from Activation import ReLu
from Activation import ReLuTest
from FullyConnected import FullyConnected
from Activation import Sigmoid
from Output import LogLoss
from Output import Generator
from tqdm import tqdm
import cupy as cp
import time
import data_utils as utils


def createStochasticBatch(sourceBatch, batchSize):
    np.random.shuffle(sourceBatch)
    newArr = sourceBatch[:batchSize]
    return newArr


def createInput(batchSize):
    input = np.random.randint(256, size=(batchSize, 784))
    return input


def combineRealandFake(arr1, arr2):
    arrOut = np.append(arr1, arr2, axis=0)
    return arrOut


def ReluGAN(showEachEpoch):
    np.random.seed(0)
    batchSize = 100  # batch size for stochastic gradient descent
    numClasses = 1  # legacy from 615. Allows you to train multiple GANs for different classes
    numFeatures = 12288  # number of pixels (features) in the flattened picture
    originalShape = (64, 64, 3)  # shape of the picture
    maxEpochs = 1000  # number of epochs to run

    print("Reading training data. Please wait...")
    trainArr = utils.restorePickleArr()  # Get training data from pickle object

    # create column vector for labelling real and fake data.
    # This will be used by the discriminator to calculate its loss
    print("\nSet up target arrays based on batch size...")
    d_trainArrTarget = np.ones((batchSize, 1), dtype=int)  # real data are labeled 1
    d_fakeArrTarget = np.zeros((batchSize, 1), dtype=int)  # fake (generated) data are labeled 0
    targetArr_d = combineRealandFake(d_trainArrTarget, d_fakeArrTarget)  # combine (doesn't need to be a method)

    print("\nTraining...")
    for i in range(numClasses):  # for creating multiple GANs for multiple classes
        print("Index " + str(i) + "...")
        learningRate_G = 0.00001  # rate at which we update the weights in the generator
        learningRate_D = 0.00001  # rate at which we update the weights in the discriminator

        # Define the model
        # Generator
        FC_G = FullyConnected(numFeatures, numFeatures, learningRate_G)  # layer where learning actually happens
        relu_G = ReLu()  # activation layer. ReLu is fast and non-linear which makes it popular
        objective_G = Generator()  # Output/objective layer. It's mostly used for calculating loss and gradients

        # Discriminator
        FC_D = FullyConnected(numFeatures, 1, learningRate_D)  # layer where learning actually happens
        sig_D = Sigmoid()  # Activation layer. Forces values between 0 and 1
        LL_D = LogLoss(targetArr_d)  # Output.objective layer. It's mostly used for calculating loss and gradients

        # Time to start training
        # parameters
        epoch = 1  # training iteration counter
        jChange = 100  # metric for stopping training if our loss doesn't change much between epochs

        batchArr = trainArr  # arrList[i] # if we had multiple classes you would load the arrays into this array
        mu = cp.average(batchArr)  # get average for the entire data set
        sigma = cp.std(batchArr)  # get the SD for the entire data set
        batchArr = cp.asarray(batchArr)

        #initialization of cuda vars
        FCG_data = None
        FCD_data = None
        #FCGweights = cp.asarray(FC_G.weights)
        FCGweights = cp.asarray(0.0001 * (np.random.rand(numFeatures, numFeatures) - 0.5))
        #FCGbiases = cp.asarray(FC_G.biases)
        FCGbiases = cp.asarray(0.0001 * (np.random.rand(1, numFeatures) - 0.5))
        #FCDweights = cp.asarray(FC_D.weights)
        FCDweights = cp.asarray(0.0001 * (np.random.rand(numFeatures, 1) - 0.5))
        #FCDbiases = cp.asarray(FC_D.biases)
        FCDbiases = cp.asarray(0.0001 * (np.random.rand(1, 1) - 0.5))
        relu_g_data = cp.array([])
        sig_d_dataIn = None
        sig_d_dataOut = None
        # while jChange > 10 ** -6 and epoch <= maxEpochs: # replaced with a loop with a progress bar
        for i in tqdm(range(maxEpochs)):  # print a progress bar, estimated time, and rate
            # print(i)
            # Fake input
            # Generate random data that has the same mean and SD as the training data
            #input_f = np.random.normal(mu, sigma, size=(batchSize, numFeatures))
            input_f = cp.random.normal(mu, sigma, size=(batchSize, numFeatures))


            # Real input
            #input_r = createStochasticBatch(batchArr, batchSize)
            cp.random.shuffle(batchArr)
            input_r = batchArr[:batchSize]


            # Forward Prop
            #X_f = FC_G.forwardPropagate(input_f)
            #X_f = relu_G.forwardPropagate(X_f)  # Generated fake data
            FCG_data = input_f
            X_f = cp.dot(input_f, FCGweights) + FCGbiases
            relu_g_data = X_f
            X_f = cp.maximum(X_f, 0)


            #input_d = combineRealandFake(input_r, X_f)
            input_d = cp.append(input_r, X_f, axis=0)

            ###############################################
            #X_d = FC_D.forwardPropagate(input_d)
            #X_d = sig_D.forwardPropagate(X_d)  # prediction by the discriminator (real=1 or fake=0)
            FCD_data = input_d
            X_d = cp.dot(input_d, FCDweights) + FCDbiases
            sig_d_dataIn = X_d
            X_d = cp.array(1 / (1 + cp.exp(-X_d)))
            sig_d_dataOut = X_d
            ###############################################
            #TODO REMOVE and keep as CP array when possible
            X_d = cp.asnumpy(X_d)
            J_d = LL_D.eval(X_d)  # print this to see the loss for the discriminator

            # back prop discriminator
            grad = LL_D.gradient(X_d)  # calculate the gradient of discriminator

            ###############################################
            #grad = sig_D.backwardPropagate(grad)  # account for the activation layer changes
            grad = cp.asarray(grad)
            temp = cp.array(1-sig_d_dataOut)
            temp = cp.multiply(sig_d_dataOut, temp)
            grad = grad * temp
            ###############################################

            ###############################################
            #FC_D.backwardPropagate(grad, epoch)  # update weights and biases for discriminator
            observationCount = grad.shape[0]  # TODO: Confirm this

            #Cache gradient before updating weights
            gradOut = cp.dot(grad,cp.transpose(FCDweights))

            # update weights
            dW = cp.transpose(FCD_data)
            dJdW = cp.dot(dW, grad)

            FCDweights = FCDweights + 0.0001/observationCount*(-dJdW)

            # update biases
            db = cp.transpose(cp.ones((observationCount, 1)))
            dJdb = cp.dot(db, grad)
            FCDbiases = FCDbiases + observationCount*(-dJdb)
            ###############################################


            # forward prop again b/c the generator should learn based on the updated discriminator, not the old one
            #X_d = FC_D.forwardPropagate(X_f)  # This time only forward prop the fake data from earlier
            FCD_data = X_f
            X_d = cp.dot(X_f,FCDweights) + FCDbiases
            #X_d = sig_D.forwardPropagate(X_d)  # Prediction from discriminator
            sig_d_dataIn = X_d
            X_d = cp.array(1 / (1 + cp.exp(-X_d)))
            sig_d_dataOut = X_d

            #J_g = objective_G.eval(X_d)  # print this to see the loss for the generator
            observationCount = X_d.shape[0]  # TODO: Confirm this
            J_g = -cp.log(X_d + (10 ** -7))

            # back again
            ###############################################
            #grad = objective_G.gradient(X_d)  # calculate the gradient of discriminator
            grad = -1/(X_d + (10** -7))
            #grad = sig_D.backwardPropagate(grad)  # account for the activation layer changes
            arr1 = sig_d_dataOut  # arr1 = self.forwardPropagate(self.dataIn)
            arr2 = cp.array(1 - arr1)
            output = cp.multiply(arr1, arr2)
            grad = grad * output
            ###############################################

            ###############################################
            #grad = FC_D.backwardPropagateNoUpdate(grad)  # account for the FC layer in discriminator but dont update it
            observationCount = grad.shape[0]  # TODO: Confirm this
            #Cache gradient before updating weights
            gradOut = cp.dot(grad , cp.transpose(FCDweights))
            # update weights
            dW = cp.transpose(FCD_data)
            dJdW = cp.dot(dW , grad)
            FCDweights = FCDweights + 0.0001/observationCount*(-dJdW)
            # update biases
            db = cp.transpose(cp.ones((observationCount, 1)))
            dJdb = cp.dot(db , grad)
            FCDbiases = FCDbiases + 0.0001/observationCount*(-dJdb)
            grad = gradOut
            ###############################################

            ###############################################
            #grad = relu_G.backwardPropagate(grad)  # account for the activation layer changes
            output = cp.where(relu_g_data > 0, 1, 0)
            grad = cp.multiply(grad, output)# gradIn * self.gradient()
            ###############################################


            ###############################################
            #FC_G.backwardPropagate(grad, epoch)  # update weights and biases in fully connected layer for generator
            observationCount = grad.shape[0]  # TODO: Confirm this

            #Cache gradient before updating weights
            gradOut = cp.dot(grad , cp.transpose(FCGweights))

            # update weights
            dW = cp.transpose(FCG_data)
            dJdW = cp.dot(dW , grad)

            FCGweights = FCGweights + 0.0001/observationCount*(-dJdW)

            # update biases
            db = cp.transpose(cp.ones((observationCount, 1)))
            dJdb = cp.dot(db , grad)
            FCGbiases = FCGbiases + 0.0001/observationCount*(-dJdb)
            ###############################################

            epoch += 1

            if showEachEpoch and epoch % 1000 == 0:
                X_d = cp.asnumpy(X_d)
                X_f = cp.asnumpy(X_f)
                best_index = np.argmax(X_d, axis=0)  # find index for the best fake image
                best = X_f[best_index]  # get the data for that image
                best = best.reshape(originalShape)  # reshape it to the original size
                utils.arrayToImage(best, epoch)  # show the image (currently also saves the image as Output.png)
                X_d = cp.asarray(X_d)
                X_f = cp.asarray(X_f)
                # utils.createImage(best, "d" + str(i) + "e" + str(epoch))  # legacy from 615
        if not showEachEpoch:
            best_index = np.argmax(X_d, axis=0)  # find index for the best fake image
            best = X_f[best_index]  # get the data for that image
            best = best.reshape(originalShape)  # reshape it to the original size
            utils.arrayToImage(best, epoch)  # show the image (currently also saves the image as Output.png)
            # utils.createImage(best, "d" + str(i) + "e" + str(epoch)) # legacy from 615


ReluGAN(True)
