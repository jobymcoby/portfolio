import math
import os
import pickle
import numpy as np

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy import linalg


def imageToCSV():
    # sprite = 'front\\front_0000_0.png'
    # img = Image.open(sprite)
    # arr = np.asarray(img)
    # print(arr.shape)

    dir = 'front\\'
    for file in os.listdir(dir):
        sprite = os.path.join(dir, file)
        img = Image.open(sprite)
        arr = np.asarray(img)
        # print(arr.shape) #64x64x4
        with open("spriteData.csv", "ab") as f:
            if(arr.shape[2] == 4):
                arr = arr.reshape(1, -1)
                print(arr.shape)
                np.savetxt(f, arr, delimiter=",", newline='\n')
            f.close()


def arrayToImage(sprite, number):
    spriteImage = Image.fromarray(sprite.astype(np.uint8))
    # spriteImage.show()
    spriteImage.save("output\\Output" + str(number) + ".png")

def createImage(arr, title):
    print("\nShow best image")

    arr = arr.reshape(28, 28)
    plt.imshow(arr, cmap=cm.gray)
    plt.axis('off')
    # plt.savefig("OutFiles\Part2-" + title + ".png")
    plt.show()

def generatePickleArr():
    arr = np.genfromtxt('spriteData.csv', delimiter=',', skip_header=False)
    arr = arr.reshape((arr.shape[0], 64, 64, 4))
    arr = arr[:,:,:, 0:3] #remove alpha layer
    print(arr.shape)
    file_pi = open('spriteArray.obj', 'wb')
    pickle.dump(arr, file_pi)

def saveAsPickle(arr, filename):
    file_pi = open(filename + ".obj", 'wb')
    pickle.dump(arr, file_pi)

def restorePickleArr(file):
    file = open(file, 'rb')
    arr = pickle.load(file)
    # print(arr.shape)
    return arr

def FID(X, Y):
    print("TEST")
    mu_x = np.mean(X)
    mu_y = np.mean(Y)
    print("mu_X:", mu_x, "mu_Y:", mu_y)

    ssdiff = np.sum((mu_x - mu_y)**2.0)

    cov_x = np.cov(X, rowvar=False)
    cov_y = np.cov(Y, rowvar=False)

    covmean = linalg.sqrtm(cov_x.dot(cov_y))

    if np.iscomplexobj(covmean):
        covmean = covmean.real
        # calculate score

    fid = ssdiff + np.trace(cov_x + cov_y - 2.0 * covmean)
    return fid
