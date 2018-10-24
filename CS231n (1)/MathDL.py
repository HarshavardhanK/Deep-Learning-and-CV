#Mathematical functions given in the lecture and lecture notes are implemented here
import numpy as np
import matplotlib.pyplot as plt

def softmax(X):

    S = []

    for k in range(X.shape[0]):

        numerator = np.exp(X[k])
        denominator = np.sum(np.exp(X))
        S.append(numerator / denominator)

    return S

#completely vectorized code

def softmax2(X):
    return np.exp(X) / np.sum(np.exp(X), axis=0)

def softmaxLoss(S):
    return -np.log(softmax(S))

def printFunctions():

    X = np.array([1,2,3,4,5])

    #print("Softmax", softmax(X))
    print("Softmax2", softmax2(X))
    print("Softmax ", softmaxLoss(X))


printFunctions()

def plotFunctions():

    X = np.arange(0,30)

    plt.plot(X, softmax2(X))
    plt.plot(X, softmaxLoss(X))
    plt.show()
