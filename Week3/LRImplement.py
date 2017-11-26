# ANDREW NG LOGISTIC REGRESSION IMPLEMENTATION

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

# Plot the logistic curve

def plot_lr_curve():

    x = np.arange(-10, 10, 0.2)
    phi_x = sigmoid(x)
    plt.plot(x, phi_x)
    plt.axvline(0.0, color='k')
    #plt.axhspan(0.0, 1.0, facecolor='1.0'.alpha=1.0, ls='dotted')
    plt.axhline(y=0.5, ls='dotted', color='k')
    plt.yticks([0.0, 0.5, 1.0])
    plt.ylim(-0.1, 1.1)
    plt.xlabel('x')
    plt.ylabel('Sigmoid curve')
    plt.show()

X = np.array([1, 2,3,4,5,6,7,7,8]).reshape(9, 1)
y = np.array([1, 0, 0, 1, 1, 0, 0, 1, 2]).reshape(9,1)
#print(X, y)
t = 2

def test_sigmoid():

    sig_X = sigmoid(X)
    print(sig_X)


def logartihmic_cost_function(X, y, t):

    m = X.shape[0]
    h = sigmoid(np.dot(X, t))

    loss1 = -y * np.log10(h)
    loss2 = (1 - y) * np.log10(1 - h)

    total_loss = (loss1 - loss2) / m

    return total_loss

#print(logartihmic_cost_function(X, y, t))

def gradient_descent(X, y, a, n_iters):

    theta = np.ones(2)

    m = X.shape[0]

    for _ in range(n_iters):

        h = sigmoid(np.dot(X, theta))
        loss = y - h

        gradient = np.dot(X.transpose(), loss) / m

        theta = theta - a * gradient

    return theta

#gd = gradient_descent(X, y, 0.01, 20)
#print(gd)
