#feed forward algorithm in py

<<<<<<< HEAD
#1/12/2017

#EDIT 1: 2/12/2017: passed hidden layer through sigmoid

#ignore addition of bias
#support for addition of bias to de done later during EDIT 2



=======
#ignore addition of bias
#support for addition of bias to de done later during EDIT 2

>>>>>>> 6a18a5c2e778f58e0e57ac60a4533996c34c2b04
import numpy as np

def sigmoid(n):
    return 1.0 / (1.0 + np.exp(-n))

#Feed forward for 1 hidden layer

def feedForward(X, w1, w2):

    a1 = X # feature vector

    h2 = w1.dot(a1.T) #h for the hidden layer

    a2 = sigmoid(h2) #activation of the hidden layer g(h(X))

    h3 = w2.dot(a2) #h for the output layer

    a3 = sigmoid(h3) #activation of the output layer g(h(X))

    return a1, h2, a2, h3, a3

X = np.arange(4).reshape(4, 1)
w1 = np.arange(3, 7).reshape(4, 1)
w2 = np.arange(5, 9).reshape(1,4) #to match dimension
w3 = np.arange(3, 7).reshape(4, 1)

def feed_forward2(X, w1, w2, w3):
    a1 = X #feature vector

    h2 = w1.dot(a1.T)

    a2 = sigmoid(h2)

    h3 = w2.dot(a2)

    a3 = sigmoid(h3)

    h4 = w3.dot(a3)

    a4 = sigmoid(h4)

    return a1, h2, a2, h3, a3, h4, a4


#a1, h2, a2, h3, a3 = feedForward(X, w1, w2)
a1, h2, a2, h3, a3, h4, a4 = feed_forward2(X, w1, w2, w3)
<<<<<<< HEAD
'''print(a1)
print(h2)
print(a2)
print(h3)
print(a3)
print(h4)
print(a4)'''

def cost():
=======
>>>>>>> 6a18a5c2e778f58e0e57ac60a4533996c34c2b04
