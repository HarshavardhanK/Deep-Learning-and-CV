# MNIST HANDWRITING RECOGNITION
import os
import struct
import numpy as np
from numpy.random import permutation as prm
import matplotlib.pyplot as plt

from scipy.special import expit
import sys

import shelve # to store the trained network

path="/Users/harshavardhank/Google Drive/Python/MachineLearning/MNIST"

def load(path, kind):
    #Load MNIST training data from path of kind

    labels_path = os.path.join(path, '%s-labels-idx1-ubyte' % kind)
    images_path = os.path.join(path, '%s-images-idx3-ubyte' % kind)

    with open(labels_path, 'rb') as l_path:
        m, n = struct.unpack('>II', l_path.read(8))
        labels = np.fromfile(l_path, dtype=np.uint8)

    with open(images_path, 'rb') as img_path:
        m, num, rows, cols = struct.unpack('>IIII', img_path.read(16))
        images = np.fromfile(img_path, dtype=np.uint8).reshape(len(labels), 784)


    return images, labels

X_train, y_train = load(path,'train')
print('Rows: %d, columns: %d' % (X_train.shape[0], X_train.shape[1]))

X_test, y_test = load(path,'t10k')
print('Rows: %d, columns: %d' % (X_test.shape[0], X_test.shape[1]))

def plot_digits(num):

    figure, axis = plt.subplots(nrows=5, ncols=5, sharex=True, sharey=True)
    axis = axis.flatten()

    for i in range(25):
        img = X_train[y_train == num][i].reshape(28, 28)
        axis[i].imshow(img, cmap='Greys', interpolation='nearest')

    axis[0].set_xticks([])
    axis[0].set_yticks([])
    plt.tight_layout()
    plt.show()

#plot_digits(4)


class NeuralNetwork(object):

    def __init__(self, num_out, num_feat, num_hid=30, l1=0.0, l2=0.0, num_iters=500, rate=0.001, alpha=0.0, dec_const=0.0, shuffle=True, min_batches=1, random_state=None):

        np.random.seed(random_state)
        self.num_out = num_out
        self.num_feat = num_feat #dimension of the features
        self.num_hid = num_hid #dimension of hidden layers

        self.w1, self.w2 = self.init_weights()

        self.l1 = l1 # L1 regularization params
        self.l2 = l2 #L2 regularization params
        self.epochs = num_iters #number of passes over the training set
        self.rate = rate #learning rate
        self.alpha = alpha
        self.dec_cosnt = dec_const #for adaptive learning rate
        self.shuffle = shuffle

        #(optional) Gradient descent to run for batches of the training set for optimal performance
        self.min_batches = min_batches

    #to convert label values, aka categorical values into numeric form

                                    #EDIT: integer encoding is foregone with
    def onehot_enc(self, y, k):
        onehot = np.zeros((k, y.shape[0]))

        for idx, val in enumerate(y):
            onehot[val, idx] = 1.0


        return onehot

    def init_weights(self):

        #randomly initialise weights to avoid making all the layers identical

        #w1: from input layer to hidden layer - (+1) for the bias

        w1 = np.random.uniform(-1.0, 1.0, size=self.num_hid * (self.num_feat + 1))
        w1 = w1.reshape(self.num_hid, self.num_feat + 1)

        #w1: from hidden layer to output layer - (+1)

        w2 = np.random.uniform(-1.0, 1.0, size=self.num_out * (self.num_hid + 1))
        w2 = w2.reshape(self.num_out, self.num_hid + 1)

        return w1, w2

    def sigmoid(self, z):
        return expit(z)

    def sigmoid_grad(self, z):
        sg = self.sigmoid(z)
        return sg * (1 - sg)

    #Add the bias unit

    #row wise wnd column-wise is to ensure it is added even after the transposes

    def add_bias(self, X, how='column'):

        if how == 'column':
            X1 = np.ones((X.shape[0], X.shape[1] + 1))
            X1[:, 1:] = X
        elif how == 'row':
            X1 = np.ones((X.shape[0] + 1, X.shape[1]))
            X1[1:, :] = X
        else:
            raise AttributeError('you must enter column or row.')

        return X1

    def feed_forward(self, X, w1, w2):

        a1 = self.add_bias(X,how='column')
        z2 = w1.dot(a1.T)

        a2 = self.sigmoid(z2)
        a2 = self.add_bias(a2, how='row')

        z3 = w2.dot(a2)

        a3 = self.sigmoid(z3)

        return a1, z2, a2, z3, a3

    def L2_reg(self, lamda, w1, w2):
        return (lamda / 2.0) * (np.sum(w1[:, 1:] ** 2) + np.sum(w2[:, 1:] ** 2))

    def L1_reg(self, lamda, w1, w2):
        return (lamda / 2.0) * (np.abs(w1[:, 1:]).sum() + np.abs(w2[:, 1:]).sum())

    def get_cost(self, y_enc, output, w1, w2):

        term1 = -y_enc * (np.log(output))
        term2 = (1 - y_enc) * np.log(1 - output)

        cost = np.sum(term1 - term2)

        #regularization, L1 and L2

        L1_term = self.L1_reg(self.l1, w1, w2)
        L2_term = self.L2_reg(self.l2, w1, w2)

        cost += L1_term + L2_term

        return cost

    def get_grad(self, a1, a2, a3, z2, y_enc, w1, w2):

        #backpropagation algorithm

        sigma3 = a3 - y_enc
        z2 = self.add_bias(z2, how='row')

        sigma2 = w2.T.dot(sigma3) * self.sigmoid_grad(z2)

        sigma2 = sigma2[1:, :]

        grad1 = sigma2.dot(a1)
        grad2 = sigma3.dot(a2.T)

        # regularize
        grad1[:, 1:] += (w1[:, 1:] * (self.l1 + self.l2))
        grad2[:, 1:] += (w2[:, 1:] * (self.l1 + self.l2))

        return grad1, grad2

    def predict(self, X):

        a1, z2, a2, z3, a3 = self.feed_forward(X, self.w1, self.w2)
        y = np.argmax(z3, axis=0) #finds the max activation in the output vector

        return y

    def curve_fit(self, X, y, print_progress=False):

        self.cost = []
        X_data, y_data = X.copy(), y.copy()
        y_enc = self.onehot_enc(y, self.num_out)

        delta_w1_prev = np.zeros(self.w1.shape)
        delta_w2_prev = np.zeros(self.w2.shape)

        for i in range(0, self.epochs):

            #adaptive learning rate
            self.rate /= (1 + self.dec_cosnt * i)

            if print_progress:
                print('Epoch: %d of %d' %(i+1, self.epochs))


            #shuffle to avoid bias or patterns, and to avoid loops
            if self.shuffle:
                idx = prm(y_data.shape[0])
                X_data, y_data = X_data[idx], y_data[idx]

            #cut the entire dataset into small batches, for better performance of GD
            mini = np.array_split(range(y_data.shape[0]), self.min_batches)

            for i in mini:

                #feedforward
                a1, z2, a2, z3, a3 = self.feed_forward(X[i], self.w1, self.w2)
                cost = self.get_cost(y_enc[:, i], a3, self.w1, self.w2)
                self.cost.append(cost)

                #compute gradient via backprop
                grad1, grad2 = self.get_grad(a1, a2, a3, z2, y_enc[:, i], self.w1, self.w2)


                #update weights
                delta_w1, delta_w2 = self.rate * grad1, self.rate * grad2

                self.w1 -= delta_w1 + (self.alpha * delta_w1_prev)
                self.w2 -= delta_w2 + (self.alpha * delta_w2_prev)

                delta_w1_prev, delta_w2_prev = delta_w1, delta_w2

        return self

    def get_weights(self):
        return self.w1, self.w2

#param values: 10 output neurons (for 10 digits), 28 input neurons (for 28x28 pixel), 50 units in hidden layers (arbitrarily chosen)
'''neuralNet = NeuralNetwork(10, X_train.shape[1], 100, 0.0, 0.1, 1000, 0.001, 0.001, 0.00001, True, 100, 1)
neuralNet.curve_fit(X_train, y_train, True)
w1, w2 = neuralNet.get_weights()'''

# iMac: 12min 35s for 1000 epochs with 100 units in hidden layer
# MacBook Pro: 12min 40s 1000 epochs with 100 units in hidden layer

'''print(w1)
print(w2)'''

def load_network(units):

    #50 units: handWT
    #100 units: Hundred

    saved_path="/Users/HarshavardhanK/Google Drive/Python/Project Manas/CourseraML/Week5/TrainedNetworks/HandwrittenDigits"

    load_shelf = shelve.open(saved_path)
    #type(load_shelf)
    neuralNet = load_shelf['Hundred']
    load_shelf.close()

    return neuralNet

neuralNet = load_network(100)

def save_network():

    #save the neuralNet using shelve to directly load instead of retraining whenever the program is run
                    #moved path on 4/12/2017
    shelf = shelve.open("/Users/HarshavardhanK/Google Drive/Python/Project Manas/CourseraML/Week5/TrainedNetworks/HandwrittenDigits", 'w')
    handWT = neuralNet #save this neural network as handWT- 'handWriting Trained'
    shelf['HandWT100'] = handWT
    shelf.close()

#save_network()

def plot_graph():

    batches = np.array_split(range(len(neuralNet.cost)), 1000)
    cost_array = np.array(neuralNet.cost)
    cost_avg = [np.mean(cost_array[i]) for i in batches]

    plt.plot(range(len(cost_avg)), cost_avg, color='red')
    plt.ylim([0, 2000])
    plt.ylabel('Cost')
    plt.xlabel('Epochs')
    plt.tight_layout()
    plt.show()

#plot_graph()

y_train_pred = neuralNet.predict(X_train)
acc = np.sum(y_train == y_train_pred, axis=0) / X_train.shape[0]
print('Training accuracy: %.2f%%' % (acc*100))

y_test_pred = neuralNet.predict(X_test)
acc = np.sum(y_test == y_test_pred, axis=0) / X_test.shape[0]
print('Test accuracy: %.2f%%' % (acc*100))


def plot_results(num):

    miscl_img = X_test[y_test == y_test_pred][:num ** 2]
    correct_lab = y_test[y_test == y_test_pred][:num ** 2]
    miscl_lab = y_test_pred[y_test == y_test_pred][:num ** 2]

    fig,ax = plt.subplots(num, num, True, True)
    ax = ax.flatten()

    for i in range(num ** 2):
        img = miscl_img[i].reshape(28, 28)
        ax[i].imshow(img, cmap='Greys', interpolation='nearest')
        ax[i].set_title('%d) t: %d p:%d' % (i+1, correct_lab[i], miscl_lab[i]))

    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.tight_layout()
    plt.show()

plot_results(7)
