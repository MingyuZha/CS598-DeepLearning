# @author: Mingyu

import h5py
import numpy as np
from time import *
from copy import *
from random import randint

def tanh(z):
        return ((np.exp(z)-np.exp(-z))/(np.exp(z)+np.exp(-z)))

def sigmoid(z):
    return (1./(1+np.exp(-z)))

def softmax(z):
    return np.exp(z)/np.sum(np.exp(z), axis=0)

def Relu(z):
    mask = np.int32(z > 0)
    return (z*mask)

class Neural_Network:
    def __init__(self, alpha, H):
        """
        Parameters:
        alpha: a scalar, learning rate
        H: a scalar, the number of hidden units
        """
        self.alpha = alpha
        self.H = H
        #Initialize parameters of the model
        self.W1 = np.random.randn(H, 784)*np.sqrt(2/784)
        self.b1 = np.zeros((H, 1))
        self.W2 = np.random.randn(10, H)*np.sqrt(2/H)
        self.b2 = np.zeros((10, 1))
        #Load the MNIST dataset into the model
        self.load_data()

    def load_data(self):
        #load MNIST data
        MNIST_data = h5py.File('MNISTdata.hdf5', 'r')
        self.x_train = np.float32(MNIST_data['x_train'][:])
        self.y_train = np.int32(np.array(MNIST_data['y_train'][:,0]))
        self.x_test = np.float32(MNIST_data['x_test'][:])
        self.y_test = np.int32(np.array(MNIST_data['y_test'][:,0]))

        MNIST_data.close()


    def forward(self, X):
        z1 = np.dot(self.W1, X)+self.b1
        A1 = sigmoid(z1)
        z2 = np.dot(self.W2, A1)+self.b2
        A2 = softmax(z2)
        y_hat = A2
        cache = {"z1":z1, "A1":A1, "z2":z2, "A2":A2}
        return y_hat, cache

    def loss(self, y_hat, y_true):
        #Calculate the loss using cross-entroy error
        l = -np.sum(list(np.log(y_hat[y_true[i], i]) for i in range(len(y_true))))
        return l

    def backward(self, f, X, Y, cache, LR):
        #Calculate the derivatives
        d_V = -1.0*f
        d_V[Y] += 1
        d_V *= -1

        d_b2 = d_V

        d_W2 = np.dot(d_V, cache['A1'].T)
        
        sigma = np.dot(self.W2.T, d_V)

        # d_b1 = sigma*(1-np.power(cache['A1'], 2))
        d_b1 = sigma*(cache['A1']*(1-cache['A1']))

        d_W1 = np.dot(d_b1, X.transpose())

        #Update parameters of the model via stochastic gradient descent
        self.W1 -= LR * d_W1
        self.b1 -= LR * d_b1
        self.W2 -= LR * d_W2
        self.b2 -= LR * d_b2

    def train(self, num_epochs, batch_size):
        print ("Start training process!")
        start = time()
        for i in range(num_epochs):
            for j in range(len(self.x_train)):
                #Randomly choose samples from training set
                rand_index = randint(0, len(self.x_train)-1)
                X = self.x_train[rand_index].reshape((-1,1))
                Y = np.array(self.y_train[rand_index])
                #Forward propagation
                y_hat, cache = self.forward(X)
                #Backward Propagation
                self.backward(y_hat, X, Y,cache, LR=self.alpha)
                if ((j+1)%1000==0): print ("Trained %i samples"%(j+1), " in epoch %i"%(i+1), "Cost: %.2f seconds"%(time()-start))
            y_predict, _ = self.forward(self.x_train.T)
            print ("Epoch %i: "%(i+1), ", Loss: ", self.loss(y_predict, self.y_train), ", Train Accuracy: ", self.accuracy(self.x_train, self.y_train))

        print ("Finished training process! Used %.2f seconds"%(time()-start))

    def accuracy(self, X, Y_true):
        output, _ = self.forward(X.T)
        y_hat = np.argmax(output, axis=0)
        acc = np.mean(np.int32(y_hat == Y_true))
        return acc

    def evaluate(self):
        output, _ = self.forward(self.x_test.transpose())
        y_predict = np.argmax(output, axis=0)
        correct = 0
        for i in range(len(y_predict)):
            if (y_predict[i] == self.y_test[i]): correct += 1
        print ("Model Accuracy: ", float(correct) / len(y_predict))

if __name__ == '__main__':
    model = Neural_Network(alpha=0.01, H = 100)
    model.train(num_epochs=30, batch_size=1)
    model.evaluate()




