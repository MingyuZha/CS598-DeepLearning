import numpy as np
import h5py
from time import *
from random import randint
# from scipy import signal

def softmax(z):
    return np.exp(z)/np.sum(np.exp(z), axis=0)

def sigmoid_prime(z):
    return (1-z)*z

def sigmoid(z):
    return (1./(1+np.exp(-z)))

def tanh(z):
    return ((np.exp(z)-np.exp(-z))/(np.exp(z)+np.exp(-z)))

def tanh_prime(h):
    return (1-np.power(h,2))

class Convolutional_Neural_Network:
    def __init__(self, LR):
        self.LR = LR

        #Load the data
        MNIST_data = h5py.File('MNISTdata.hdf5', 'r')
        self.x_train = np.float32(MNIST_data['x_train'][:])
        self.y_train = np.int32(np.array(MNIST_data['y_train'][:,0]))
        self.x_test = np.float32(MNIST_data['x_test'][:])
        self.y_test = np.int32(np.array(MNIST_data['y_test'][:,0]))
        self.d = int(np.sqrt(self.x_train[0].shape[0]))
        self.x_train = self.x_train.reshape(self.x_train.shape[0], self.d, self.d)
        self.x_test = self.x_test.reshape(self.x_test.shape[0], self.d, self.d)
        MNIST_data.close()

        #Initialize 3 channels with each channel having a shape of d_y*d_x
        self.d_y = self.d_x = 3
        self.num_channels = 3
        self.num_class = 10
        self.K = np.random.normal(0, np.sqrt(2/(784+self.d_y*self.d_x)), size=(self.num_channels, self.d_y, self.d_x))
        self.W = np.random.randn(self.num_class, self.num_channels, self.d-self.d_y+1, self.d-self.d_x+1)*np.sqrt(2/(self.num_channels*self.d_y*self.d_x))
        self.b = np.zeros((self.num_class,1))

    def convolution(self, X, K): 
        d = X.shape[0]
        d_y = K.shape[0]
        d_x = K.shape[1]
        Z = np.zeros((d-d_y+1, d-d_x+1))
        # Z = signal.convolve2d(X, K, mode='valid')
        for i in range(d-d_y+1):
            for j in range(d-d_x+1):
                Z[i,j] = np.tensordot(K, X[i:(i+d_y), j:(j+d_x)], axes=((0,1),(0,1)))
        return Z

    def forward(self, X):
        Z = np.zeros((self.num_channels, self.d-self.d_y+1, self.d-self.d_x+1))

        for k in range(self.num_channels):
            for i in range(self.d-self.d_y+1):
                for j in range(self.d-self.d_x+1):
                    Z[k,i,j] = np.tensordot(self.K[k], X[i:(i+self.d_y), j:(j+self.d_x)], axes = ((0,1),(0,1)))
        # for k in range(self.num_channels):
        #     Z[k] = signal.convolve2d(X, self.K[k], mode='valid')
        H = sigmoid(Z)
        U = (np.tensordot(self.W, H, axes=((1,2,3),(0,1,2)))).reshape(-1,1)+self.b
        f = softmax(U)

        cache = {"H": H, "U":U}
        return f, cache

    def backward(self, f, X, y, cache):
        d_V = -1.0*f
        d_V[y] += 1
        d_V *= -1

        d_b = d_V
        d_W = np.zeros(self.W.shape)
        for i in range(10):
            d_W[i] = d_V[i]*cache["H"]

        delta = np.tensordot(self.W, d_V.flatten(), axes=((0),(0)))
        

        d_K = np.zeros(self.K.shape)
        for i in range(self.num_channels):
            d_K[i] = self.convolution(X, sigmoid_prime(cache["H"][i]) * delta[i])

        #Update
        self.b -= self.LR * d_b
        self.W -= self.LR * d_W
        self.K -= self.LR * d_K

  

    def train(self, num_epochs):
        print ("Start training process!")
        start = time()
        for i in range(num_epochs):
            for j in range(20000):
                #Randomly choose samples from training set
                rand_index = randint(0, len(self.x_train)-1)
                X = self.x_train[rand_index]
                Y = np.array(self.y_train[rand_index])
                #Forward propagation
                f, cache = self.forward(X)
                #Backward Propagation
                self.backward(f, X, Y,cache)
                if ((j+1)%1000==0): print ("Trained %i samples"%(j+1), " in epoch %i"%(i+1), "Cost: %.2f seconds"%(time()-start))
            # f = []
            # for j in range(len(self.x_train)):
            #     y_predict, _ = self.forward(self.x_train[j])
            #     f.append(y_predict.flatten())
            # f = np.array(f)

            print ("Epoch %i Finined"%(i+1))
            # print ("Epoch %i: "%(i+1), ", Loss: ", self.loss(f.T, self.y_train), ", Train Accuracy: ", self.accuracy(f.T, self.y_train))

        print ("Finished training process! Used %.2f seconds"%(time()-start))
        


    def accuracy(self, f, Y_true):
        y_hat = np.argmax(f, axis=0)
        acc = np.mean(np.int32(y_hat == Y_true))
        return acc

    def loss(self, y_hat, y_true):
        #Calculate the loss using cross-entroy error
        l = -np.sum(list(np.log(y_hat[y_true[i], i]) for i in range(len(y_true))))
        return l

    def evaluate(self):
        f = []
        for i in range(len(self.x_test)):
            y_predict, _ = self.forward(self.x_test[i])
            f.append(y_predict.flatten())
        f = np.array(f)
        acc = self.accuracy(f.T, self.y_test)
        print ("Model Accuracy: ", acc)

if __name__ == "__main__":
    model = Convolutional_Neural_Network(0.1)
    model.train(num_epochs=5)
    model.evaluate()
    





