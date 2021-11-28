import numpy as np
import MNIST_Loader as load
import random
import matplotlib.pyplot as plt

#%% Functions

def sigmoid(z):
    return 1/(1+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

#%% Main neural net

class Network(object):
    
    def __init__(self, sizes):
        '''
        Initialises Network class with number of neurons in each layer of the network,
        e.g the list [10, 20, 1] would correspond to a 3 layer network with 10 input
        neurons, 20 hidden layer neurons and 1 output neuron. The initial biases are
        initialised to one and the initial weights are initialised randomly using a 
        Normal distribution, mean zero and variance 1. The biases for the input layer
        are initialised to zero as these should only affect the hidden layers.
        '''
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.ones((y, 1)) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):       
        """
        returns a list of the output of the network, along with the arguments z of the 
        activation function at each step.
        """
        activation = a
        activations = [a]
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        return activations, zs
    
    def backprop(self, x, y):
        """
        Backpropagates test output y and computes "error" in each neuron after feeding forward
        the corresponding test sample x.
        Returns a tuple of the errors in the biases and weights 
        """
        (activations, zs) = self.feedforward(x)
        output = activations[-1]
        Delta_b = [np.zeros(b.shape) for b in self.biases]
        Delta_w = [np.zeros(w.shape) for w in self.weights]
        
        delta = (output - y)*sigmoid_prime(zs[-1])
        
        Delta_b[-1] = delta
        Delta_w[-1] = np.dot(delta, activations[-2].transpose())
        
        for i in range(2, self.num_layers):
            z = zs[-i]
            delta = np.dot(self.weights[-i+1].transpose(), delta)*sigmoid_prime(z)
            Delta_b[-i] = delta
            Delta_w[-i] = np.dot(delta, activations[-i-1].transpose())
        
        return (Delta_b, Delta_w)
       
    def train(self, training_data, alpha, epochs, mini_batch_size, plot_iterations = False):
        """
        Updates the biases and weights of the network via gradient descent using the tuples
        returned from backpropagating. Iterates over entire, randomly shuffled, batched training set, 
        with epochs determining number of times the training set is run through (can get slow).
        Also gives the option to plot the cost of each sample against iteration cycle.
        (Cost given below)
        """
        costs = [self.cost(training_data)]
        m = len(training_data)
        for j in range(epochs):
            #Split training dataset into random mini batches
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, m, mini_batch_size)]
            for mini_batch in mini_batches:
                grad_b = [np.zeros(b.shape) for b in self.biases]
                grad_w = [np.zeros(w.shape) for w in self.weights]
                
                # for each mini batch, apply gradient descent
                for x, y in mini_batch:             
                    Delta_b, Delta_w = self.backprop(x, y)
                    grad_b = [nb + dnb for nb, dnb in zip(grad_b, Delta_b)]
                    grad_w = [nw + dnw for nw, dnw in zip(grad_w, Delta_w)]
                    
                self.weights = [w-(alpha/len(mini_batch))*nw for w, nw in zip(self.weights, grad_w)]
                self.biases = [b-(alpha/len(mini_batch))*nb for b, nb in zip(self.biases, grad_b)]
            
            #calculate cost
            costs.append(self.cost(training_data))
            print("epoch {0} Complete".format(j+1))
        
        #plot cost if desired
        if plot_iterations == True:
            iterations = np.arange(0, epochs+1, 1)
            plt.plot(iterations, costs)
            plt.xlabel("Epoch")
            plt.ylabel("Cost")
            plt.show()    
        else:
            pass
       
    def cost(self, training_data):
        """
        Calculates error in forward propagated training samples and corresponding outputs based off 
        sigmoid activation function.
        """
        cost = []
        for x, y in training_data:
            activations, zs = self.feedforward(x)
            cost.append(sum(np.dot(y.transpose(), np.log(activations[-1])) + np.dot((1-y).transpose(), np.log(1-activations[-1]))))
        return float((-1/len(training_data))*sum(cost))
            
    def predict(self, test_data, index, plot_image = False):
        """
        Returns neural net's prediction given test sample
        Also gives option to display the test sample
        """
        activations, zs = self.feedforward(test_data[index][0])
        result = activations[-1]
        if plot_image == True:
            load.display_image(test_data, index)
            print("prediction: {}".format(np.argmax(result)))
        else:    
            print("prediction: {}".format(np.argmax(result)))
        
    def evaluate(self, test_data):
        """
        Evaluates performance of neural net over entire test data set. 
        Returns percentage of correct predictions.
        """
        test_results = [(np.argmax(self.feedforward(x)[0][-1]), y) for x, y in test_data]
        percentage = 100*sum(int(x==y) for x, y in test_results)/len(test_data)
        print("Accuracy: {}%".format(percentage))