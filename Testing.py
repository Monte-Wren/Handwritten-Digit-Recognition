#%% Import packages

import MNIST_Loader as load
import Neural_net as net

#%% Import data and unzip

training_data, validation_data, test_data = load.load_data_wrapper()
training_data = [i for i in training_data]
val_data = [i for i in validation_data]
test_data = [i for i in test_data]

#%% initialise neural network with 784 input neurons corresponding to 28x28 pixels
# and 10 output neurons corresponding to each digit from 0-9

Neural_net = net.Network([784, 30, 10])

#%% Train neural net on training_data
# Can adjust each parameter to play around but alpha = 3, mini_batch_size = 10 over 5 epochs gives a high enough accuracy
# without taking too long.
# Will take a couple minutes to train with these settings

Neural_net.train(training_data, 3, 5, 10, plot_iterations = True)

#%% Evaluate performance

Neural_net.evaluate(test_data)

