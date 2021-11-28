# Handwritten-Digit-Recognition
A neural network, trained off of the MNIST data set, that recognises handwritten digits with 95%+ accuracy.
Heavily inspired by Andrew Ng's Machine Learning course on Coursera and the free online book: 'Neural Networks and Deep Learning' by Michael Nielsen, specifically chapter 1. The MNIST_Loader.py, with a couple alterations, was taken directly from the online book.

Run the MNIST_Loader.py and Neural_net.py files to initialise everything. The Testing.py file initialises and trains the neural net for you as well as evaluating the performance.
Neural_net.predict(test_data, index, plot_image = True) will plot the test digit and output the neural network's prediction for that digit for a given index in the dataset. Other implementations are pretty self explanatory.
