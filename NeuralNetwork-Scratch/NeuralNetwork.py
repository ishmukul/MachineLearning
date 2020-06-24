# A simple two layer Neural Network from scratch to solve XOR problem, which is not linearly separable.
# Python numpy library has been used for generating arrays.
#
# Class NeuralNet should be instantiated first.
# Functions get_weights and get_steps implemented.
# Input
# X = (-0.5, -0.5), (-0.5, 0.5), (0.5, -0.5), (0.5, 0.5)
# Y = (-0.5), (0.5), (0.5), (-0.5)

# Import numpy and plotting library
import numpy as np
import matplotlib.pyplot as plt

# Close any open matplotlib graph window
plt.close('all')


# Define a 1D plotting functions
def plot_1d(var, figsame=False, dpi=150, alabel=None, llabel=None, title=None, filename="", xtick=None):
    '''
    :param figsame: Use same canvas or start a new one
    :param dpi: Dot per Inch, resolution of image
    :param alabel: Axis labels
    :param llabel: Legend label
    :param title: Plot Title label
    :param filename: Filename for saving file
    :param xtick: Xtick labels, if other than normal
    '''
    # : Use same Canvas or start a new canvas
    if figsame:
        plt.figure(figsize=plt.figaspect(1.), dpi=dpi)
    if alabel is None:
        alabel = ["Steps", "Counts"]
    if title is None:
        title = "1D Plot"
    if llabel is None:
        llabel = "Plot"
    x2 = var
    x1 = range(1, len(var) + 1)

    plt.plot(x1, x2, '-', Linewidth=3, label=llabel)

    plt.xlabel(alabel[0])
    plt.ylabel(alabel[1])

    if xtick:
        plt.xticks(x1, xtick, rotation=90)
    plt.title(title)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.grid()
    if filename:
        plt.savefig(filename, bbox_inches='tight')


class NeuralNet:

    def __init__(self, n_h=2, activ_func="tanh", precision=1e-5, learning_rate=0.01, print_step=100):
        '''
        :param n_h: Number of neurons in the hidden layer
        :param activ_func: Activation function. sigmoid, tanh, ReLU
        :param precision: Precision value at which optimizer stops
        :param learning_rate: Learning rate of Optimizer
        :param print_step: Option to print costs after print_step steps
        '''
        self.n_h = n_h
        self.activ_func = activ_func
        self.precision = precision
        self.learning_rate = learning_rate
        self.print_step = print_step

    # Define variable activation functions.
    def activation(self, x):
        '''
        Activation function
        :param x: X matrix
        Activation function Options: sigmoid, tanh, ReLU
        '''
        if self.activ_func == "sigmoid":
            return 1 / (1 + np.exp(-x))
        elif self.activ_func == "tanh":
            return np.tanh(x)
        elif self.activ_func == "relu":
            return np.maximum(0, x)

    # Define derivatives of activation functions
    def activation_d(self, x):
        '''
        Derivative of activation function
        :param x: X matrix
        Activation function Options: sigmoid, tanh, ReLU
        '''
        if self.activ_func == "sigmoid":
            z = 1 / (1 + np.exp(-x))
            return z * (1 - z)
        elif self.activ_func == "tanh":
            return 1 - np.square(np.tanh(x))
        elif self.activ_func == "relu":
            x[x <= 0] = 0
            x[x > 0] = 1
            return x

    def fit(self, x, y):
        '''
        Fitting function for neural network.
        :param x: X feature matrix
        :param y: Y output classifiers
        :return: Returns Youtput, costs and number of steps in calculations
        '''

        self.x = x
        self.y = y

        # Calculate Number of samples, Number of neurons
        m = self.x.shape[1]  # Number of samples
        n_x = self.x.shape[0]  # Features in X
        n_h = self.n_h  # Number of neurons
        n_y = self.y.shape[0]  # Features in Y

        # Initialize Weights and bias for forward propagation
        self.w1 = np.random.randn(n_h, n_x)
        self.w2 = np.random.randn(n_y, n_h)
        self.b1 = np.zeros((n_h, 1))
        self.b2 = np.zeros((n_y, 1))
        self.a2 = np.zeros(y.shape)

        # Initialize cost function list
        self.costs = []
        cost1 = 0  # Used to determine stopping condition for while loop
        delta_cost = np.inf  # Initialize delta_cost
        self.n_step = 0  # Step number increment

        # In case you need to run a for loop, n_iter needs to be defined in the function
        # for i in range(1, n_iter):
        # Loop for optimization
        while delta_cost > self.precision:
            self.n_step += 1

            # Forward propagation
            z1 = self.w1 @ self.x + self.b1
            a1 = self.activation(z1)
            z2 = self.w2 @ a1 + self.b2
            self.a2 = self.activation(z2)

            # Cost function. A simple sum of square difference
            cost = np.sum(np.square(self.a2 - self.y))
            self.costs.append(cost)

            # Back propagation and gradients
            dz2 = 2 * (self.a2 - self.y) * self.activation_d(z2)
            dw2 = dz2 @ a1.T
            db2 = 1 / m * np.sum(dz2, axis=1, keepdims=True)
            dz1 = (self.w2.T @ dz2) * self.activation_d(z1)
            dw1 = dz1 @ self.x.T
            db1 = 1 / m * np.sum(dz1, axis=1, keepdims=True)

            # Update parameters
            self.w1 = self.w1 - self.learning_rate * dw1
            self.w2 = self.w2 - self.learning_rate * dw2
            self.b1 = self.b1 - self.learning_rate * db1
            self.b2 = self.b2 - self.learning_rate * db2

            # Calculations for while loop
            delta_cost = np.abs(cost1 - cost)
            cost1 = cost

            # Option to print at some interval.
            # if n_step % print_step == 0:
            #     print("Cost at step %d is %f" % (n_step, cost))

        return self.a2, self.costs, self.n_step

    def get_weights(self):
        '''
        Get weight parameters of fiited neural network.
        :return: Weights and bias
        '''
        # print("Weights W1:", self.w1)
        # print("Weights b1:", self.b1)
        # print("Weights W2:", self.w2)
        # print("Weights b2:", self.b2)
        return self.w1, self.b1, self.w2, self.b2

    def get_steps(self):
        '''
        Get number of steps for reaching precision value
        :return: N steps
        '''
        # print(self.n_step)
        return self.n_step


# Input data
# Input is a matrix with dimensions (2,4) where 2 is number of X features and 4 is training sample.
# Configuration is different from regular convention; Followed from Andrew Ng's course on Neural Networks.

# X = np.array([[0, 0, 1, 1], [0, 1, 0, 1]])
X = np.array([[-0.5, 0.5, -0.5, 0.5], [-0.5, -0.5, 0.5, 0.5]])

# Output Y values for training
# Y = np.array([[0, 1, 1, 0]])
Y = np.array([[-0.5, 0.5, 0.5, -0.5]])

# Setting seed for consistent values in each run. Disable to generate random results.
np.random.seed(1)

# This could can run for a fixed number of neurons. However, I am searching the effect of neuron numbers on output.
# Running for 5 loops with neurons in hidden layer to 2,4,6,8,10
for n in np.arange(2, 12, 2):
    # Instantiate Neural Network
    nn = NeuralNet(n_h=n, activ_func='tanh', precision=1e-5, learning_rate=0.1, print_step=1000)

    # Fit network to get Output, costs and n_steps to reach precision.
    Yhat, costs, n_steps = nn.fit(X, Y)
    # print(Yhat)  # Print calculated YHat values
    # nn.get_weights()  # Optional to print weights

    # Since Yhat gives probabilities, a probability of greater than 0.5 corresponds to 1 (high) output.
    # Yhat = (Yhat > 0.5).astype(int)

    Yhat = np.round(Yhat, 1)  # Round values to 1 decimal place.
    print('N_neurons:' + str(n) + ', N_steps:' + str(n_steps) + ', Output:' + str(Yhat))

    # Plot cost function and save in a file
    axislabel = ['Steps', 'Cost']
    gtitle = 'Cost function'
    fname = 'Cost_NN.png'
    plotlabel = 'Neurons: ' + str(n)
    plot_1d(costs, figsame=False, alabel=axislabel, llabel=plotlabel, title=gtitle, filename=fname)

# Output
# Plot of cost function is saved.
# N_neurons:2, N_steps:215, Output:[[-0.5  0.5  0.5 -0.5]]
# N_neurons:4, N_steps:96, Output:[[-0.5  0.5  0.5 -0.5]]
# N_neurons:6, N_steps:92, Output:[[-0.5  0.5  0.5 -0.5]]
# N_neurons:8, N_steps:87, Output:[[-0.5  0.5  0.5 -0.5]]
# N_neurons:10, N_steps:108, Output:[[-0.5  0.5  0.5 -0.5]]

# It seems that Layer 8 converges fastest.
# Since seed number is fixed, output will remain same at each run.

# A simple 2 layer neural network can solve XOR problem easily.
# We will try for more hidden layers in next update.
