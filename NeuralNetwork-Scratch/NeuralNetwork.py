# A simple two layer Neural Network from scratch to solve XOR problem, which is not linearly separable.
# Python numpy library has been used for generating arrays.
#
# Classes are not used in this example since it was a simple 2 layer network.
# In next update, multi layer network will be implemented with class variables.

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


# Define variable activation functions.
def activation(x, func):
    '''
    Activation function
    :param x: X matrix
    :param func: Activation function
                Options: sigmoid, tanh, ReLU
    '''
    if func == "sigmoid":
        return 1 / (1 + np.exp(-x))
    elif func == "tanh":
        return np.tanh(x)
    elif func == "relu":
        return np.maximum(0, x)


# Define derivatives of activation functions
def activation_d(x, func):
    '''
    Derivative of activation function
    :param x: X matrix
    :param func: Activation function
                Options: sigmoid, tanh, ReLU
    '''
    if func == "sigmoid":
        z = 1 / (1 + np.exp(-x))
        return z * (1 - z)
    elif func == "tanh":
        return 1 - np.square(np.tanh(x))
    elif func == "relu":
        x[x <= 0] = 0
        x[x > 0] = 1
        return x


def fit(x, y, n_h=2, activ_func="tanh", precision=1e-5, learning_rate=0.01, print_step=100):
    '''
    Fitting function for 2 layer neural network.
    :param x: X feature matrix
    :param y: Y output classifiers
    :param n_h: Number of neurons in the hidden layer
    :param activ_func: Activation function. sigmoid, tanh, ReLU
    :param precision: Precision value at which optimizer stops
    :param learning_rate: Learning rate of Optimizer
    :param print_step: Option to print costs after print_step steps
    :return: Returns Youtput, costs and number of steps in calculations
    '''

    # Calculate Number of samples, Number of neurons
    m = x.shape[1]  # Number of samples
    n_x = x.shape[0]  # Features in X
    n_h = n_h  # Number of neurons
    n_y = y.shape[0]  # Features in Y

    # Initialize Weights and bias for forward propagation
    w1 = np.random.randn(n_h, n_x)
    w2 = np.random.randn(n_y, n_h)
    b1 = np.zeros((n_h, 1))
    b2 = np.zeros((n_y, 1))
    a2 = np.zeros(y.shape)

    # Initialize cost function list
    costs = []
    cost1 = 0  # Used to determine stopping condition for while loop
    delta_cost = np.inf  # Initialize delta_cost
    n_step = 0  # Step number increment

    # In case you need to run a for loop, n_iter needs to be defined in the function
    # for i in range(1, n_iter):
    # Loop for optimization
    while delta_cost > precision:
        n_step += 1

        # Forward propagation
        z1 = w1 @ x + b1
        a1 = activation(z1, activ_func)
        z2 = w2 @ a1 + b2
        a2 = activation(z2, activ_func)

        # Cost function. A simple sum of square difference
        cost = np.sum(np.square(a2 - y))
        costs.append(cost)

        # Back propagation and gradients
        dz2 = 2 * (a2 - y) * activation_d(z2, activ_func)
        dw2 = dz2 @ a1.T
        db2 = 1 / m * np.sum(dz2, axis=1, keepdims=True)
        dz1 = (w2.T @ dz2) * activation_d(z1, activ_func)
        dw1 = dz1 @ x.T
        db1 = 1 / m * np.sum(dz1, axis=1, keepdims=True)

        # Update parameters
        w1 = w1 - learning_rate * dw1
        w2 = w2 - learning_rate * dw2
        b1 = b1 - learning_rate * db1
        b2 = b2 - learning_rate * db2

        # Calculations for while loop
        delta_cost = np.abs(cost1 - cost)
        cost1 = cost

        # Option to print at some interval.
        # if n_step % print_step == 0:
        #     print("Cost at step %d is %f" % (n_step, cost))

    return a2, costs, n_step


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
    Yhat, costs, n_steps = fit(X, Y, n_h=n, activ_func='tanh', precision=1e-5, learning_rate=0.1, print_step=1000)
    # print(Yhat)  # Print calculated YHat values

    # Since Yhat gives probabilities, a probability of greater than 0.5 corresponds to 1 (high) output.
    # Yhat = (Yhat > 0.5).astype(int)

    Yhat = np.round(Yhat, 1)  # Round values to 1 decimal place.
    print('N_hidden_layers:' + str(n) + ', N_steps:' + str(n_steps) + ', Output:' + str(Yhat))
    axislabel = ['Steps', 'Cost']
    gtitle = 'Cost function'
    fname = 'Cost_NN.png'
    plotlabel = 'Hidden layer: ' + str(n)
    plot_1d(costs, figsame=False, alabel=axislabel, llabel=plotlabel, title=gtitle, filename=fname)

# Output
# Plot of cost function is saved.
# N_hidden_layers:2, N_steps:215, Output:[[-0.5  0.5  0.5 -0.5]]
# N_hidden_layers:4, N_steps:96, Output:[[-0.5  0.5  0.5 -0.5]]
# N_hidden_layers:6, N_steps:92, Output:[[-0.5  0.5  0.5 -0.5]]
# N_hidden_layers:8, N_steps:87, Output:[[-0.5  0.5  0.5 -0.5]]
# N_hidden_layers:10, N_steps:108, Output:[[-0.5  0.5  0.5 -0.5]]

# It seems that Layer 8 converges fastest.
# Since seed number is fixed, output will remain same at each run.

# A simple 2 layer neural network can solve XOR problem easily.
# We will try for more hidden layers in next update.
