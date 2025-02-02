"""
neuralnet.py

What you need to do:
- Complete random_init
- Implement SoftMaxCrossEntropy methods
- Implement Sigmoid methods
- Implement Linear methods
- Implement NN methods

It is ***strongly advised*** that you finish the Written portion -- at the
very least, problems 1 and 2 -- before you attempt this programming 
assignment; the code for forward and backprop relies heavily on the formulas
you derive in those problems.

Sidenote: We annotate our functions and methods with type hints, which
specify the types of the parameters and the returns. For more on the type
hinting syntax, see https://docs.python.org/3/library/typing.html.
"""

import numpy as np
import argparse
from typing import Callable, List, Tuple
import matplotlib.pyplot as plt

# This takes care of command line argument parsing for you!
# To access a specific argument, simply access args.<argument name>.
parser = argparse.ArgumentParser()
parser.add_argument('train_input', type=str,
                    help='path to training input .csv file')
parser.add_argument('validation_input', type=str,
                    help='path to validation input .csv file')
parser.add_argument('train_out', type=str,
                    help='path to store prediction on training data')
parser.add_argument('validation_out', type=str,
                    help='path to store prediction on validation data')
parser.add_argument('metrics_out', type=str,
                    help='path to store training and testing metrics')
parser.add_argument('num_epoch', type=int,
                    help='number of training epochs')
parser.add_argument('hidden_units', type=int,
                    help='number of hidden units')
parser.add_argument('init_flag', type=int, choices=[1, 2],
                    help='weight initialization functions, 1: random')
parser.add_argument('learning_rate', type=float,
                    help='learning rate')
parser.add_argument("--plotfig", type=int, default=0,
                    help='show figures.')
parser.add_argument("--print_weight", type=int, default=0,
                    help='print weights.')
parser.add_argument("--plot2hiddenlayer", type=int, default=0,
                    help='print weights.')


def args2data(args) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
str, str, str, int, int, int, float]:
    """
    DO NOT modify this function.

    Parse command line arguments, create train/test data and labels.
    :return:
    X_tr: train data *without label column and without bias folded in
        (numpy array)
    y_tr: train label (numpy array)
    X_te: test data *without label column and without bias folded in*
        (numpy array)
    y_te: test label (numpy array)
    out_tr: file for predicted output for train data (file)
    out_te: file for predicted output for test data (file)
    out_metrics: file for output for train and test error (file)
    n_epochs: number of train epochs
    n_hid: number of hidden units
    init_flag: weight initialize flag -- 1 means random, 2 means zero
    lr: learning rate
    """
    # Get data from arguments
    out_tr = args.train_out
    out_te = args.validation_out
    out_metrics = args.metrics_out
    n_epochs = args.num_epoch
    n_hid = args.hidden_units
    init_flag = args.init_flag
    lr = args.learning_rate
    plotfig = args.plotfig
    print_weight = args.print_weight
    plot2hiddenlayer = args.plot2hiddenlayer

    X_tr = np.loadtxt(args.train_input, delimiter=',')
    y_tr = X_tr[:, 0].astype(int)
    X_tr = X_tr[:, 1:]  # cut off label column

    X_te = np.loadtxt(args.validation_input, delimiter=',')
    y_te = X_te[:, 0].astype(int)
    X_te = X_te[:, 1:]  # cut off label column

    return (X_tr, y_tr, X_te, y_te, out_tr, out_te, out_metrics,
            n_epochs, n_hid, init_flag, lr, plotfig, print_weight, plot2hiddenlayer)


def shuffle(X, y, epoch):
    """
    DO NOT modify this function.

    Permute the training data for SGD.
    :param X: The original input data in the order of the file.
    :param y: The original labels in the order of the file.
    :param epoch: The epoch number (0-indexed).
    :return: Permuted X and y training data for the epoch.
    """
    np.random.seed(epoch)
    N = len(y)
    ordering = np.random.permutation(N)
    return X[ordering], y[ordering]


def zero_init(shape):
    """
    DO NOT modify this function.

    ZERO Initialization: All weights are initialized to 0.

    :param shape: list or tuple of shapes
    :return: initialized weights
    """
    return np.zeros(shape=shape)


def random_init(shape):
    """

    RANDOM Initialization: The weights are initialized randomly from a uniform
        distribution from -0.1 to 0.1.

    :param shape: list or tuple of shapes
    :return: initialized weights
    """
    M, D = shape
    np.random.seed(M * D)  # Don't change this line!

    # TODO: create the random matrix here!
    # Hint: numpy might have some useful function for this
    # Hint: make sure you have the right distribution
    return np.random.uniform(-0.1, 0.1, size=shape)


class SoftMaxCrossEntropy:

    def _softmax(self, z: np.ndarray) -> np.ndarray:
        """
        Implement softmax function.
        :param z: input logits of shape (num_classes,)
        :return: softmax output of shape (num_classes,)
        """
        # TODO: implement
        exp = np.exp(z)
        sum_exp = np.sum(exp)
        return exp/sum_exp

    def _cross_entropy(self, y: int, y_hat: np.ndarray) -> float:
        """
        Compute cross entropy loss.
        :param y: integer class label
        :param y_hat: prediction with shape (num_classes,)
        :return: cross entropy loss
        """
        # TODO: implement
        return -np.log(y_hat[y])

    def forward(self, z: np.ndarray, y: int) -> Tuple[np.ndarray, float]:
        """
        Compute softmax and cross entropy loss.
        :param z: input logits of shape (num_classes,)
        :param y: integer class label
        :return:
            y: predictions from softmax as an np.ndarray
            loss: cross entropy loss
        """
        # TODO: Call your implementations of _softmax and _cross_entropy here
        yh = self._softmax(z)
        loss = self._cross_entropy(y, yh)
        return yh, loss

    def backward(self, y: int, y_hat: np.ndarray) -> np.ndarray:
        """
        Compute gradient of loss w.r.t. ** softmax input **.
        Note that here instead of calculating the gradient w.r.t. the softmax
        probabilities, we are directly computing gradient w.r.t. the softmax
        input.

        Try deriving the gradient yourself (see Question 1.2(b) on the written),
        and you'll see why we want to calculate this in a single step.

        :param y: integer class label
        :param y_hat: predicted softmax probability with shape (num_classes,)
        :return: gradient with shape (num_classes,)
        """
        # TODO: implement using the formula you derived in the written
        y_one_hot = np.zeros(y_hat.shape)
        y_one_hot[y] = 1
        return y_hat - y_one_hot


class Sigmoid:
    def __init__(self):
        """
        Initialize state for sigmoid activation layer
        """
        # TODO Initialize any additional values you may need to store for the
        #  backward pass here
        self._forward_result = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Take sigmoid of input x.
        :param x: Input to activation function (i.e. output of the previous 
                  linear layer), with shape (output_size,)
        :return: Output of sigmoid activation function with shape
            (output_size,)
        """
        # TODO: perform forward pass and save any values you may need for
        #  the backward pass
        result = 1/(1+np.exp(-x))
        self._forward_result = result
        return result


    def backward(self, dz: np.ndarray) -> np.ndarray:
        """
        :param dz: partial derivative of loss with respect to output of
            sigmoid activation
        :return: partial derivative of loss with respect to input of
            sigmoid activation
        """
        # TODO: implement
        return dz*self._forward_result*(1-self._forward_result)

# This refers to a function type that takes in a tuple of 2 integers (row, col)
# and returns a numpy array (which should have the specified dimensions).
INIT_FN_TYPE = Callable[[Tuple[int, int]], np.ndarray]


class Linear:
    def __init__(self, input_size: int, output_size: int,
                 weight_init_fn: INIT_FN_TYPE, learning_rate: float):
        """
        :param input_size: number of units in the input of the layer 
                           *not including* the folded bias
        :param output_size: number of units in the output of the layer
        :param weight_init_fn: function that creates and initializes weight 
                               matrices for layer. This function takes in a 
                               tuple (row, col) and returns a matrix with
                               shape row x col.
        :param learning_rate: learning rate for SGD training updates
        """
        # Initialize learning rate for SGD
        self.lr = learning_rate

        # TODO: Initialize weight matrix for this layer - since we are
        #  folding the bias into the weight matrix, be careful about the
        #  shape you pass in.
        #  To be consistent with the formulas you derived in the written and
        #  in order for the unit tests to work correctly,
        #  the first dimension should be the output size
        weight_shape     = (output_size, input_size+1)
        self.w           = weight_init_fn(weight_shape)

        # TODO: set the bias terms to zero
        self.w[:, 0] = 0

        # TODO: Initialize matrix to store gradient with respect to weights
        self.dw = np.zeros(weight_shape, dtype=np.float32)

        # TODO: Initialize any additional values you may need to store for the
        #  backward pass here
        #  without bias term
        self.input_x = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        :param x: Input to linear layer with shape (input_size,)
                  where input_size *does not include* the folded bias.
                  In other words, the input does not contain the bias column 
                  and you will need to add it in yourself in this method.
                  Since we train on 1 example at a time, batch_size should be 1
                  at training.
        :return: output z of linear layer with shape (output_size,)

        HINT: You may want to cache some of the values you compute in this
        function. Inspect your expressions for backprop to see which values
        should be cached.
        """
        # TODO: perform forward pass and save any values you may need for
        #  the backward pass
        x_reshape_2d = x.reshape((x.shape[0], 1))
        x_with_bias = np.vstack([np.ones((1, 1)), x_reshape_2d])
        self.input_x = x_with_bias.copy()
        return self.w@x_with_bias

    def backward(self, dz: np.ndarray) -> np.ndarray:
        """
        :param dz: partial derivative of loss with respect to output z
            of linear
        :return: dx, partial derivative of loss with respect to input x
            of linear
        
        Note that this function should set self.dw
            (gradient of loss with respect to weights)
            but not directly modify self.w; NN.step() is responsible for
            updating the weights.

        HINT: You may want to use some of the values you previously cached in 
        your forward() method.
        """
        # TODO: implement
        dx = (self.w.T @ dz).reshape(self.input_x.shape)[1:, :]
        self.dw = dz * self.input_x.T
        return dx


    def step(self) -> None:
        """
        Apply SGD update to weights using self.dw, which should have been 
        set in NN.backward().
        """
        # TODO: implement
        self.w  -=  self.lr * self.dw


class NN:
    def __init__(self, input_size: int, hidden_size: int, output_size: int,
                 weight_init_fn: INIT_FN_TYPE, learning_rate: float):
        """
        Initalize neural network (NN) class. Note that this class is composed
        of the layer objects (Linear, Sigmoid) defined above.

        :param input_size: number of units in input to network
        :param hidden_size: number of units in the hidden layer of the network
        :param output_size: number of units in output of the network - this
                            should be equal to the number of classes
        :param weight_init_fn: function that creates and initializes weight 
                               matrices for layer. This function takes in a 
                               tuple (row, col) and returns a matrix with 
                               shape row x col.
        :param learning_rate: learning rate for SGD training updates
        """
        self.weight_init_fn = weight_init_fn
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # TODO: initialize modules (see section 9.1.2 of the writeup)
        #  Hint: use the classes you've implemented above!
        self.Linear1  = Linear(input_size, hidden_size, weight_init_fn, learning_rate)
        self.Sigmoid1 = Sigmoid()
        self.Linear2  = Linear(hidden_size, output_size, weight_init_fn, learning_rate)
        self.SoftMax  = SoftMaxCrossEntropy()
        self.linear1 = self.Linear1
        self.linear2 = self.Linear2

        self.Linear1_2hiddenlayer  = Linear(input_size, hidden_size, weight_init_fn, learning_rate)
        self.Sigmoid1_2hiddenlayer = Sigmoid()
        self.Linear2_2hiddenlayer  = Linear(hidden_size, hidden_size, weight_init_fn, learning_rate)
        self.Sigmoid2_2hiddenlayer = Sigmoid()
        self.Linear3_2hiddenlayer  = Linear(hidden_size, output_size, weight_init_fn, learning_rate)
        self.SoftMax_2hiddenlayer  = SoftMaxCrossEntropy()



    def forward_2hiddenlayer(self, x: np.ndarray, y: int) -> Tuple[np.ndarray, float]:
        """
        Neural network forward computation. 
        Follow the pseudocode!
        :param x: input data point *without the bias folded in*
        :param y: prediction with shape (num_classes,)
        :return:
            y_hat: output prediction with shape (num_classes,). This should be
                a valid probability distribution over the classes.
            loss: the cross_entropy loss for a given example
        """
        # TODO: call forward pass for each layer
        a1 = self.Linear1_2hiddenlayer.forward(x)
        z1 = self.Sigmoid1_2hiddenlayer.forward(a1)

        a2 = self.Linear2_2hiddenlayer.forward(z1)
        z2 = self.Sigmoid2_2hiddenlayer.forward(a2)

        b = self.Linear3_2hiddenlayer.forward(z2)
        y_hat, J = self.SoftMax_2hiddenlayer.forward(b, y)
        return y_hat, J

    def forward(self, x: np.ndarray, y: int) -> Tuple[np.ndarray, float]:
        """
        Neural network forward computation. 
        Follow the pseudocode!
        :param x: input data point *without the bias folded in*
        :param y: prediction with shape (num_classes,)
        :return:
            y_hat: output prediction with shape (num_classes,). This should be
                a valid probability distribution over the classes.
            loss: the cross_entropy loss for a given example
        """
        # TODO: call forward pass for each layer
        a = self.Linear1.forward(x)
        z = self.Sigmoid1.forward(a)
        b = self.Linear2.forward(z)
        y_hat, J = self.SoftMax.forward(b, y)
        return y_hat, J

    def backward_2hiddenlayer(self, y: int, y_hat: np.ndarray) -> None:
        """
        Neural network backward computation.
        Follow the pseudocode!
        :param y: label (a number or an array containing a single element)
        :param y_hat: prediction with shape (num_classes,)
        """
        # TODO: call backward pass for each layer
        gb = self.SoftMax_2hiddenlayer.backward(y, y_hat)
        gz2 = self.Linear3_2hiddenlayer.backward(gb)

        ga2 = self.Sigmoid2_2hiddenlayer.backward(gz2)
        gz1 = self.Linear2_2hiddenlayer.backward(ga2)

        ga1 = self.Sigmoid1_2hiddenlayer.backward(gz1)
        gx = self.Linear1_2hiddenlayer.backward(ga1)


    def backward(self, y: int, y_hat: np.ndarray) -> None:
        """
        Neural network backward computation.
        Follow the pseudocode!
        :param y: label (a number or an array containing a single element)
        :param y_hat: prediction with shape (num_classes,)
        """
        # TODO: call backward pass for each layer
        gb = self.SoftMax.backward(y, y_hat)
        gz = self.Linear2.backward(gb)
        ga = self.Sigmoid1.backward(gz)
        gx = self.Linear1.backward(ga)

    def step_2hiddenlayer(self):
        """
        Apply SGD update to weights.
        """
        # TODO: call step for each relevant layer
        self.Linear1_2hiddenlayer.step()
        self.Linear2_2hiddenlayer.step()
        self.Linear3_2hiddenlayer.step()

    def step(self):
        """
        Apply SGD update to weights.
        """
        # TODO: call step for each relevant layer
        self.Linear1.step()
        self.Linear2.step()

    def compute_loss_2hiddenlayer(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute nn's average (cross entropy) loss over the dataset (X, y)
        :param X: Input dataset of shape (num_points, input_size)
        :param y: Input labels of shape (num_points,)
        :return: Mean cross entropy loss
        """
        # TODO: compute loss over the entire dataset
        #  Hint: reuse your forward function
        J = []
        for i in range(y.shape[0]):
            y_hat, Ji = self.forward_2hiddenlayer(X[i], y[i])
            J.append(Ji)
        return np.mean(J)


    def compute_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute nn's average (cross entropy) loss over the dataset (X, y)
        :param X: Input dataset of shape (num_points, input_size)
        :param y: Input labels of shape (num_points,)
        :return: Mean cross entropy loss
        """
        # TODO: compute loss over the entire dataset
        #  Hint: reuse your forward function
        J = []
        for i in range(y.shape[0]):
            y_hat, Ji = self.forward(X[i], y[i])
            J.append(Ji)
        return np.mean(J)

    def train_2hiddenlayer(self, X_tr: np.ndarray, y_tr: np.ndarray,
              X_test: np.ndarray, y_test: np.ndarray,
              n_epochs: int, print_weight=False) -> Tuple[List[float], List[float]]:
        """
        Train the network using SGD for some epochs.
        :param X_tr: train data
        :param y_tr: train label
        :param X_test: train data
        :param y_test: train label
        :param n_epochs: number of epochs to train for
        :return:
            train_losses: Training losses *after* each training epoch
            test_losses: Test losses *after* each training epoch
        """
        # TODO: train network
        w_shape0 = (self.hidden_size, self.input_size)
        w_shape1 = (self.output_size, self.hidden_size)
        #self.weight_init_fn(w_shape0)
        #self.weight_init_fn(w_shape1)
        J_train = []
        J_test = []

        for epoch_num in range(n_epochs):
            X_shuffled, y_shuffled = shuffle(X_tr, y_tr, epoch_num)

            for i, X_input in enumerate(X_shuffled):
                y_input = y_shuffled[i]
                y_hat, Ji = self.forward_2hiddenlayer(X_input, y_input)
                self.backward_2hiddenlayer(y_input, y_hat)
                self.step_2hiddenlayer()
            J_train_i = self.compute_loss_2hiddenlayer(X_tr, y_tr)
            J_test_i = self.compute_loss_2hiddenlayer(X_test, y_test)

            J_train.append(J_train_i)
            J_test.append(J_test_i)

        return J_train, J_test

    def train(self, X_tr: np.ndarray, y_tr: np.ndarray,
              X_test: np.ndarray, y_test: np.ndarray,
              n_epochs: int, print_weight=False) -> Tuple[List[float], List[float]]:
        """
        Train the network using SGD for some epochs.
        :param X_tr: train data
        :param y_tr: train label
        :param X_test: train data
        :param y_test: train label
        :param n_epochs: number of epochs to train for
        :return:
            train_losses: Training losses *after* each training epoch
            test_losses: Test losses *after* each training epoch
        """
        # TODO: train network
        w_shape0 = (self.hidden_size, self.input_size)
        w_shape1 = (self.output_size, self.hidden_size)
        #self.weight_init_fn(w_shape0)
        #self.weight_init_fn(w_shape1)
        J_train = []
        J_test = []

        for epoch_num in range(n_epochs):
            if print_weight:
                if epoch_num == 0 or epoch_num == 1 or epoch_num == 2 or epoch_num == 3:
                    np.set_printoptions(linewidth = 400)
                    print(f"epoch_num = {epoch_num}")
                    print(f"alpha.shape = {self.Linear1.w.shape}")
                    print(f"alpha = {self.Linear1.w}")
                    print(f"beta.shape = {self.Linear2.w.shape}")
                    print(f"beta = {self.Linear2.w}")

            X_shuffled, y_shuffled = shuffle(X_tr, y_tr, epoch_num)

            for i, X_input in enumerate(X_shuffled):
                y_input = y_shuffled[i]
                y_hat, Ji = self.forward(X_input, y_input)
                self.backward(y_input, y_hat)
                self.step()
            J_train_i = self.compute_loss(X_tr, y_tr)
            J_test_i = self.compute_loss(X_test, y_test)

            J_train.append(J_train_i)
            J_test.append(J_test_i)

        return J_train, J_test

    def test(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Compute the label and error rate.
        :param X: input data
        :param y: label
        :return:
            labels: predicted labels
            error_rate: prediction error rate
        """
        # TODO: make predictions and compute error
        err_cnt = 0
        labels = np.zeros(y.shape, dtype=int)
        for i in range(y.shape[0]):
            y_hat, Ji = self.forward(X[i], y[i])
            y_val = np.argmax(y_hat)
            if y_val != y[i]:
                err_cnt += 1
            labels[i] = y_val

        return labels, err_cnt/float(y.shape[0])

def runPrintWeights(X_tr, y_tr, X_val, y_val, hidden_units, n_epochs, lr):
    weight_init_fn_plot = zero_init

    nnplot = NN(
        input_size=X_tr.shape[-1],
        hidden_size=hidden_units,
        output_size=len(labels),
        weight_init_fn=weight_init_fn_plot,
        learning_rate=lr
    )

    train_losses, val_losses = nnplot.train(X_tr, y_tr, X_val, y_val, n_epochs, print_weight=True)

def runPlotLearningRates_2hiddenLayer(X_tr, y_tr, X_val, y_val, hidden_units, n_epochs, lr):
    weight_init_fn_plot = random_init
    train_loss_plot = []
    val_loss_plot = []

    nnplot = NN(
        input_size=X_tr.shape[-1],
        hidden_size=hidden_units,
        output_size=len(labels),
        weight_init_fn=weight_init_fn_plot,
        learning_rate=lr
    )

    train_losses, val_losses = nnplot.train(X_tr, y_tr, X_val, y_val, n_epochs)
    train_losses_2hiddenlayer, val_losses_2hiddenlayer = nnplot.train_2hiddenlayer(X_tr, y_tr, X_val, y_val, n_epochs)

    # Plot 1-hidden-layer model
    plt.plot(np.arange(n_epochs), train_losses, label="1 Hidden Layer - Training Loss", color="blue", linestyle='-')
    plt.plot(np.arange(n_epochs), val_losses, label="1 Hidden Layer - Validation Loss", color="blue", linestyle='--')

    # Plot 2-hidden-layer model
    plt.plot(np.arange(n_epochs), train_losses_2hiddenlayer, label="2 Hidden Layers - Training Loss", color="orange", linestyle='-')
    plt.plot(np.arange(n_epochs), val_losses_2hiddenlayer, label="2 Hidden Layers - Validation Loss", color="orange", linestyle='--')

    # Labels and title
    plt.xlabel("Epoch Number")
    plt.ylabel("Average Cross-Entropy Loss")
    plt.title("1 vs 2 Hidden Layers on Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.show()    


def runPlotLearningRates(X_tr, y_tr, X_val, y_val, hidden_units, n_epochs, lr):
    weight_init_fn_plot = random_init
    train_loss_plot = []
    val_loss_plot = []

    nnplot = NN(
        input_size=X_tr.shape[-1],
        hidden_size=hidden_units,
        output_size=len(labels),
        weight_init_fn=weight_init_fn_plot,
        learning_rate=lr
    )

    train_losses, val_losses = nnplot.train(X_tr, y_tr, X_val, y_val, n_epochs)

    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(n_epochs), train_losses, label="Training Loss")
    plt.plot(np.arange(n_epochs), val_losses, label="Validation Loss")
    plt.xlabel("Number of Epochs")
    plt.ylabel("Average Cross-Entropy Loss")
    plt.title("Training and Validation Loss vs Number of Epochs")
    plt.legend()
    plt.show()

def runPlotHiddenUnits(X_tr, y_tr, X_val, y_val, n_epochs):
    weight_init_fn_plot = random_init
    lr = 0.001
    train_loss_plot = []
    val_loss_plot = []
    hidden_units_list = [5, 20, 50, 100, 200]


    for n_hid in hidden_units_list:
        nnplot = NN(
            input_size=X_tr.shape[-1],
            hidden_size=n_hid,
            output_size=len(labels),
            weight_init_fn=weight_init_fn_plot,
            learning_rate=lr
        )
        train_losses, val_losses = nnplot.train(X_tr, y_tr, X_val, y_val, n_epochs)
        train_loss_plot.append(train_losses[-1])
        val_loss_plot.append(val_losses[-1])

    plt.figure(figsize=(10, 6))
    plt.plot(hidden_units_list, train_loss_plot, label="Training Loss")
    plt.plot(hidden_units_list, val_loss_plot, label="Validation Loss")
    plt.xlabel("Number of Hidden Units")
    plt.ylabel("Average Cross-Entropy Loss")
    plt.title("Training and Validation Loss vs Number of Hidden Units")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    args = parser.parse_args()
    # Note: You can access arguments like learning rate with args.learning_rate
    # Generally, you can access each argument using the name that was passed 
    # into parser.add_argument() above (see lines 24-44).

    # Define our labels
    labels = ["a", "e", "g", "i", "l", "n", "o", "r", "t", "u"]

    # Call args2data to get all data + argument values
    # See the docstring of `args2data` for an explanation of 
    # what is being returned.
    (X_tr, y_tr, X_test, y_test, out_tr, out_te, out_metrics,
     n_epochs, n_hid, init_flag, lr, plotfig, print_weight, plot2hiddenlayer) = args2data(args)

    nn = NN(
        input_size=X_tr.shape[-1],
        hidden_size=n_hid,
        output_size=len(labels),
        weight_init_fn=zero_init if init_flag == 2 else random_init,
        learning_rate=lr
    )

    # train model
    # (this line of code is already written for you)
    train_losses, test_losses = nn.train(X_tr, y_tr, X_test, y_test, n_epochs)

    # test model and get predicted labels and errors 
    # (this line of code is written for you)
    train_labels, train_error_rate = nn.test(X_tr, y_tr)
    test_labels, test_error_rate = nn.test(X_test, y_test)

    # Write predicted label and error into file
    # Note that this assumes train_losses and test_losses are lists of floats
    # containing the per-epoch loss values.
    with open(out_tr, "w") as f:
        for label in train_labels:
            f.write(str(label) + "\n")
    with open(out_te, "w") as f:
        for label in test_labels:
            f.write(str(label) + "\n")
    with open(out_metrics, "w") as f:
        for i in range(len(train_losses)):
            cur_epoch = i + 1
            cur_tr_loss = train_losses[i]
            cur_te_loss = test_losses[i]
            f.write("epoch={} crossentropy(train): {}\n".format(
                cur_epoch, cur_tr_loss))
            f.write("epoch={} crossentropy(validation): {}\n".format(
                cur_epoch, cur_te_loss))
        f.write("error(train): {}\n".format(train_error_rate))
        f.write("error(validation): {}\n".format(test_error_rate))

    if plotfig:
        runPlotHiddenUnits(X_tr, y_tr, X_test, y_test, n_epochs=100)
        runPlotLearningRates(X_tr, y_tr, X_test, y_test, 50, 100, 0.03)
        runPlotLearningRates(X_tr, y_tr, X_test, y_test, 50, 100, 0.003)
        runPlotLearningRates(X_tr, y_tr, X_test, y_test, 50, 100, 0.0003)
    if print_weight:
        runPrintWeights(X_tr, y_tr, X_test, y_test, n_hid, 100, lr)

    if plot2hiddenlayer:
        runPlotLearningRates_2hiddenLayer(X_tr, y_tr, X_test, y_test, 50, 100, 0.003)
