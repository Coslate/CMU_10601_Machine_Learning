import numpy as np
import argparse
import csv
from numpy.typing import NDArray
from typing import Tuple, Any

def load_tsv_dataset(file):
    """
    Read formatted*.tsv file into labels and features.

    Parameters:
        file (str): File path to the formatted*.tsv file

    Returns:
        X: (N, D) ndarray containing features
        y: (N,) ndarray containing labels
    """
    y = []
    X = []
    with open(file, encoding='utf-8') as f:
        read_file = csv.reader(f, delimiter='\t')
        for row in read_file:
            label, feature = row[0], row[1:]
            y.append(np.float64(label))
            X.append(np.float64(feature))
    return np.array(X), np.array(y)


def sigmoid(x : np.ndarray):
    """
    Implementation of the sigmoid function.

    Parameters:
        x (np.ndarray): Input np.ndarray.

    Returns:
        An np.ndarray after applying the sigmoid function element-wise to the
        input.
    """
    e = np.exp(x)
    return e / (1 + e)


def train(
    theta : np.ndarray, # shape (D,) where D is feature dim
    X : np.ndarray,     # shape (N, D) where N is num of examples
    y : np.ndarray,     # shape (N,)
    num_epoch : int, 
    learning_rate : float
) -> None:
    # TODO: Implement `train` using vectorization
    # Folded intercept into theta: D+1 dimension
    theta_include_intercept = np.concatenate((np.zeros(1), theta))

    # Add 1 to each X for intercept term
    ones_column = np.ones((X.shape[0], 1))
    X_intercept = np.concatenate((ones_column, X), axis=1)
    
    # SGD
    for num in range(num_epoch):
        for i in range(X_intercept.shape[0]):
            dot_val = np.dot(theta_include_intercept, X_intercept[i])
            gradient = X_intercept[i]*(sigmoid(dot_val)-y[i])
            theta_include_intercept -= learning_rate*gradient

    return theta_include_intercept


def hx(x: np.ndarray, theta: np.ndarray) -> [[]]:
    dot_val = np.dot(theta, x)
    return sigmoid(dot_val) >= 0.5

def predict(
    theta : np.ndarray,
    X : np.ndarray
) -> np.ndarray:
    # TODO: Implement `predict` using vectorization
    # Add 1 to each X for intercept term
    ones_column = np.ones((X.shape[0], 1))
    X_intercept = np.concatenate((ones_column, X), axis=1)
    y_res = np.zeros((X.shape[0], 1), dtype=int)

    for i in range(X_intercept.shape[0]):
        y_res[i] = int(hx(X_intercept[i], theta))

    return y_res

def compute_error(
    y_pred : np.ndarray, 
    y : np.ndarray
) -> float:
    # TODO: Implement `compute_error` using vectorization
    len_gt_y = y.shape[0]
    len_pd_y = y_pred.shape[0]
    if len_gt_y != len_pd_y:
        print(f"Error: length of gt_y, {len_gt_y}, deos not equal to length of pd_y, {len_pd_y}.")

    error = 0
    for i in range(y.shape[0]):
        if y_pred[i] != y[i]:
            error += 1

    return error/len_gt_y

def writeOutFile(output_file: str, output_content_arr: np.ndarray) -> None :
    with open(output_file, 'w') as f:
        for data_line in output_content_arr:
            f.write(f"{data_line[0]}\n")

def writeOutMetrix(output_file: str, error_train: float, error_test: float) -> None:
    with open(output_file, 'w') as f:
        f.write(f"error(train): {error_train:.6f}\n")
        f.write(f"error(test): {error_test:.6f}\n")

if __name__ == '__main__':
    # This takes care of command line argument parsing for you!
    # To access a specific argument, simply access args.<argument name>.
    # For example, to get the learning rate, you can use `args.learning_rate`.
    parser = argparse.ArgumentParser()
    parser.add_argument("train_input", type=str, help='path to formatted training data')
    parser.add_argument("validation_input", type=str, help='path to formatted validation data')
    parser.add_argument("test_input", type=str, help='path to formatted test data')
    parser.add_argument("train_out", type=str, help='file to write train predictions to')
    parser.add_argument("test_out", type=str, help='file to write test predictions to')
    parser.add_argument("metrics_out", type=str, help='file to write metrics to')
    parser.add_argument("num_epoch", type=int, 
                        help='number of epochs of stochastic gradient descent to run')
    parser.add_argument("learning_rate", type=float,
                        help='learning rate for stochastic gradient descent')
    parser.add_argument("--is_debug", type=int, default=0,
                        help='show debug msg.')
    args = parser.parse_args()

    train_input           = args.train_input
    validation_input      = args.validation_input
    test_input            = args.test_input
    train_out             = args.train_out
    test_out              = args.test_out
    metrics_out           = args.metrics_out
    num_epoch             = args.num_epoch
    learning_rate         = args.learning_rate
    is_debug              = args.is_debug

    # Loading
    if is_debug: print(f"> load_tsv_dataset()...")
    train_X, train_y = load_tsv_dataset(train_input)
    train_X = np.array(train_X)
    train_y = np.array(train_y)
    test_X, test_y = load_tsv_dataset(test_input)
    test_X = np.array(test_X)
    test_y = np.array(test_y)

    # Training
    if is_debug: print(f"> train()...")
    theta_initial = np.zeros(train_X.shape[1])
    theta_trained = train(theta_initial, train_X, train_y, num_epoch, learning_rate)

    # Predicting
    if is_debug: print(f"> predict()...")
    predict_train_x = predict(theta_trained, train_X)
    predict_test_x = predict(theta_trained, test_X)

    # Evaluating.
    if is_debug: print(f"> compute_error())...")
    error_train = compute_error(train_y, predict_train_x)
    error_test = compute_error(test_y, predict_test_x)

    # WriteOut.
    if is_debug: print(f"> writeOutFile()...")
    writeOutFile(train_out, predict_train_x)
    writeOutFile(test_out, predict_test_x)
    writeOutMetrix(metrics_out, error_train, error_test)
