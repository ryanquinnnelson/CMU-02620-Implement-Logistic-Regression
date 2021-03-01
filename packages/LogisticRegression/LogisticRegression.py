import numpy as np
import packages.LogisticRegression.gradient_descent as gd


def _set_weights(cols):
    """
    Creates an array of weights of length equal to cols, with each element set to 0.
    :param cols:
    :return:
    """
    return np.zeros(cols)


def _add_x0(X):
    """
    Adds a column to the left of matrix X with each element set to 1.
    :param X:
    :param rows:
    :return:
    """
    rows = X.shape[0]
    ones = np.ones(rows)
    return np.insert(X, 0, ones, axis=1)


class LogisticRegression:

    def __init__(self, eta, epsilon):
        """

        :param eta: learning rate
        :param epsilon: convergence threshold
        """
        self.eta = eta
        self.epsilon = epsilon
        self.weights = None

    def fit(self, X, y):
        """

        :param X:
        :param y:
        :return:
        """
        # append imaginary column X_0=1 to accommodate w_0
        X_aug = _add_x0(X)

        # set initial weights
        cols = X_aug.shape[1]
        weights = _set_weights(cols)

        # perform gradient descent until convergence
        weights = gd.gradient_descent(X_aug, y, self.eta, self.epsilon, weights)
        self.weights = weights

        return self
