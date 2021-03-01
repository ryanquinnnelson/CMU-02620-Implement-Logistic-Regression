import numpy as np
import packages.LogisticRegression.gradient_descent as gd


# ?? strategy to set weights
def _set_weights(cols):
    return np.ones(cols)


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
        # set initial weights
        weights = _set_weights(X.shape[1] + 1)  # add new column to accommodate w_0

        # perform gradient descent until convergence
        weights = gd.gradient_descent(X, y, self.eta, self.epsilon, weights)
        self.weights = weights

        return self
