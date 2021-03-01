import numpy as np
import packages.LogisticRegression.gradient_descent as gd


# tested
def _set_weights(X):
    """
    Creates an array of weights with each element set to 0.
    :param cols:
    :return:
    """
    cols = X.shape[1]
    return np.zeros(cols)


# tested
# does not work properly with a single sample
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

    # tested
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

        :param X: L x n matrix, where L is the number of samples and n is the number of features
        :param y: L x 1 matrix
        :return:
        """
        # append imaginary column X_0=1 to accommodate w_0
        X_aug = _add_x0(X)

        # set initial weights
        weights = _set_weights(X_aug)

        # perform gradient descent until convergence
        weights = gd.gradient_descent(X_aug, y, weights, self.eta, self.epsilon)
        self.weights = weights

        return self

    # tested
    def predict(self, X):
        """
        Returns predicted label for each sample.
        :param X: L x n matrix, where L is the number of samples and n is the number of features
        :return: L x 1 vector
        """
        # append imaginary column X_0=1 to accommodate w_0
        X_aug = _add_x0(X)

        y_pred = gd.get_y_predictions(X_aug, self.weights)
        return np.round(y_pred)

    def predict_proba(self, X):
        """
        Returns calculated probabilities Y=1 for each sample.
        :param X: L x n matrix, where L is the number of samples and n is the number of features
        :return: L x 1 vector
        """
        # append imaginary column X_0=1 to accommodate w_0
        X_aug = _add_x0(X)

        y_pred = gd.get_y_predictions(X_aug, self.weights)
        return y_pred
