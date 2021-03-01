"""
Implementation of gradient descent for Logistic Regression, as defined
by Machine Learning (Mitchell)
Assumes "imaginary" X_0 = 1 for all samples has been added to the matrix.
"""

import numpy as np

"""
Note 1 - Explanation of how _calc_gradient() works.

    d l(W)/d w_i = sum_l X_i^l ( Y^l - P(Y^l = 1 |X^l,W))

X                y_err
|x11|x12|x13|    | y1 |
|X21|x22|x23|    | y2 |
|x31|x32|x33|    | y3 |

For each of L samples, the formula has us multiply X_i by y_err:

d l(W) / d w_1 = sum_l ( X_1 * y_err)


Calculating the partial for the ith feature is the same as taking the dot product of the ith column of X and y_err.

                 X_1         y_err
                 |x11|       | y1 | 
d l(W) / d w_1 = |x21|  dot  | y2 |  = (x11*y1) + (x21*y2) + (x31*y3)
                 |x31|       | y3 |
                 

                 X_2         y_err
                 |x12|       | y1 | 
d l(W) / d w_2 = |x22|  dot  | y2 |  = (x12*y1) + (x22*y2) + (x32*y3)
                 |x32|       | y3 |


Instead of calculating partials one by one, we can perform matrix multiplication to do all at the same time. 
However, we need to transpose X first, because we want to perform column-wise multiplications.

X              y_err
|x11|x12|x13|  | y1 | 
|x21|x22|x23|  | y2 |
|x31|x32|x33|  | y3 |

(X)(y_err)
| (x11*y1) + (x12*y2) + (x13*y3) |
| (x21*y1) + (x22*y2) + (x23*y3) |
| (x31*y1) + (x32*y2) + (x33*y3) |

X_transpose    y_err
|x11|x21|x31|  | y1 | 
|x12|x22|x32|  | y2 |
|x13|x23|x33|  | y3 |

W = (X_transpose)(y_err)
| w1 |    | (x11*y1) + (x21*y2) + (x31*y3) |
| w2 | =  | (x12*y1) + (x22*y2) + (x32*y3) |
| w3 |    | (x13*y1) + (x23*y2) + (x33*y3) |
"""

"""
Note 2 - Explanation of how _get_y_predictions() works
Consider L x n matrix X and n x 1 vector w, where L is the number of samples and n is the number of features.

X                w
|x11|x12|x13|    | w0 |
|X21|x22|x23|    | w1 |
|x31|x32|x33|    | w2 |

For each sample, we need to calculate a prediction. According to Mitchell, we do this by calculating P(Y=1|X,W).

P(Y=1|X,W) = exp(w_0 + SUM_i (w_i X_i) / 1 + exp(w_0 + SUM_i (w_i X_i)
where i is the ith feature

For a single sample, the sum can be calculated as a dot product. Note that in this implementation, we assume matrix X 
has been augmented so that it's first column is 1's to accommodate w_0:

For l=1: w_0 + SUM_i (w_i X_i) = X * w = (x11 * w1) + (x12 * w2) + (x13 * w3)
For l=2: w_0 + SUM_i (w_i X_i) = X * w = (x21 * w1) + (x22 * w2) + (x23 * w3)

However, we can efficiently calculate this sum for all samples at the same time, using matrix multiplication:

     | y_pred_1 |   | (x11 * w1) + (x12 * w2) + (x13 * w3) |
Xw = | y_pred_2 |   | (x21 * w1) + (x22 * w2) + (x23 * w3) |
     | y_pred_3 |   | (x31 * w1) + (x32 * w2) + (x33 * w3) |

Then we can perform exponentiation and addition on the resulting vector.
"""


def _calc_log_likelihood(X, y_true, w):
    """

    :param X:
    :param y_true:
    :param w:
    :return:
    """
    # left half
    XW = np.matmul(X, w)
    YXW = y_true + XW  # all L samples can be summed in parallel

    # right half
    num_rows = X.shape[0]
    ones = np.ones(num_rows)
    inner = ones + np.exp(XW)
    ln_XW = np.log(inner)

    return YXW - ln_XW


# tested
def _get_y_prediction(x, w):
    """
    Obtains predicted label for one sample. See Note 2 for explanation of function logic.

    :param x: n x 1 vector
    :param w: n x 1 vector
    :return:  scalar
    """

    a = np.dot(x, w)
    b = np.exp(a)
    c = 1 + np.exp(a)

    return b / c


# tested
def _get_y_predictions(X, w):
    """
    Obtains predicted labels for all L samples.  See Note 2 for explanation of function logic.

    :param X: L x n matrix
    :param w: n x 1 vector
    :return:  L x 1 vector
    """
    num_rows = X.shape[0]

    Xw = np.matmul(X, w)  # all samples can be summed in parallel
    top = np.exp(Xw)
    ones = np.ones(num_rows)
    bottom = ones + top
    return top / bottom


def _calc_gradient(X, y_true, y_pred):
    """
    Calculates the gradient. See Note 1 for explanation of function logic.

    :param X: L x n matrix, where L is the number of samples and n is the number of features
    :param y_true: L x 1 vector
    :param y_pred: L x 1 vector
    :return: Gradient in the form of an L x 1 vector
    """
    y_err = y_true - y_pred
    return np.matmul(X.T, y_err)


# tested
def _update_weights(w, eta, gradient):
    """
    Updates regression coefficients using the following formula:
    W <- W + (eta * gradient)

    :param w: n x 1 vector
    :param eta:
    :param gradient: n x 1 vector
    :return:
    """
    change = eta * gradient
    return w + change


# ?? diff is a vector, how to compare with epsilon?
def gradient_descent(X, y_true, w, eta, epsilon):
    """
    Performs gradient descent to derive optimal regression coefficients.

    :param X:
    :param y_true:
    :param eta:
    :param epsilon:
    :param w:
    :return:
    """
    # set initial weights
    weights = w

    # calculate original log likelihood
    prev_log_likelihood = _calc_log_likelihood(X, y_true, weights)

    # perform gradient descent
    count = 0
    diff = np.Inf
    while diff > epsilon:

        count += 1
        if count > 100000:
            break  # stop descending

        # update weights
        y_pred = _get_y_predictions(X, weights)
        gradient = _calc_gradient(X, y_true, y_pred)
        weights = _update_weights(weights, eta, gradient)

        # calculate difference
        log_likelihood = _calc_log_likelihood(X, y_true, weights)
        diff = prev_log_likelihood - log_likelihood

        # save log likelihood for next round
        prev_log_likelihood = log_likelihood

    print('count of rounds', count)
    return weights
