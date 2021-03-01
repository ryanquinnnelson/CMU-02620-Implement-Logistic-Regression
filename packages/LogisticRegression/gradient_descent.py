import numpy as np

"""
Note 1 - Explanation of how _calc_gradient() works.

X                y_err
| 5 | 1 | 1 |    | 0 |
| 1 | 1 | 1 |    |-1 |
| 1 | 2 | 3 |    | 1 |

Calculating the partial for the ith feature is the same as taking the dot product of the ith column of X and y_err.

X_1         y_err
|x11|       | y1 | 
|x21|  dot  | y2 | 
|x31|       | y3 |

w_1 = (x11*y1) + (x21*y2) + (x31*y3)

X_2         y_err
|x12|       | y1 | 
|x22|  dot  | y2 | 
|x32|       | y3 |

w_2 = (x12*y1) + (x22*y2) + (x32*y3)

X_3         y_err
|x13|       | y1 | 
|x23|  dot  | y2 | 
|x33|       | y3 |

w_3 = (x13*y1) + (x23*y2) + (x33*y3)

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


# tested
def _add_x0(X):
    num_rows = X.shape[0]
    ones = np.ones(num_rows)  # create a vector of 1's
    return np.insert(X, 0, ones, axis=1)


def _calc_log_likelihood(X, y_true, W):
    # left half
    XW = np.matmul(X, W)
    YXW = y_true + XW  # all L samples can be summed in parallel

    # right half
    num_rows = X.shape[0]
    ones = np.ones(num_rows)  # create a vector of 1's
    inner = ones + np.exp(XW)
    ln_XW = np.log(inner)

    return YXW - ln_XW


def _get_y_prediction(X, W):
    '''
    Gets y predictions for one X_i.
    Assumes "imaginary" X_0 = 1 for all samples has been added to the matrix.
    '''

    a = np.dot(X, W)
    b = np.exp(a)
    c = 1 + np.exp(a)

    return b / c


def _get_y_predictions(X, W):
    '''
    Gets y predictions for all samples.
    Assumes "imaginary" X_0 = 1 for all samples has been added to the matrix.
    '''
    num_rows = X.shape[0]
    XW = np.matmul(X, W)  # all L samples can be summed in parallel
    top = np.exp(XW)
    ones = np.ones(num_rows)  # create a vector of 1's
    bottom = ones + top
    return top / bottom


def _calc_gradient(X, y_true, y_pred):
    """
    Calculates the gradient.
    Assumes "imaginary" X_0 = 1 for all samples has been added to the matrix.
    See Note 1 on gradient calculation.

    :param X:  L x n matrix, where L is the number of samples and n is the number of features
    :param y_true:  L x 1 vector
    :param y_pred: L x 1 vector
    :return: Gradient in the form of an L x 1 vector
    """
    y_err = y_true - y_pred
    return np.matmul(X.T, y_err)


def _update_weights(W, eta, gradient):
    """
    W <- W + (eta * gradient)

    :param W:
    :param eta:
    :param gradient:
    :return:
    """
    change = eta * gradient
    return W + change


# ?? diff is a vector, how to compare with epsilon?
def gradient_descent(X, y_true, eta, epsilon, W):
    """
    Performs gradient descent to derive regression coefficients.

    :param X:
    :param y_true:
    :param eta:
    :param epsilon:
    :param W:
    :return:
    """
    # set initial weights
    weights = W

    # append imaginary column X_0=1 to accommodate w_0
    X_aug = _add_x0(X)

    # calculate original log likelihood
    prev_log_likelihood = _calc_log_likelihood(X_aug, y_true, weights)

    # perform gradient descent
    count = 0
    diff = np.Inf
    while diff > epsilon:

        count += 1
        if count > 100000:
            break  # stop descending

        # update weights
        y_pred = _get_y_predictions(X_aug, weights)
        gradient = _calc_gradient(X_aug, y_true, y_pred)
        weights = _update_weights(weights, eta, gradient)

        # calculate difference
        log_likelihood = _calc_log_likelihood(X_aug, y_true, weights)
        diff = prev_log_likelihood - log_likelihood

        # save log likelihood for next round
        prev_log_likelihood = log_likelihood

    print('count of rounds', count)
    return weights
