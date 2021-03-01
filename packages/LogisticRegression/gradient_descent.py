"""
Implementation of gradient descent for Logistic Regression, as defined by Machine Learning (Mitchell).
Assumes "imaginary" X_0 = 1 for all samples has been added to the matrix to accommodate w_0 (i.e. X has been augmented
so that its first column is 1's).
"""

import numpy as np

"""
Note 1 - Explanation of _calc_inner()

Many of the formulas for Logistic Regression involve a calculation between n x 1 weights vector w and L x n feature 
matrix X:

w_0 + SUM_i^n w_i X_i^L
where
- i is the ith feature
- n is the number of features
- L is the number of samples

This can be accomplished efficiently using matrix multiplication, as explained below. As discussed in Mitchell, we 
augment X with imaginary X_0 column to accommodate w_0.

CASE 1 - Single feature, single sample
-----------------------------------------
X        w                 X_aug        w
|x11|    | w0 |            | 1 |x12|    | w0 | 
         | w1 |   ->                    | w1 |      
                      
We perform the summation using dot product and get a scalar.

X_aug dot w = 1*w0 + x12*w1


CASE 2 - Multiple features, single sample
-----------------------------------------
X            w                 X_aug            w
|x11|x12|    | w0 |            | 1 |x12|x13|    | w0 | 
             | w1 |   ->                        | w1 | 
             | w2 |                             | w2 |                             

                      
We perform the summation using dot product and get a scalar.

X_aug dot w = 1*w0 + x12*w1 + x13*w2


CASE 3 - Single feature, multiple samples
-----------------------------------------
X        w                 X_aug         w   
|x11|    | w0 |            | 1 |x12|     | w0 |
|X21|    | w1 |   ->       | 1 |x22|     | w1 |
|x31|                      | 1 |x32|

We could perform the summation for each sample separately and place the results into a vector, or we could use matrix 
multiplication and do all calculations simultaneously. The result is a vector.

               |  1*w0 + x12*w1  |
(X_aug)(w) =   |  1*w0 + x22*w1  |
               |  1*w0 + x32*w1  |


CASE 4 - Multiple features, multiple samples
--------------------------------------------
X                w             X_aug                w
|x11|x12|x13|    | w0 |        | 1 |x12|x13|x14|    | w0 |
|X21|x22|x23|    | w1 |   ->   | 1 |x22|x23|x24|    | w1 |
|x31|x32|x33|    | w2 |        | 1 |x32|x33|x34|    | w2 |
                 | w3 |                             | w3 | 

This works the same as CASE 3. The result is a vector.

               |  1*w0 + x12*w1 + x13*w2 + x14*w3  |
(X_aug)(w) =   |  1*w0 + x22*w1 + x23*w2 + x24*w3  |
               |  1*w0 + x32*w1 + x33*w2 + x34*w3  |

"""




"""
Note 1 - Explanation of how _calc_gradient() works. According to Mitchell, the ith partial in the gradient can be
calculated as:

    d l(W)/d w_i = SUM_L X_i^L y_err^L
    
where 
- L is the number of samples
- i is the ith feature
- y_err = ( Y^L - P(Y^L = 1 |X^L,W))

This can be accomplished efficiently using matrix multiplication, as explained below.

X                y_err
|x11|x12|x13|    | y1 |
|X21|x22|x23|    | y2 |
|x31|x32|x33|    | y3 |

We could do this column by column. Calculating the partial for the ith feature is the same as taking the dot product of 
the ith column of X and y_err:

                 X_1         y_err
                 |x11|       | y1 | 
d l(W) / d w_1 = |x21|  dot  | y2 |  = (x11*y1) + (x21*y2) + (x31*y3)
                 |x31|       | y3 |
                 

                 X_2         y_err
                 |x12|       | y1 | 
d l(W) / d w_2 = |x22|  dot  | y2 |  = (x12*y1) + (x22*y2) + (x32*y3)
                 |x32|       | y3 |


However, we can perform matrix multiplication to do all these calculations at the same time. We need to transpose X 
first because we want to perform column-wise multiplications.

X              y_err
|x11|x12|x13|  | y1 | 
|x21|x22|x23|  | y2 |
|x31|x32|x33|  | y3 |

              | (x11*y1) + (x12*y2) + (x13*y3) |
(X)(y_err) =  | (x21*y1) + (x22*y2) + (x23*y3) |       (incorrect!)
              | (x31*y1) + (x32*y2) + (x33*y3) |

X_transpose    y_err
|x11|x21|x31|  | y1 | 
|x12|x22|x32|  | y2 |
|x13|x23|x33|  | y3 |
                                   | d w1 |    | (x11*y1) + (x21*y2) + (x31*y3) |
gradient = (X_transpose)(y_err) =  | d w2 | =  | (x12*y1) + (x22*y2) + (x32*y3) |    (correct!)
                                   | d w3 |    | (x13*y1) + (x23*y2) + (x33*y3) |
"""

"""
Note 2 - Explanation of how _get_y_predictions() works
Consider L x n matrix X and n x 1 vector w, where L is the number of samples and n is the number of features.



For each sample, we need to calculate a predicted label. According to Mitchell, we do this by calculating P(Y=1|x,w).

P(Y=1|x,w) = exp(A) / 1 + exp(A)
where
- i is the ith feature
- A = w_0 + SUM_i (w_i X_i)

For a single sample, A can be calculated as a dot product. Note that in this implementation, we assume matrix X 
has been augmented so that its first column is 1's to accommodate w_0:

For l=1: w_0 + SUM_i (w_i X_i) = X dot w = (x11*w0) + (x12*w1) + (x13*w2)
For l=2: w_0 + SUM_i (w_i X_i) = X dot w = (x21*w0) + (x22*w1) + (x23*w2)

However, we can efficiently calculate A for all samples at the same time, using matrix multiplication:

     | y_pred_1 |   | (x11*w0) + (x12*w1) + (x13*w2) |
Xw = | y_pred_2 | = | (x21*w0) + (x22*w1) + (x23*w2) |
     | y_pred_3 |   | (x31*w0) + (x32*w1) + (x33*w2) |

Then we can perform exponentiation and addition on the resulting vector.
"""

"""
Note 3 - Explanation of how _calc_log_likelihood() works

According to Mitchell, log likelihood l(W) can be calculated as follows:

l(W) = SUM_L [ Y^L * A - ln( 1 + exp(A)) ]

where 
- L is the number of samples
- A = w_0 + SUM_i^n w_i X_i^L
- i is the ith feature
- n is the number of features

The math required to calculate A is the same as Note 2. 
"""


# tested
def _calc_inner(X, w):
    """
    Performs the inner calculation w_0 + SUM_i w_i X_i^L. See Note 1 for explanation of function logic.

    :param X:  L x n matrix, where L is the number of samples and n is the number of features
    :param w: n x 1 vector
    :return:
    """
    return np.matmul(X, w)


def _calc_log_likelihood(X, y_true, w):
    """
    Calculates log likelihood. See Note 3 for explanation of function logic.

    :param X: L x n matrix, where L is the number of samples and n is the number of features
    :param y_true: L x 1 vector
    :param w: n x 1 vector
    :return:
    """
    # left half
    Xw = np.matmul(X, w)
    yXw = y_true + Xw  # all L samples can be summed in parallel

    # right half
    num_rows = X.shape[0]
    ones = np.ones(num_rows)
    inner = ones + np.exp(Xw)
    ln_Xw = np.log(inner)

    return yXw - ln_Xw



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
    :return: Gradient in the form of an n x 1 vector
    """
    y_err = y_true - y_pred
    return np.matmul(X.T, y_err)



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
