import numpy as np
import packages.LogisticRegression.gradient_descent as gd


def test__calc_log_likelihood():
    X = np.array([[.1, .5, .1, .1],
                  [.1, .1, .1, .1],
                  [.1, .1, .2, .3]])
    w = np.array([2, 4, 5, 6])
    y_true = np.array([1, 0, 0])

    # generate expected values
    num_rows = X.shape[0]
    ones = np.ones(num_rows)
    expect_YXW = np.array([4.3, 1.7, 3.4])
    XW_test = np.matmul(X, w)
    expect_inner = ones + np.exp(XW_test)
    expected = expect_YXW - np.log(expect_inner)

    # compare
    actual = gd._calc_log_likelihood(X, y_true, w)
    np.testing.assert_allclose(actual, expected)  # rounding makes exact match impossible


def test__get_y_prediction():
    x = np.array([1, 1, 2])
    w = np.array([4, 5, 6])
    expected = np.exp(21) / (1 + np.exp(21))
    actual = gd._get_y_prediction(x, w)
    assert actual == expected


def test__get_y_predictions():
    X = np.array([[.1, .5, .1, .1],
                  [.1, .1, .1, .1],
                  [.1, .1, .2, .3]])
    w = np.array([2, 4, 5, 6])

    a = gd._get_y_prediction(X[0], w)  # row 1
    b = gd._get_y_prediction(X[1], w)  # row 2
    c = gd._get_y_prediction(X[2], w)  # row 3

    expected = np.array([a, b, c])
    actual = gd._get_y_predictions(X, w)
    np.testing.assert_array_equal(actual, expected)


def test__calc_gradient():
    X = np.array([[1, 5, 1, 1],
                  [1, 1, 1, 1],
                  [1, 1, 2, 3]])  # includes imaginary X_0=1 column
    y_true = np.array([1, 0, 1])
    y_pred = np.array([1, 1, 0])
    expected = np.array([0, 0, 1, 2])
    actual = gd._calc_gradient(X, y_true, y_pred)
    np.testing.assert_array_equal(actual, expected)


def test__update_weights():
    eta = 0.01
    gradient = np.array([1, -2, 3])
    w = np.array([4, 5, 6])

    expected = np.array([4.01, 4.98, 6.03])
    actual = gd._update_weights(w, eta, gradient)
    np.testing.assert_array_equal(actual, expected)
