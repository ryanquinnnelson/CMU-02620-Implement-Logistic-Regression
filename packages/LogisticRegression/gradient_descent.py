import numpy as np

def add_X0(X):
    num_rows = X.shape[0]
    ones = np.ones(num_rows)  # create a vector of 1's
    return np.insert(X,0,ones,axis=1)

print('test add_X0()')
X_test = np.array([[1],[2],[3]])
actual = add_X0(X_test)
expected = np.array([[1, 1],[1, 2],[1, 3]])
assert expected.tolist() == actual.tolist()

print('test add_X0()')
X_test = np.array([[1,9],[2,7],[3,8]])
actual = add_X0(X_test)
expected = np.array([[1, 1, 9],[1, 2, 7],[1, 3, 8]])
assert expected.tolist() == actual.tolist()