import numpy as np
from scipy.spatial import distance_matrix


# tested
def _position_centroids(X, n_clusters):
    """
    Selects random point in X for each requested centroid, without replacement.

    :param X: n x d matrix
    :param n_clusters: scalar
    :return:
    """
    idx = np.random.choice(X.shape[0], n_clusters, replace=False)
    centroids = X[idx]

    return centroids


# tested
def _position_centroids_with_initialization(n_clusters, d, initialization):
    """
    Generates requested number of centroids with given initialization values.

    :param n_clusters: scalar
    :param d: scalar
    :param initialization: n_cluster x d matrix
    :return: n_cluster x d matrix
    """
    # validate dimension and quantity of initialization
    rows = initialization.shape[0]
    cols = initialization.shape[1]

    if not n_clusters == rows:
        raise ValueError('The number of centroids in the initialization does not match the number of clusters:', rows)

    if not d == cols:
        raise ValueError('The dimension of centroids in the initialization does not match the dimension of the data:',
                         cols)

    return initialization


# tested
def _calculate_distances(X, centroids):
    """
    Calculates squared Euclidean distance between each sample and each centroid.
    :param X: n x d matrix
    :param centroids: n_clusters x d matrix
    :return: n x n_clusters matrix where (i,j) represents distance between sample i and cluster j
    """
    return np.square(distance_matrix(X, centroids))


# tested
def _assign_clusters(X, distances):
    """
    Assigns each sample to a cluster.
    Todo - find more efficient way to find min distance for each sample
    :param X:  n x d matrix
    :param distances: n x n_clusters matrix
    :return: n x 1 vector, where (i,1) contains the cluster for the ith sample
    """

    # for each sample, select the closest centroid as its cluster
    # cluster number is the order of clusters in the centroid matrix
    rows = X.shape[0]
    assignments = np.full((rows,), -1)  # set up array to define cluster for each sample

    for i in range(rows):
        cluster = np.argmin(distances[i], axis=0)
        assignments[i] = cluster

    return assignments


# tested
def _calculate_centroid(selected):
    """
    Calculates the d-dimension centroid of selected samples.
    :param selected: m x d matrix where m is the number of samples assigned to the current cluster
    :return: d x 1 vector
    """
    return np.mean(selected, axis=0)


# tested
def _move_centroids(X, assignments, n_clusters):
    """
    Calculates the new centroid of each cluster based on cluster assignments of the samples.
    Todo - consider deriving n_clusters from assignments to avoid potential errors
    :param X:  n x d matrix
    :param assignments: n x 1 vector
    :param n_clusters: scalar
    :return: n_clusters x d matrix
    """
    centroids = []
    for i in range(n_clusters):
        # identify all samples assigned to the ith cluster
        idx = np.where(assignments == i)

        # calculate the centroid of the ith cluster
        centroid = _calculate_centroid(X[idx])
        centroids.append(centroid)

    return np.array(centroids)


# tested
def _assignments_changed(before, after):
    """
    Compares assignments before and after moving centroid.
    :param before: n x 1 vector
    :param after: n x 1 vector
    :return: True if no assignments changed, False otherwise.
    """
    return not np.array_equal(before, after)


# tested
def _cluster_unused(n_clusters, assignments):
    """
    Checks whether every cluster has at least one assignment.
    :param n_clusters: scalar
    :param assignments: n x 1 vector
    :return: True if every cluster has at least one assignment, False otherwise.
    """
    # get unique assignments
    unique = np.unique(assignments)
    for i in range(n_clusters):
        if i not in unique:
            return True

    return False


# tested
def _initialize_centroids(X, n_clusters, initialization=None):
    """
    Initializes centroids in positions so that every cluster has at least one assignment.
    If (1) no initialization is provided and (2) one or more centroids does not contain samples after first assignment, re-initializes clusters to new positions.
    If (1) initialization is provided (2) one or more centroids does not contain samples after first assignment, raises ValueError (otherwise fit() would be stuck in an infinite loop.)
    :param X: n x d matrix
    :param n_clusters: scalar
    :param initialization: Initial positions for centroids. Default value is None.
    :return: (n_clusters x d matrix, n x 1 vector) Tuple containing (centroids, assignments)
    """
    centroids = None
    assignments = None

    cluster_unused = True
    while cluster_unused:

        # initialize centroids
        cols = X.shape[1]
        if initialization is not None:
            centroids = _position_centroids_with_initialization(n_clusters, cols, initialization)
        else:
            centroids = _position_centroids(X, n_clusters)

        # assign samples to clusters
        distances = _calculate_distances(X, centroids)
        assignments = _assign_clusters(X, distances)

        # stop loop if necessary
        cluster_unused = _cluster_unused(n_clusters, assignments)

        if cluster_unused and initialization is not None:
            raise ValueError('Initialization results in at least one cluster not being assigned any samples.')

    return centroids, assignments


# tested
def _sum_cluster_distances(k, distances, assignments):
    """
    For all samples assigned to the kth cluster, adds up all of the distances between samples and centroids.
    :param k: scalar
    :param distances: n x n_clusters matrix
    :param assignments: n x 1 vector
    :return: scalar
    """

    idx = np.where(assignments == k)  # indexes for all samples assigned to cluster k

    # select only those rows of the distance matrix for samples assigned to cluster k
    # select only kth column
    assigned_distances = distances[idx, k]

    return np.sum(assigned_distances)


class KMeans:
    """
    Implements K-means clustering.
    """

    def __init__(self, n_clusters):
        """

        :param n_clusters: Number of clusters to use for K-means clustering.
        """
        self.n_clusters = n_clusters
        self.centroids = None

    # tested
    def fit(self, X, initialization=None):
        """
        Finds the centroid for each of n_clusters using given data.

        :param X: n x d matrix
        :param initialization: n_clusters x d matrix containing initial positions for centroids. Default value is None.
        :return:
        """
        # initialize position of centroids and assign clusters
        centroids, assignments_before = _initialize_centroids(X, self.n_clusters, initialization)

        assignments_changed = True
        while assignments_changed:

            # move centroids
            centroids = _move_centroids(X, assignments_before, self.n_clusters)

            # reassign samples to clusters
            distances = _calculate_distances(X, centroids)
            assignments_after = _assign_clusters(X, distances)

            # check if assignments changed
            assignments_changed = _assignments_changed(assignments_before, assignments_after)
            if assignments_changed:
                assignments_before = assignments_after  # save assignments for the next round

        # centroids have been defined for each cluster
        self.centroids = centroids
        return self

    # tested
    def predict(self, X):
        """
        Determines the cluster for each sample in the data.
        :param X: n x d matrix
        :return: n x 1 vector where (i,1) is the cluster assignment for the ith sample
        """
        distances = _calculate_distances(X, self.centroids)
        assignments = _assign_clusters(X, distances)

        return assignments

    # tested
    def score(self, X):
        """
        Calculates the value of the objective function:

        SUM_k SUM_{i in C_k} || x_i - mu_k||_2^2

        where
        - k is the number of clusters
        - x_i is the ith sample in the kth cluster
        - mu_k is the kth centroid
        :param X: n x d matrix
        :return: scalar
        """
        # Calculate squared Euclidean distance between each sample and each centroid
        distances = _calculate_distances(X, self.centroids)
        assignments = _assign_clusters(X, distances)

        # sum the distances for each cluster
        sums = []
        for k in range(self.n_clusters):
            sum_k = _sum_cluster_distances(k, distances, assignments)
            sums.append(sum_k)

        # sum the cluster sums
        objective = sum(sums)

        return objective

    def fit_and_score(self, X, initialization=None):
        """
        Scores the model after every positioning of the centroids.
        :param X: n x d matrix
        :param initialization: n_clusters x d matrix containing initial positions for centroids. Default value is None.
        :return: List of scores
        """
        # initialize position of centroids and assign clusters
        centroids, assignments_before = _initialize_centroids(X, self.n_clusters, initialization)

        scores = []
        assignments_changed = True
        while assignments_changed:

            # score model
            # sum the distances for each cluster
            distances = _calculate_distances(X, centroids)

            sums = []
            for k in range(self.n_clusters):
                sum_k = _sum_cluster_distances(k, distances, assignments_before)
                sums.append(sum_k)

            # sum the cluster sums
            objective = sum(sums)
            scores.append(objective)

            # move centroids
            centroids = _move_centroids(X, assignments_before, self.n_clusters)

            # reassign samples to clusters
            distances = _calculate_distances(X, centroids)
            assignments_after = _assign_clusters(X, distances)

            # check if assignments changed
            assignments_changed = _assignments_changed(assignments_before, assignments_after)
            if assignments_changed:
                assignments_before = assignments_after  # save assignments for the next round

        # centroids have been defined for each cluster
        self.centroids = centroids

        return scores
