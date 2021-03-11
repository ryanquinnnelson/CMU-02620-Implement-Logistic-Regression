import pytest
import numpy as np
import packages.kmeans.kmeans as km


def test__position_centroids():
    X = np.array([[1, 2],
                  [1, 3],
                  [1, 4],
                  [1, 5]])
    actual = km._position_centroids(X, 3)
    assert actual.shape == (3, 2)

    # check that every centroid came from X
    for each in actual.tolist():
        assert each in X


def test__initialize_centroids_with_initialization_not_match_clusters():
    n_clusters = 5
    d = 3
    initialization = np.array([[1, 2, 3],
                               [1, 2, 4]])

    with pytest.raises(ValueError):
        km._position_centroids_with_initialization(n_clusters, d, initialization)


def test__initialize_centroids_with_initialization_not_match_dimension():
    n_clusters = 2
    d = 2
    initialization = np.array([[1, 2, 3],
                               [1, 2, 4]])

    with pytest.raises(ValueError):
        km._position_centroids_with_initialization(n_clusters, d, initialization)


def test__initialize_centroids_with_initialization():
    n_clusters = 2
    d = 3
    initialization = np.array([[1, 2, 3],
                               [1, 2, 4]])
    expected = initialization
    actual = km._position_centroids_with_initialization(n_clusters, d, initialization)
    np.testing.assert_array_equal(actual, expected)


def test__calculate_distances():
    X = np.array([[0, 2],
                  [3, 6]])

    centroids = np.array([[0, 0],
                          [0, 2],
                          [4, 3]])

    dist_0_2_and_0_0 = 4.0
    dist_0_2_and_0_2 = 0.0
    dist_0_2_and_4_3 = 17

    dist_3_6_and_0_0 = 45
    dist_3_6_and_0_2 = 25.0
    dist_3_5_and_4_3 = 10

    expected = np.array([[dist_0_2_and_0_0, dist_0_2_and_0_2, dist_0_2_and_4_3],
                         [dist_3_6_and_0_0, dist_3_6_and_0_2, dist_3_5_and_4_3]])
    actual = km._calculate_distances(X, centroids)
    np.testing.assert_array_equal(np.round(actual,2), expected)


def test__assign_clusters():
    X = np.array([[1, 2],
                  [0, 2],
                  [1, 1],
                  [5, 5],
                  [5, 6]])

    # (i,j) is distance between ith sample and jth cluster
    distances = np.array([[1, 5],
                          [1, 4],
                          [2, 3],
                          [5, 1],
                          [6, 2]])

    expected = np.array([0, 0, 0, 1, 1])
    actual = km._assign_clusters(X, distances)
    np.testing.assert_array_equal(actual, expected)


def test__calculate_centroid():
    selected = np.array([[1, 2, 3, 4],
                         [5, 6, 3, 8],
                         [9, 10, 3, 12]])
    expected = np.array([5.0, 6.0, 3.0, 8.0])
    actual = km._calculate_centroid(selected)
    np.testing.assert_array_equal(actual, expected)


def test__move_centroids():
    X = np.array([[1, 2],
                  [0, 2],
                  [1, 1],
                  [5, 5],
                  [5, 6]])

    assignments = np.array([0, 0, 0, 1, 1])
    n_clusters = 2
    expected = np.array([[2 / 3, 5 / 3],
                         [5.0, 5.5]])
    actual = km._move_centroids(X, assignments, n_clusters)
    np.testing.assert_array_equal(actual, expected)


def test__assignments_changed_true():
    before = np.array([1, 2, 3, 4])
    after = np.array([1, 2, 1, 4])
    assert km._assignments_changed(before, after)


def test__assignments_changed_false():
    before = np.array([1, 2, 3, 4])
    after = np.array([1, 2, 3, 4])
    assert km._assignments_changed(before, after) is False


def test__cluster_unused_false():
    n_clusters = 3
    assignments = np.array([0, 2, 0, 1, 1])
    assert km._cluster_unused(n_clusters, assignments) is False


def test__cluster_unused_true():
    n_clusters = 3
    assignments = np.array([0, 0, 0, 1, 1])
    assert km._cluster_unused(n_clusters, assignments)


def test__initialize_centroids_initialization_value_error():
    X = np.array([[0, 0],
                  [1, 1],
                  [5, 5],
                  [5, 6]])
    initialization = np.array([[0, 0],
                               [-1, -1]])

    with pytest.raises(ValueError):
        km._initialize_centroids(X, 2, initialization)


def test__initialize_centroids_initialization():
    X = np.array([[0, 0],
                  [1, 1],
                  [5, 5],
                  [5, 6]])
    n_clusters = 2
    initialization = np.array([[0, 0],
                               [4, 4]])
    expected_centroids = initialization
    expected_assignments = np.array([0, 0, 1, 1])
    actual_centroids, actual_assignments = km._initialize_centroids(X, n_clusters, initialization)
    np.testing.assert_array_equal(actual_centroids, expected_centroids)
    np.testing.assert_array_equal(actual_assignments, expected_assignments)


def test__initialize_centroids():
    X = np.array([[0, 0],
                  [1, 1],
                  [5, 5],
                  [5, 6]])
    n_clusters = 2
    centroids, assignments = km._initialize_centroids(X, n_clusters)
    assert centroids.shape[0] == 2


def test__sum_cluster_distances():
    cluster = 0
    distances = np.array([[1, 2],
                          [3, 4],
                          [5, 6],
                          [7, 8]])
    assignments = np.array([1, 1, 0, 0])
    expected = 12
    actual = km._sum_cluster_distances(cluster, distances, assignments)
    assert actual == expected

    cluster = 1
    expected = 6
    actual = km._sum_cluster_distances(cluster, distances, assignments)
    assert actual == expected


def test_kmeans__init__():
    k = km.KMeans(5)
    assert k.n_clusters == 5.0
    assert k.centroids is None


def test_kmeans_fit_initialization():
    k = km.KMeans(2)
    X = np.array([[0, 0],
                  [1, 1],
                  [5, 5],
                  [5, 6]])
    initialization = np.array([[0, 0],
                               [4, 4]])

    expected_centroid_1 = [5.0, 5.5]
    expected_centroid_2 = [0.5, 0.5]

    k.fit(X, initialization)
    actual_centroid_1 = k.centroids[0].tolist()
    actual_centroid_2 = k.centroids[1].tolist()

    expected_1_found = expected_centroid_1 == actual_centroid_1 or expected_centroid_1 == actual_centroid_2
    expected_2_found = expected_centroid_2 == actual_centroid_1 or expected_centroid_2 == actual_centroid_2

    assert expected_1_found
    assert expected_2_found


def test_kmeans_fit_no_initialization():
    k = km.KMeans(2)
    X = np.array([[0, 0],
                  [1, 1],
                  [5, 5],
                  [5, 6]])

    expected_centroid_1 = [5.0, 5.5]
    expected_centroid_2 = [0.5, 0.5]

    k.fit(X)
    actual_centroid_1 = k.centroids[0].tolist()
    actual_centroid_2 = k.centroids[1].tolist()

    expected_1_found = expected_centroid_1 == actual_centroid_1 or expected_centroid_1 == actual_centroid_2
    expected_2_found = expected_centroid_2 == actual_centroid_1 or expected_centroid_2 == actual_centroid_2

    assert expected_1_found
    assert expected_2_found


def test_kmeans_predict():
    k = km.KMeans(2)
    X = np.array([[0, 0],
                  [1, 1],
                  [5, 5],
                  [5, 6]])

    k.centroids = np.array([[5.0, 5.5],
                            [0.5, 0.5]])

    expected = np.array([1, 1, 0, 0])
    actual = k.predict(X)
    np.testing.assert_array_equal(actual, expected)


def test_kmeans_score():
    k = km.KMeans(2)
    X = np.array([[0, 0],
                  [1, 1],
                  [5, 5],
                  [5, 6]])

    k.centroids = np.array([[5.0, 5.5],
                            [0.5, 0.5]])
    expected = 1.5
    actual = k.score(X)
    assert round(actual,2) == expected
