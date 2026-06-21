import numpy as np
import libpysal
from scipy.sparse import csr_matrix, csc_matrix, random
import pytest
import matplotlib
from pytest_mock import mocker


from esda.aple import APLE


# Helper function to create a simple spatial weights matrix
def create_simple_W(n):
    W = libpysal.weights.lat2W(int(np.sqrt(n)), int(np.sqrt(n)))
    return W.sparse


# Helper function to create random data
def create_random_X(n):
    return np.random.rand(n)


# Test the initialization of APLE class
def test_aple_init():
    n = 100
    X = create_random_X(n)
    W = create_simple_W(n)

    aple_model = APLE(X, W, trace=True)

    assert isinstance(aple_model.X, np.ndarray)
    assert isinstance(aple_model.Z, np.ndarray)
    assert isinstance(aple_model.W, csr_matrix)
    assert aple_model.n == n
    assert aple_model.trace is True


# Test the normalization of W
def test_normalize_w():
    n = 100
    X = create_random_X(n)
    W_libpysal = create_simple_W(n)
    W_sparse = random(100, 100, density=0.2, format="csr")

    aple_model_libpysal = APLE(X, W_libpysal, trace=True)
    aple_model_sparse = APLE(
        X, W_sparse, trace=True
    )  # Ensure W_sparse has the correct shape

    assert isinstance(aple_model_libpysal.W, csr_matrix)
    assert isinstance(aple_model_sparse.W, csr_matrix)


# Test the pre-computation logic with trace=True
def test_compute_pre_trace():
    n = 100
    X = create_random_X(n)
    W = create_simple_W(n)

    aple_model = APLE(X, W, trace=True)

    assert aple_model.statistic_ is not None
    assert isinstance(aple_model.A, csc_matrix)
    assert isinstance(aple_model.wwt2, csr_matrix)


# Test the pre-computation logic with trace=False
def test_compute_pre_no_trace():
    n = 100
    X = create_random_X(n)
    W = create_simple_W(n)

    aple_model = APLE(X, W, trace=False)

    assert aple_model.statistic_ is not None
    assert isinstance(aple_model.A, csc_matrix)
    assert isinstance(aple_model.wwt2, csr_matrix)


# Test the plot method (mocking the plotting to avoid visual checks)
def test_plot(mocker):
    n = 100
    X = create_random_X(n)
    W = create_simple_W(n)

    aple_model = APLE(X, W, trace=True)

    # Mock the plt methods to prevent actual plotting
    mocker.patch("matplotlib.pyplot.scatter")
    mocker.patch("matplotlib.pyplot.plot")
    mocker.patch("matplotlib.pyplot.xlabel")
    mocker.patch("matplotlib.pyplot.ylabel")
    mocker.patch("matplotlib.pyplot.title")
    mocker.patch("matplotlib.pyplot.legend")
    mocker.patch("matplotlib.pyplot.show")

    aple_model.plot()

    # Assert that the plotting methods were called
    matplotlib.pyplot.scatter.assert_called_once()
    matplotlib.pyplot.plot.assert_called_once()
    matplotlib.pyplot.xlabel.assert_called_with("Original Variable (X)")
    matplotlib.pyplot.ylabel.assert_called_with("Transformed Signal (Y)")
    matplotlib.pyplot.title.assert_called_with(
        "APLE Spatial Transformation with Fit Line"
    )
    matplotlib.pyplot.legend.assert_called_once()
    matplotlib.pyplot.show.assert_called_once()


# Test edge case: Small n (less than 2)
def test_edge_case_small_n():
    n = 1
    X = create_random_X(n)
    W = create_simple_W(n)

    with pytest.raises(ValueError, match="inconsistent shapes"):
        APLE(X, W, trace=True)


# Test edge case: Invalid W type
def test_invalid_W_type():
    n = 100
    X = create_random_X(n)
    W = np.random.rand(n, n)  # Non-sparse matrix

    with pytest.raises(
        ValueError,
        match="W must be a libpysal.weights.W, libpysal.graph.Graph, or scipy.sparse matrix.",
    ):
        APLE(X, W, trace=True)


# Test edge case: X is not a vector
def test_invalid_X_type():
    X = np.random.rand(10, 10)  # 2D array instead of 1D vector
    W = create_simple_W(10)

    with pytest.raises(ValueError):
        APLE(X, W, trace=True)


# Test edge case: Size mismatch X and W
def test_size_X_and_W():
    X = create_random_X(11)
    W = csr_matrix((10, 10))

    with pytest.raises(ValueError):
        APLE(X, W, trace=True)


# Test the calculation of APLE statistic
def test_aple_statistic():
    n = 100
    X = create_random_X(n)
    W = create_simple_W(n)

    aple_model_trace = APLE(X, W, trace=True)
    aple_model_no_trace = APLE(X, W, trace=False)

    assert np.isclose(aple_model_trace.statistic_, aple_model_no_trace.statistic_)


# Test the pre-computation of matrices
def test_pre_computed_matrices():
    n = 100
    X = create_random_X(n)
    W = create_simple_W(n)

    aple_model = APLE(X, W, trace=True)

    assert isinstance(aple_model.A, csc_matrix)
    assert isinstance(aple_model.wwt2, csr_matrix)


# Test the transformation of Y
def test_Y_transformation(mocker):
    n = 100
    X = create_random_X(n)
    W = create_simple_W(n)

    aple_model = APLE(X, W, trace=True)

    # Mock the plt methods to prevent actual plotting
    mocker.patch("matplotlib.pyplot.scatter")
    mocker.patch("matplotlib.pyplot.plot")
    mocker.patch("matplotlib.pyplot.xlabel")
    mocker.patch("matplotlib.pyplot.ylabel")
    mocker.patch("matplotlib.pyplot.title")
    mocker.patch("matplotlib.pyplot.legend")
    mocker.patch("matplotlib.pyplot.show")

    aple_model.plot()
    # Assert that the plotting methods were called
    matplotlib.pyplot.scatter.assert_called_once()
    matplotlib.pyplot.plot.assert_called_once()
    matplotlib.pyplot.xlabel.assert_called_with("Original Variable (X)")
    matplotlib.pyplot.ylabel.assert_called_with("Transformed Signal (Y)")
    matplotlib.pyplot.title.assert_called_with(
        "APLE Spatial Transformation with Fit Line"
    )
    matplotlib.pyplot.legend.assert_called_once()
    matplotlib.pyplot.show.assert_called_once()

    assert isinstance(aple_model.Y, np.ndarray)
    assert aple_model.Y.shape == (n, 1)
