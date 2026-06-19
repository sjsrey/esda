import pytest
import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as spla

from esda.fdr import SpatialFDR


class MockGraph:
    """A lightweight mock Graph object mimicking libpysal.graph.Graph for testing."""

    def __init__(self, n, matrix_type="symmetric"):
        self.n = n
        if matrix_type == "symmetric":
            W_raw = np.zeros((n, n))
            for i in range(n):
                if i > 0:
                    W_raw[i, i - 1] = 0.5
                if i < n - 1:
                    W_raw[i, i + 1] = 0.5
            self.sparse = sparse.csr_matrix(W_raw)
        elif matrix_type == "zero":
            self.sparse = sparse.csr_matrix((n, n))
        else:
            raise ValueError("Unknown matrix type")

    def transform(self, strategy):
        return self


def test_spatial_fdr_shape_mismatch():
    """Ensure ValueError is raised if any array length doesn't match graph size."""
    graph = MockGraph(n=50)
    pvalues = np.random.uniform(0, 1, 50)
    y_mismatch = np.random.normal(0, 1, 45)  # Intentional size mismatch

    with pytest.raises(ValueError, match="Shape mismatch"):
        SpatialFDR(pvalues, y_mismatch, graph)


def test_spatial_fdr_invalid_transform():
    """Ensure invalid transform strings trigger a clear ValueError."""
    graph = MockGraph(n=10)
    pvalues = np.random.uniform(0, 1, 10)
    y = np.random.normal(0, 1, 10)

    with pytest.raises(ValueError, match="Transform must be"):
        SpatialFDR(pvalues, y, graph, transform="Z")


def test_spatial_fdr_zero_variance_handling():
    """Ensure class gracefully handles an empty weights field/zero spatial process variance."""
    n = 20
    graph = MockGraph(n=n, matrix_type="zero")
    pvalues = np.ones(n) * 0.05
    y_flat = np.ones(n) * 10.0  # Perfectly flat spatial field

    fdr = SpatialFDR(pvalues, y_flat, graph)
    assert fdr.rho == 0.0
    assert fdr.n_eff == float(n)


def test_spatial_fdr_neff_ceilings():
    """Ensure n_eff values safely clip at nominal sample size boundaries."""
    n = 100
    graph = MockGraph(n=n)
    np.random.seed(1234)
    pvalues = np.random.uniform(0.5, 1.0, n)
    y = np.random.normal(0, 1, n)

    fdr = SpatialFDR(pvalues, y, graph)
    assert fdr.n_eff <= n
    assert fdr.n_eff > 0


def test_spatial_fdr_index_mapping():
    """Verify output arrays perfectly preserve the original un-sorted layout order."""
    n = 5
    graph = MockGraph(n=n)
    pvalues = np.array([0.9, 0.0001, 0.5, 0.0002, 0.8])
    y = np.array([1.2, 3.4, -0.5, 2.1, 0.8])

    fdr = SpatialFDR(pvalues, y, graph, alpha=0.05)

    # Assert truthiness of the indices corresponding to original positions
    assert fdr.reject[1]
    assert fdr.reject[3]
    assert not fdr.reject[0]

    assert fdr.critical_values.shape == (n,)


def test_spatial_fdr_separation_power():
    """Verify the correction method successfully isolates signals using y and p-values together."""
    n = 100
    graph = MockGraph(n=n)

    np.random.seed(42)
    y = np.random.normal(0, 1, n)
    pvalues = np.random.uniform(0.1, 1.0, n)

    # Simulate a true hot-spot cluster signal
    significant_indices = [10, 25, 50, 75, 90]
    pvalues[significant_indices] = 0.00001

    fdr = SpatialFDR(pvalues, y, graph, alpha=0.05)

    for idx in significant_indices:
        assert fdr.reject[idx]

    assert not fdr.reject[0]


def test_spatial_fdr_positive_autocorrelation():
    """Verify that when rho > 0, APLE catches it and n_eff drops below n."""
    n = 100
    graph = MockGraph(n=n, matrix_type="symmetric")

    # 1. Generate a true SAR process with positive rho (e.g., rho = 0.6)
    # y = (I - rho * W)^(-1) * epsilon
    W_sparse = graph.sparse
    I = sparse.eye(n, format="csr")
    true_rho = 0.6

    np.random.seed(42)
    epsilon = np.random.normal(0, 1, size=n)

    # Solve the spatial system to create clustered data
    A = I - true_rho * W_sparse
    y_spatial = spla.spsolve(A, epsilon)

    # 2. Setup p-values (mix of significant signals and noise)
    pvalues = np.random.uniform(0.1, 1.0, n)
    significant_indices = [12, 35, 67]
    pvalues[significant_indices] = 0.00001

    # 3. Fit the SpatialFDR
    fdr = SpatialFDR(pvalues, y_spatial, graph, alpha=0.05, transform="B")

    # 4. Assertions
    # APLE should recover a distinct positive spatial parameter
    assert fdr.rho > 0.1
    assert fdr.rho < 1.0

    # High spatial dependency must collapse the effective sample size
    assert fdr.n_eff < n

    # The true signals should still be detected under the modified threshold
    for idx in significant_indices:
        assert fdr.reject[idx]
