import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as spla

from .aple import APLE


class SpatialFDR:
    """
    Spatially-Dependent False Discovery Rate (FDR) Correction.

    Parameters:
    -----------
    pvalues : array-like
        1D array of unadjusted p-values from local spatial statistical tests.
    y : array-like
        1D array of the underlying spatial variable (or regression residuals)
        used to calculate the spatial process parameter rho.
    graph : libpysal.graph.Graph
        The spatial connectivity graph object underlying the study area.
    alpha : float, optional
        The target False Discovery Rate. Default is 0.05.
    transform : str, optional
        The weight transformation strategy applied to the graph ('R' or 'B').
    """

    def __init__(self, pvalues, y, graph, alpha=0.05, transform="R"):
        self.pvalues = np.asarray(pvalues, dtype=float).ravel()
        self.y = np.asarray(y, dtype=float).ravel()
        self.graph = graph
        self.alpha = alpha
        self.transform = transform.upper()

        if len(self.pvalues) != self.graph.n or len(self.y) != self.graph.n:
            raise ValueError(
                f"Shape mismatch: Ensure pvalues ({len(self.pvalues)}), "
                f"y ({len(self.y)}), and graph nodes ({self.graph.n}) match."
            )
        y_arr = np.asarray(y)

        if np.var(y_arr) < 1e-10:
            raise ValueError(
                "Input array 'y' must have a non-zero variance. "
                "The provided array is constant or contains insufficient variation."
            )

        self._fit()

    def _fit(self):
        n = self.graph.n

        if self.transform == "R":
            W = self.graph.transform("R").sparse
        elif self.transform == "B":
            W = self.graph.sparse
        else:
            raise ValueError("Transform must be 'R' or 'B'.")

        self.rho = APLE(self.y, W).statistic_

        I = sparse.eye(n, format="csr")
        A = I - self.rho * W
        ones = np.ones(n)

        v = spla.spsolve(A.T, ones)
        x = spla.spsolve(A, v)
        denom_neff = np.sum(x)

        self.n_eff = float((n**2) / denom_neff)
        if self.n_eff > n or self.n_eff <= 0:
            self.n_eff = float(n)

        sort_idx = np.argsort(self.pvalues)
        sorted_p = self.pvalues[sort_idx]
        ranks = np.arange(1, n + 1)

        # Dynamic threshold scaling
        sorted_thresholds = (ranks / self.n_eff) * self.alpha

        # CRITICAL FIX: If the threshold curve eclipses alpha or 1.0 due to a low n_eff,
        # it must cap out. You cannot have a critical hurdle less stringent than alpha itself
        # for independent tests when ranks push past n_eff.
        standard_bh_ceiling = (ranks / n) * self.alpha
        sorted_thresholds = np.minimum(sorted_thresholds, self.alpha)

        # Find the largest k where p_(k) <= threshold_(k)
        significant_indicators = sorted_p <= sorted_thresholds

        if not np.any(significant_indicators):
            max_k = 0
        else:
            max_k = np.max(ranks[significant_indicators])

        self.reject = np.zeros(n, dtype=bool)
        if max_k > 0:
            self.reject[sort_idx[:max_k]] = True

        unsort_idx = np.argsort(sort_idx)
        self.critical_values = sorted_thresholds[unsort_idx]
