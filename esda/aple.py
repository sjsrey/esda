"""
Approximate Profile-Likelihood Estimator (APLE) for Spatial Dependence.

This module implements the APLE statistic, a closed-form estimator for the
spatial-dependence parameter (:math:`\\rho`) in a Spatial Autoregressive (SAR).

While Moran's I is widely used in exploratory spatial data analysis (ESDA), it
is known to be a poor estimator of the spatial dependence parameter :math:`\\rho`
when the true value is not near zero. The APLE statistic
serves as a computationally efficient, closed-form alternative that
demonstrates superior performance in estimating :math:`\\rho` across a
broader range of values compared to Moran's I and Ord's statistic.

This module also provides an APLE scatterplot, an ESDA tool analogous to the
Moran scatterplot, designed to better visualize the strength of spatial
autocorrelation.

References
----------
.. [1] Li, H., Calder, C. A., & Cressie, N. (2007). Beyond Moran's I:
   Testing for Spatial Dependence Based on the Spatial Autoregressive Model.
   Geographical Analysis, 39(4), 357-375.
"""

import numpy as np
import libpysal
from scipy import sparse
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt

__all__ = ["APLE"]


class APLE:
    """
    Approximate Profile-Likelihood Estimator (APLE) for the SAR model spatial dependence parameter.

    The APLE statistic provides a closed-form estimator for the spatial dependence
    parameter :math:`\\rho` in a Spatial Autoregressive (SAR) model.


    Parameters
    ----------
    X : array-like
        The spatial variable (observations on a lattice).
    W : libpysal.weights.W, libpysal.graph.Graph, or scipy.sparse matrix
        The spatial neighborhood matrix defining the connectivity of the lattice.
    trace : bool, default=True
        If True, calculates the sum of squared eigenvalues using the trace of
        the squared weights matrix :math:`W^2`. If False, computes eigenvalues
        directly, which is computationally more expensive.

    Attributes
    ----------
    statistic_ : float
        The calculated APLE statistic.
    n : int
        The number of observations.
    Z : ndarray
        The detrended observations (X - mean(X)).
    A : scipy.sparse matrix
        The pre-computed denominator matrix :math:`W^T W + (\\lambda^T \\lambda / n) I`
        used in the APLE calculation.

    Methods
    -------
    plot()
        Generates an APLE scatterplot, a visualization tool that plots
        transformed signal Y against X, analogous to the Moran scatterplot
        but more informative for assessing spatial autocorrelation.


    References
    ----------
    .. [1] Li, H., Calder, C. A., & Cressie, N. (2007). Beyond Moran's I:
       Testing for Spatial Dependence Based on the Spatial Autoregressive Model.
       Geographical Analysis, 39(4), 357-375.
    """

    def __init__(self, X, W, trace=True):
        # Explicitly store X, then derive Z
        self.X = np.asanyarray(X).flatten()
        self.Z = (self.X - self.X.mean()).reshape(-1, 1)
        self.W = self._normalize_w(W)
        self.n = len(self.Z)
        self.trace = trace

        # State attributes for lazy calculation
        self.statistic_ = None
        self.A = None
        self.wwt2 = None
        self._vals = None
        self._vecs = None

        # Run pre-calculation
        self._compute_pre()

    def _normalize_w(self, W):
        if isinstance(W, (libpysal.weights.W, libpysal.graph.Graph)):
            return W.sparse
        elif isinstance(W, sparse.spmatrix):
            return W
        else:
            raise ValueError(
                "W must be a libpysal.weights.W, libpysal.graph.Graph, or scipy.sparse matrix."
            )
        return sparse.csr_matrix(W)

    def _compute_pre(self):
        # Calculate sum of squared eigenvalues using Trace(W^2)
        if self.trace:
            ss_sq_lam = (self.W @ self.W).diagonal().sum()
        else:
            lam = np.linalg.eigvals(self.W.toarray())
            ss_sq_lam = np.vdot(lam, lam).real

        # Pre-compute matrices used for both stat and plotting
        self.wwt2 = (self.W + self.W.T) / 2
        wtw = self.W.T @ self.W
        self.A = wtw + (ss_sq_lam / self.n) * sparse.eye(self.n)

        # Calculate APLE
        numerator = self.Z.T @ self.wwt2 @ self.Z
        denominator = self.Z.T @ self.A @ self.Z
        self.statistic_ = numerator[0, 0] / denominator[0, 0]

    def plot(self):
        """Computes the transformed data, plots it against X, and adds the APLE fit line."""
        if self._vals is None:
            # Perform sparse spectral decomposition on-demand
            self._vals, self._vecs = eigsh(self.A, k=self.n - 2, which="SM")

        # Calculate A^(-1/2) * ((W + W')/2) * Z
        A_inv_sqrt = self._vecs @ np.diag(1.0 / np.sqrt(self._vals)) @ self._vecs.T
        Y_transformed = A_inv_sqrt @ (self.wwt2 @ self.Z)
        self.Y = Y_transformed

        # Flatten Y for plotting
        y_flat = Y_transformed.flatten()

        # Create the scatter plot
        plt.scatter(self.X, y_flat, alpha=0.6, label="Data Points")

        # Calculate the fit line
        # Formula: Y = mX + b where m = self.statistic_
        m = self.statistic_
        b = np.mean(y_flat) - m * np.mean(self.X)

        x_fit = np.array([self.X.min(), self.X.max()])
        y_fit = m * x_fit + b

        plt.plot(
            x_fit,
            y_fit,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Fit (m={m:.4f})",
        )

        plt.xlabel("Original Variable (X)")
        plt.ylabel("Transformed Signal (Y)")
        plt.title(f"APLE Spatial Transformation with Fit Line")
        plt.legend()
        plt.show()


# Example usage
if __name__ == "__main__":
    import pandas as pd
    import geopandas as gpd
    import libpysal

    # The URL to the CSV file for the Mercer and Hall wheat data
    url = "https://hpc.niasra.uow.edu.au/ckan/dataset/5d8398c1-d090-44f3-acda-cffacb38d80d/resource/9f0d148f-e7c2-4a04-a1e2-62c021966abb/download/wheat.csv"

    # Load the data into a DataFrame
    df = pd.read_csv(url)

    # Display the first few rows
    print(df.head())
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["lon"], df["lat"]))

    gdf["y"] = gdf["yield"] - gdf.groupby("lon")["yield"].transform("median")
    overall_mean = gdf["y"].mean()
    gdf["y"] = gdf["y"] - overall_mean

    g12 = libpysal.graph.Graph.build_knn(gdf, k=12)
    g12 = g12.transform("r")

    # Initialize APLE class
    aple_model = APLE(gdf.y, g12, trace=True)

    # Print the APLE statistic
    print(f"APLE Statistic: {aple_model.statistic_}")

    # Plot the results
    aple_model.plot()
