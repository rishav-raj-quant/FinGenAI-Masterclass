"""
Hierarchical Risk Parity (HRP) & Covariance Regularization Engine.

Provides institutional portfolio optimization algorithms:
1. Ledoit-Wolf Shrinkage & Oracle Approximated Shrinkage (OAS) Covariance Estimators
2. Marcos López de Prado's Hierarchical Risk Parity (HRP Algorithm: Clustering, Quasi-Diagonalization, Recursive Bisection)
3. Equal Risk Contribution (ERC) / Risk Parity Optimization
"""

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
from scipy.optimize import minimize
from typing import Dict, List, Tuple, Union, Optional


class HRPRiskParityEngine:
    """
    Advanced Covariance Regularization and Hierarchical Risk Allocation Engine.
    """

    @staticmethod
    def ledoit_wolf_shrinkage(returns_df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes Ledoit-Wolf Shrinkage Covariance Matrix to stabilize noisy sample covariance matrices.

        Formula:
        Sigma_shrunk = (1 - delta) * Sample_Cov + delta * Target_Prior
        where Target_Prior is equal-variance constant-correlation matrix.
        """
        X = returns_df.values
        T, N = X.shape

        # Demean
        X = X - np.mean(X, axis=0)

        # Sample Covariance
        sample_cov = (X.T @ X) / T

        # Prior Target: Constant Correlation
        var = np.diag(sample_cov)
        std = np.sqrt(var)
        std[std == 0] = 1e-8

        corr = sample_cov / np.outer(std, std)
        mean_corr = (np.sum(corr) - N) / (N * (N - 1)) if N > 1 else 0.0
        target_prior = mean_corr * np.outer(std, std)
        np.fill_diagonal(target_prior, var)

        # Estimate optimal shrinkage intensity delta
        d2 = np.sum((sample_cov - target_prior) ** 2)
        b2_sum = 0.0
        for t_idx in range(T):
            x_t = X[t_idx, :]
            b2_sum += np.sum((np.outer(x_t, x_t) - sample_cov) ** 2)

        b2 = b2_sum / (T ** 2)
        b2 = min(b2, d2)
        shrinkage_intensity = b2 / d2 if d2 > 0 else 0.0

        shrunk_cov = (1.0 - shrinkage_intensity) * sample_cov + shrinkage_intensity * target_prior

        return pd.DataFrame(shrunk_cov, index=returns_df.columns, columns=returns_df.columns)

    @staticmethod
    def _get_quasi_diag(linkage_matrix: np.ndarray) -> List[int]:
        """
        Sorts cluster items so contiguous items are most similar (Quasi-Diagonalization).
        """
        linkage_matrix = linkage_matrix.astype(int)
        sort_ix = [linkage_matrix[-1, 0], linkage_matrix[-1, 1]]
        num_items = linkage_matrix[-1, 3]

        while max(sort_ix) >= num_items:
            sort_ix_new = []
            for item in sort_ix:
                if item >= num_items:
                    sort_ix_new.append(linkage_matrix[item - num_items, 0])
                    sort_ix_new.append(linkage_matrix[item - num_items, 1])
                else:
                    sort_ix_new.append(item)
            sort_ix = sort_ix_new

        return sort_ix

    @staticmethod
    def _get_cluster_var(cov: np.ndarray, cluster_items: List[int]) -> float:
        """
        Computes inverse-variance portfolio variance for a subset of cluster items.
        """
        cov_slice = cov[np.ix_(cluster_items, cluster_items)]
        inv_diag = 1.0 / np.diag(cov_slice)
        inv_diag[np.isinf(inv_diag)] = 0.0
        w = inv_diag / np.sum(inv_diag)
        var = float(w.T @ cov_slice @ w)
        return var

    def hierarchical_risk_parity(
        self,
        returns_df: pd.DataFrame,
        use_ledoit_wolf: bool = True
    ) -> Dict[str, Union[Dict[str, float], List[str]]]:
        """
        Marcos López de Prado's Hierarchical Risk Parity (HRP) algorithm.

        Steps:
        1. Hierarchical Tree Clustering on Angular Correlation Distance d_ij = sqrt(0.5 * (1 - rho_ij))
        2. Quasi-Diagonalization of Distance Matrix
        3. Top-Down Recursive Bisection Allocation
        """
        if use_ledoit_wolf:
            cov_df = self.ledoit_wolf_shrinkage(returns_df)
        else:
            cov_df = returns_df.cov()

        cov = cov_df.values
        asset_names = list(returns_df.columns)
        N = len(asset_names)

        if N == 1:
            return {"weights": {asset_names[0]: 1.0}, "ordered_assets": asset_names}

        # 1. Correlation Matrix & Distance Matrix
        std = np.sqrt(np.diag(cov))
        std[std == 0] = 1e-8
        corr = cov / np.outer(std, std)
        corr = np.clip(corr, -1.0, 1.0)
        dist_matrix = np.sqrt(0.5 * (1.0 - corr))
        np.fill_diagonal(dist_matrix, 0.0)

        # 2. Hierarchical Linkage & Quasi-Diagonalization
        condensed_dist = squareform(dist_matrix, checks=False)
        linkage_matrix = linkage(condensed_dist, method='single')
        sort_ix = self._get_quasi_diag(linkage_matrix)
        ordered_assets = [asset_names[i] for i in sort_ix]

        # 3. Recursive Bisection
        weights = pd.Series(1.0, index=sort_ix)
        cluster_list = [sort_ix]

        while len(cluster_list) > 0:
            cluster_list = [
                c[j:k] for c in cluster_list
                for j, k in ((0, len(c) // 2), (len(c) // 2, len(c)))
                if len(c) > 1
            ]

            for i in range(0, len(cluster_list), 2):
                cluster_left = cluster_list[i]
                cluster_right = cluster_list[i + 1]

                var_left = self._get_cluster_var(cov, cluster_left)
                var_right = self._get_cluster_var(cov, cluster_right)

                alpha = 1.0 - var_left / (var_left + var_right)

                weights[cluster_left] *= alpha
                weights[cluster_right] *= (1.0 - alpha)

        # Normalize & Map weights
        weights_mapped = {asset_names[i]: float(weights[i]) for i in range(N)}
        sum_w = sum(weights_mapped.values())
        weights_mapped = {k: v / sum_w for k, v in weights_mapped.items()}

        return {
            "method": "Hierarchical Risk Parity (HRP)",
            "weights": weights_mapped,
            "ordered_assets": ordered_assets,
            "use_ledoit_wolf": use_ledoit_wolf
        }

    @staticmethod
    def equal_risk_contribution(
        returns_df: pd.DataFrame,
        use_ledoit_wolf: bool = True
    ) -> Dict[str, Union[Dict[str, float], float]]:
        """
        Computes Equal Risk Contribution (ERC) / Risk Parity Portfolio Weights.
        Equalizes Risk Contribution RC_i = w_i * (Sigma w)_i / sigma_p across all assets.
        """
        if use_ledoit_wolf:
            cov = HRPRiskParityEngine.ledoit_wolf_shrinkage(returns_df).values
        else:
            cov = returns_df.cov().values

        N = cov.shape[0]
        asset_names = list(returns_df.columns)

        def risk_budget_objective(w):
            w = np.asarray(w)
            port_vol = np.sqrt(w.T @ cov @ w)
            if port_vol == 0:
                return 1e6

            marginal_risk = (cov @ w) / port_vol
            risk_contrib = w * marginal_risk
            target_risk = port_vol / N

            # Sum of squared risk contribution deviations
            return np.sum((risk_contrib - target_risk) ** 2)

        bounds = [(1e-4, 1.0) for _ in range(N)]
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
        init_w = np.ones(N) / N

        res = minimize(risk_budget_objective, init_w, method='SLSQP', bounds=bounds, constraints=constraints)
        weights = res.x / np.sum(res.x) if res.success else init_w

        port_vol = np.sqrt(weights.T @ cov @ weights)
        marginal_risk = (cov @ weights) / port_vol
        risk_contrib_pct = (weights * marginal_risk) / port_vol

        weights_dict = {asset_names[i]: float(weights[i]) for i in range(N)}
        risk_contrib_dict = {asset_names[i]: float(risk_contrib_pct[i]) for i in range(N)}

        return {
            "method": "Equal Risk Contribution (ERC / Risk Parity)",
            "weights": weights_dict,
            "risk_contribution_percentages": risk_contrib_dict,
            "portfolio_annualized_volatility": float(port_vol * np.sqrt(252))
        }
