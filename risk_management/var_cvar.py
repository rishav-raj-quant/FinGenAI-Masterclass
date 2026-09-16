"""
Value at Risk (VaR) and Expected Shortfall (CVaR / Conditional VaR) Engine.

Provides institutional-grade implementations for single-asset and multi-asset portfolios:
1. Parametric Gaussian VaR & CVaR
2. Parametric Student-t Fat-Tailed VaR & CVaR
3. Historical Simulation VaR & CVaR
4. Filtered Historical Simulation (FHS with EWMA Volatility Scaling)
5. Monte Carlo Simulation VaR & CVaR (Correlated Multi-Asset via Cholesky Factorization)
6. Extreme Value Theory (EVT) Generalized Pareto Distribution (GPD) Tail Modeling
"""

import numpy as np
import pandas as pd
from scipy.stats import norm, t, genpareto
from typing import Dict, Union, Tuple, Optional


class VaRCVaREngine:
    """
    Quantitative Risk Engine for computing Value at Risk (VaR) and Expected Shortfall (CVaR).
    All outputs represent dollar loss or percentage loss (positive numbers denote loss).
    """

    def __init__(self, confidence_level: float = 0.95, time_horizon_days: int = 1):
        """
        Parameters:
        -----------
        confidence_level : float
            Confidence level (e.g. 0.95 for 95% VaR, 0.99 for 99% VaR).
        time_horizon_days : int
            Holding period horizon in days (e.g., 1 day, 10 days).
        """
        if not (0.5 < confidence_level < 1.0):
            raise ValueError("Confidence level must be between 0.5 and 1.0 (e.g., 0.95 or 0.99).")
        if time_horizon_days < 1:
            raise ValueError("Time horizon must be at least 1 day.")

        self.alpha = 1.0 - confidence_level
        self.confidence_level = confidence_level
        self.horizon = time_horizon_days
        self.sqrt_h = np.sqrt(time_horizon_days)

    def parametric_gaussian(
        self,
        returns: Union[np.ndarray, pd.Series],
        portfolio_value: float = 1_000_000.0
    ) -> Dict[str, float]:
        """
        Computes Parametric Gaussian VaR and CVaR.

        Formula:
        VaR_alpha = - (mu * h + z_alpha * sigma * sqrt(h)) * Portfolio_Value
        CVaR_alpha = - (mu * h - (pdf(z_alpha) / alpha) * sigma * sqrt(h)) * Portfolio_Value
        """
        returns_arr = np.asarray(returns, dtype=float)
        mu = np.mean(returns_arr)
        sigma = np.std(returns_arr, ddof=1)

        z_alpha = norm.ppf(self.alpha)
        pdf_z = norm.pdf(z_alpha)

        pct_var = -(mu * self.horizon + z_alpha * sigma * self.sqrt_h)
        pct_cvar = -(mu * self.horizon - (pdf_z / self.alpha) * sigma * self.sqrt_h)

        return {
            "method": "Parametric Gaussian",
            "confidence_level": self.confidence_level,
            "horizon_days": self.horizon,
            "pct_var": float(pct_var),
            "pct_cvar": float(pct_cvar),
            "dollar_var": float(pct_var * portfolio_value),
            "dollar_cvar": float(pct_cvar * portfolio_value),
            "mean_daily": float(mu),
            "std_daily": float(sigma)
        }

    def parametric_student_t(
        self,
        returns: Union[np.ndarray, pd.Series],
        portfolio_value: float = 1_000_000.0
    ) -> Dict[str, float]:
        """
        Computes Parametric Student-t VaR and CVaR to account for fat tails / kurtosis.
        """
        returns_arr = np.asarray(returns, dtype=float)
        df, loc, scale = t.fit(returns_arr)
        if df <= 2:
            df = 2.01  # Ensure finite variance

        t_alpha = t.ppf(self.alpha, df)
        pdf_t = t.pdf(t_alpha, df)

        pct_var = -(loc * self.horizon + t_alpha * scale * self.sqrt_h)
        cvar_factor = (pdf_t / self.alpha) * ((df + t_alpha**2) / (df - 1))
        pct_cvar = -(loc * self.horizon - cvar_factor * scale * self.sqrt_h)

        return {
            "method": "Parametric Student-t",
            "confidence_level": self.confidence_level,
            "horizon_days": self.horizon,
            "degrees_of_freedom": float(df),
            "pct_var": float(pct_var),
            "pct_cvar": float(pct_cvar),
            "dollar_var": float(pct_var * portfolio_value),
            "dollar_cvar": float(pct_cvar * portfolio_value)
        }

    def historical_simulation(
        self,
        returns: Union[np.ndarray, pd.Series],
        portfolio_value: float = 1_000_000.0
    ) -> Dict[str, float]:
        """
        Computes Historical Simulation VaR and CVaR using empirical quantile loss.
        """
        returns_arr = np.asarray(returns, dtype=float)
        losses = -returns_arr * self.sqrt_h  # Loss convention

        pct_var = np.percentile(losses, (1.0 - self.alpha) * 100)
        tail_losses = losses[losses >= pct_var]
        pct_cvar = np.mean(tail_losses) if len(tail_losses) > 0 else pct_var

        return {
            "method": "Historical Simulation",
            "confidence_level": self.confidence_level,
            "horizon_days": self.horizon,
            "pct_var": float(pct_var),
            "pct_cvar": float(pct_cvar),
            "dollar_var": float(pct_var * portfolio_value),
            "dollar_cvar": float(pct_cvar * portfolio_value),
            "sample_size": len(returns_arr)
        }

    def filtered_historical_simulation(
        self,
        returns: Union[np.ndarray, pd.Series],
        portfolio_value: float = 1_000_000.0,
        decay_factor: float = 0.94
    ) -> Dict[str, float]:
        """
        Computes Filtered Historical Simulation (FHS) VaR and CVaR.
        Scales historical returns by EWMA volatility relative to current volatility.
        """
        returns_arr = np.asarray(returns, dtype=float)
        T = len(returns_arr)

        # EWMA Volatility estimation
        variances = np.zeros(T)
        variances[0] = np.var(returns_arr)
        for t_idx in range(1, T):
            variances[t_idx] = decay_factor * variances[t_idx - 1] + (1 - decay_factor) * (returns_arr[t_idx - 1] ** 2)

        volatilities = np.sqrt(variances)
        current_vol = volatilities[-1]

        # Standardized residuals scaled to current volatility regime
        standardized_returns = returns_arr / np.maximum(volatilities, 1e-8)
        scaled_returns = standardized_returns * current_vol
        losses = -scaled_returns * self.sqrt_h

        pct_var = np.percentile(losses, (1.0 - self.alpha) * 100)
        tail_losses = losses[losses >= pct_var]
        pct_cvar = np.mean(tail_losses) if len(tail_losses) > 0 else pct_var

        return {
            "method": "Filtered Historical Simulation (FHS-EWMA)",
            "confidence_level": self.confidence_level,
            "horizon_days": self.horizon,
            "current_volatility_annualized": float(current_vol * np.sqrt(252)),
            "pct_var": float(pct_var),
            "pct_cvar": float(pct_cvar),
            "dollar_var": float(pct_var * portfolio_value),
            "dollar_cvar": float(pct_cvar * portfolio_value)
        }

    def monte_carlo_portfolio(
        self,
        returns_df: pd.DataFrame,
        weights: np.ndarray,
        portfolio_value: float = 1_000_000.0,
        num_simulations: int = 50_000,
        random_seed: int = 42
    ) -> Dict[str, float]:
        """
        Computes Correlated Multi-Asset Monte Carlo Portfolio VaR and CVaR using Cholesky Decomposition.
        """
        np.random.seed(random_seed)
        weights = np.asarray(weights, dtype=float)
        weights = weights / np.sum(weights)

        mean_returns = returns_df.mean().values
        cov_matrix = returns_df.cov().values

        # Ensure Positive Semi-Definite for Cholesky
        min_eig = np.min(np.real(np.linalg.eigvals(cov_matrix)))
        if min_eig < 0:
            cov_matrix -= 1.1 * min_eig * np.eye(cov_matrix.shape[0])

        L = np.linalg.cholesky(cov_matrix)
        num_assets = len(weights)

        # Generate correlated standard normal random variables
        Z = np.random.normal(size=(num_simulations, num_assets))
        simulated_asset_returns = mean_returns * self.horizon + (Z @ L.T) * self.sqrt_h

        # Compute portfolio returns and losses
        simulated_port_returns = simulated_asset_returns @ weights
        losses = -simulated_port_returns

        pct_var = np.percentile(losses, (1.0 - self.alpha) * 100)
        tail_losses = losses[losses >= pct_var]
        pct_cvar = np.mean(tail_losses) if len(tail_losses) > 0 else pct_var

        return {
            "method": "Monte Carlo Multi-Asset Simulation",
            "confidence_level": self.confidence_level,
            "horizon_days": self.horizon,
            "num_simulations": num_simulations,
            "num_assets": num_assets,
            "pct_var": float(pct_var),
            "pct_cvar": float(pct_cvar),
            "dollar_var": float(pct_var * portfolio_value),
            "dollar_cvar": float(pct_cvar * portfolio_value)
        }

    def extreme_value_theory_gpd(
        self,
        returns: Union[np.ndarray, pd.Series],
        portfolio_value: float = 1_000_000.0,
        threshold_quantile: float = 0.90
    ) -> Dict[str, float]:
        """
        Computes Extreme Value Theory (EVT) VaR and CVaR fitting a Generalized Pareto Distribution (GPD)
        to tail losses exceeding the specified threshold quantile.
        """
        returns_arr = np.asarray(returns, dtype=float)
        losses = -returns_arr * self.sqrt_h

        threshold = np.percentile(losses, threshold_quantile * 100)
        exceedances = losses[losses > threshold] - threshold
        N = len(losses)
        N_u = len(exceedances)

        if N_u < 10:
            # Fallback to historical if insufficient tail exceedances
            return self.historical_simulation(returns, portfolio_value)

        # Fit Generalized Pareto Distribution (GPD): c=shape (xi), loc=0, scale=sigma_gpd
        c_shape, loc_gpd, scale_gpd = genpareto.fit(exceedances, floc=0)

        # EVT-GPD VaR formula: u + (scale / c) * [ ((N / N_u) * (1 - confidence))^(-c) - 1 ]
        prob_tail = 1.0 - self.confidence_level
        if abs(c_shape) < 1e-6:
            pct_var = threshold - scale_gpd * np.log((N / N_u) * prob_tail)
            pct_cvar = pct_var + scale_gpd
        else:
            pct_var = threshold + (scale_gpd / c_shape) * (((N / N_u) * prob_tail) ** (-c_shape) - 1.0)
            pct_cvar = (pct_var + scale_gpd - c_shape * threshold) / (1.0 - c_shape)

        return {
            "method": "Extreme Value Theory (EVT-GPD)",
            "confidence_level": self.confidence_level,
            "horizon_days": self.horizon,
            "threshold_quantile": threshold_quantile,
            "gpd_shape_xi": float(c_shape),
            "gpd_scale_beta": float(scale_gpd),
            "num_exceedances": int(N_u),
            "pct_var": float(pct_var),
            "pct_cvar": float(pct_cvar),
            "dollar_var": float(pct_var * portfolio_value),
            "dollar_cvar": float(pct_cvar * portfolio_value)
        }
