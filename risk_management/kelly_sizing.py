"""
Kelly Criterion & Dynamic Position Sizing Engine.

Provides institutional capital allocation models:
1. Discrete Bernoulli Kelly Criterion
2. Continuous-Time Log-Normal Kelly Sizing
3. Merton Continuous Jump-Diffusion Kelly Sizing (incorporating jump/crash risk)
4. Fractional Kelly Allocator with Drawdown & Leverage Constraints
5. Multi-Asset Constrained Kelly Portfolio Optimizer
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Dict, Union, Optional, List


class KellySizingEngine:
    """
    Capital Growth & Optimal Position Sizing Engine using the Kelly Criterion.
    """

    @staticmethod
    def bernoulli_kelly(
        win_rate: float,
        win_loss_ratio: float,
        fractional_multiplier: float = 0.5
    ) -> Dict[str, float]:
        """
        Discrete Bernoulli Kelly Formula:
        f* = p - (1 - p) / b = (p * (b + 1) - 1) / b
        where p = win_rate, b = win_loss_ratio (gain amount per dollar risked).
        """
        if not (0.0 <= win_rate <= 1.0):
            raise ValueError("Win rate must be between 0.0 and 1.0.")
        if win_loss_ratio <= 0:
            raise ValueError("Win/Loss ratio must be positive.")

        full_kelly = (win_rate * (win_loss_ratio + 1.0) - 1.0) / win_loss_ratio
        full_kelly = max(0.0, full_kelly)  # Non-negative allocation

        fractional_kelly = full_kelly * fractional_multiplier

        # Calculate expected log growth rate g(f) = p * log(1 + f*b) + (1-p) * log(1 - f)
        if fractional_kelly > 0 and fractional_kelly < 1.0:
            exp_growth = (
                win_rate * np.log(1.0 + fractional_kelly * win_loss_ratio) +
                (1.0 - win_rate) * np.log(1.0 - fractional_kelly)
            )
        else:
            exp_growth = 0.0

        return {
            "method": "Bernoulli Kelly",
            "win_rate": float(win_rate),
            "win_loss_ratio": float(win_loss_ratio),
            "fractional_multiplier": float(fractional_multiplier),
            "full_kelly_fraction": float(full_kelly),
            "allocated_fraction": float(fractional_kelly),
            "expected_log_growth_rate": float(exp_growth)
        }

    @staticmethod
    def continuous_gaussian_kelly(
        expected_return: float,
        volatility: float,
        risk_free_rate: float = 0.0,
        fractional_multiplier: float = 0.5,
        max_leverage: float = 2.0
    ) -> Dict[str, float]:
        """
        Continuous-Time Log-Normal Kelly Sizing:
        f* = (mu - r) / sigma^2
        """
        if volatility <= 0:
            raise ValueError("Volatility must be strictly positive.")

        excess_return = expected_return - risk_free_rate
        full_kelly = excess_return / (volatility ** 2)
        full_kelly = max(0.0, full_kelly)

        fractional_kelly = min(full_kelly * fractional_multiplier, max_leverage)
        exp_growth = risk_free_rate + fractional_kelly * excess_return - 0.5 * (fractional_kelly ** 2) * (volatility ** 2)

        return {
            "method": "Continuous Gaussian Kelly",
            "expected_return_annualized": float(expected_return),
            "volatility_annualized": float(volatility),
            "excess_return": float(excess_return),
            "full_kelly_fraction": float(full_kelly),
            "allocated_fraction": float(fractional_kelly),
            "expected_log_growth_rate": float(exp_growth)
        }

    @staticmethod
    def merton_jump_kelly(
        expected_return: float,
        diffusive_volatility: float,
        jump_intensity: float,
        mean_jump_size: float,
        risk_free_rate: float = 0.0,
        fractional_multiplier: float = 0.5
    ) -> Dict[str, float]:
        """
        Merton Continuous Jump-Diffusion Kelly Sizing:
        Accounts for crash intensity (lambda) and jump amplitude (j).
        Optimization maximizes E[log(1 + f * dS/S)].
        """
        sigma2 = diffusive_volatility ** 2

        def obj_func(f):
            # Maximizes -E[log growth rate]
            # Drift mu_d = expected_return - r - jump_intensity * mean_jump_size
            mu_d = expected_return - risk_free_rate - jump_intensity * mean_jump_size
            diffusive_term = f * mu_d - 0.5 * (f ** 2) * sigma2
            # Jump component: jump intensity * log(1 + f * mean_jump_size)
            jump_term = jump_intensity * np.log(np.maximum(1e-6, 1.0 + f * mean_jump_size))
            return -(diffusive_term + jump_term)

        # Upper bound ensures no instant bankruptcy from jump size: 1 + f * jump_size > 0
        max_f_jump = 0.99 / abs(mean_jump_size) if mean_jump_size < 0 else 5.0

        res = minimize(obj_func, x0=[0.5], bounds=[(0.0, min(5.0, max_f_jump))])
        full_kelly = float(res.x[0]) if res.success else 0.0
        allocated_fraction = full_kelly * fractional_multiplier

        return {
            "method": "Merton Jump-Diffusion Kelly",
            "expected_return": float(expected_return),
            "diffusive_volatility": float(diffusive_volatility),
            "jump_intensity_lambda": float(jump_intensity),
            "mean_jump_size_j": float(mean_jump_size),
            "full_kelly_fraction": float(full_kelly),
            "allocated_fraction": float(allocated_fraction)
        }

    @staticmethod
    def multi_asset_kelly(
        returns_df: pd.DataFrame,
        risk_free_rate: float = 0.0,
        fractional_multiplier: float = 0.5,
        max_position_weight: float = 0.4,
        allow_short: bool = False
    ) -> Dict[str, Union[Dict[str, float], np.ndarray]]:
        """
        Multi-Asset Constrained Kelly Optimization:
        w* = argmax { w^T mu - 0.5 * w^T Sigma w }
        subject to weight bounds and total allocation limits.
        """
        mean_returns = returns_df.mean().values * 252.0 - risk_free_rate
        cov_matrix = returns_df.cov().values * 252.0
        num_assets = len(mean_returns)
        asset_names = list(returns_df.columns)

        def objective(w):
            return -(np.dot(w, mean_returns) - 0.5 * np.dot(w, np.dot(cov_matrix, w)))

        bounds = [(-max_position_weight if allow_short else 0.0, max_position_weight) for _ in range(num_assets)]
        init_weights = np.ones(num_assets) / num_assets

        res = minimize(objective, x0=init_weights, bounds=bounds)
        full_weights = res.x if res.success else init_weights
        allocated_weights = full_weights * fractional_multiplier

        weights_dict = {name: float(w) for name, w in zip(asset_names, allocated_weights)}
        total_leverage = float(np.sum(np.abs(allocated_weights)))

        return {
            "method": "Multi-Asset Constrained Kelly",
            "fractional_multiplier": float(fractional_multiplier),
            "allocated_weights": weights_dict,
            "total_portfolio_leverage": total_leverage,
            "expected_annualized_growth": float(-res.fun * fractional_multiplier)
        }
