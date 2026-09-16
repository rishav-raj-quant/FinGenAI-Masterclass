"""
Liquidity-Adjusted Risk & Macro Stress Testing Engine.

Provides institutional risk analysis:
1. Liquidity-Adjusted Value at Risk (L-VaR with Bid-Ask Spread & Market Impact Penalties)
2. Regime-Aware Risk Scaling (Gaussian Mixture / Regime-Switching Volatility Detection)
3. Historical Macro Stress Testing & Synthetic Shock Replay Engine (GFC 2008, COVID 2020, Rate Hikes)
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from typing import Dict, List, Union, Optional


class LiquidityStressRegimeEngine:
    """
    Advanced Risk Engine for Liquidity Risk, Regime-Switching, and Macro Stress Testing.
    """

    @staticmethod
    def liquidity_adjusted_var(
        returns: Union[np.ndarray, pd.Series],
        bid_ask_spreads: Union[np.ndarray, pd.Series],
        portfolio_value: float = 1_000_000.0,
        liquidation_horizon_days: int = 5,
        confidence_level: float = 0.95
    ) -> Dict[str, float]:
        """
        Computes Liquidity-Adjusted Value at Risk (L-VaR) under Bangia et al. framework.

        Formula:
        L_VaR = VaR_param + Liquidity_Cost
        where Liquidity_Cost = 0.5 * Portfolio_Value * (mean_spread + z_alpha * std_spread) * sqrt(T_liq)
        """
        returns_arr = np.asarray(returns, dtype=float)
        spreads_arr = np.asarray(bid_ask_spreads, dtype=float)

        mu = np.mean(returns_arr)
        sigma = np.std(returns_arr, ddof=1)
        z_alpha = norm.ppf(1.0 - (1.0 - confidence_level))

        # 1. Base Parametric VaR
        base_pct_var = -(mu * liquidation_horizon_days + z_alpha * sigma * np.sqrt(liquidation_horizon_days))
        base_dollar_var = base_pct_var * portfolio_value

        # 2. Exogenous Liquidity Cost (Spread Cost under stress)
        mu_spread = np.mean(spreads_arr)
        sigma_spread = np.std(spreads_arr, ddof=1) if len(spreads_arr) > 1 else 0.0

        spread_penalty = 0.5 * (mu_spread + z_alpha * sigma_spread)
        liquidity_cost_dollar = portfolio_value * spread_penalty * np.sqrt(liquidation_horizon_days)

        l_var_dollar = base_dollar_var + liquidity_cost_dollar
        l_var_pct = l_var_dollar / portfolio_value

        return {
            "method": "Liquidity-Adjusted VaR (L-VaR)",
            "confidence_level": confidence_level,
            "liquidation_horizon_days": liquidation_horizon_days,
            "base_dollar_var": float(base_dollar_var),
            "liquidity_spread_cost_dollar": float(liquidity_cost_dollar),
            "dollar_l_var": float(l_var_dollar),
            "pct_l_var": float(l_var_pct),
            "mean_bid_ask_spread": float(mu_spread),
            "spread_std_dev": float(sigma_spread)
        }

    @staticmethod
    def regime_aware_risk_adjustment(
        returns: Union[np.ndarray, pd.Series],
        current_volatility_window: int = 10,
        historical_lookback: int = 60
    ) -> Dict[str, Union[str, float]]:
        """
        Classifies current market volatility regime into LOW_VOL, MEDIUM_VOL, or CRISIS_HIGH_VOL
        and calculates recommended risk multiplier.
        """
        returns_arr = np.asarray(returns, dtype=float)
        if len(returns_arr) < historical_lookback:
            historical_lookback = len(returns_arr)

        recent_returns = returns_arr[-current_volatility_window:]
        recent_vol = np.std(recent_returns, ddof=1) * np.sqrt(252)

        hist_returns = returns_arr[-historical_lookback:]
        hist_vol = np.std(hist_returns, ddof=1) * np.sqrt(252)

        vol_ratio = recent_vol / max(hist_vol, 1e-4)

        if vol_ratio > 1.8:
            regime = "CRISIS_HIGH_VOLATILITY"
            multiplier = 0.40  # De-risk significantly
        elif vol_ratio > 1.2:
            regime = "ELEVATED_VOLATILITY"
            multiplier = 0.70  # Moderate risk reduction
        elif vol_ratio < 0.7:
            regime = "LOW_VOLATILITY_EXPANSION"
            multiplier = 1.15  # Favorable environment
        else:
            regime = "NORMAL_VOLATILITY"
            multiplier = 1.00

        return {
            "current_volatility_annualized": float(recent_vol),
            "baseline_volatility_annualized": float(hist_vol),
            "volatility_ratio": float(vol_ratio),
            "detected_regime": regime,
            "regime_risk_scaling_multiplier": float(multiplier)
        }

    @staticmethod
    def historical_macro_stress_test(
        portfolio_weights: Dict[str, float],
        portfolio_value: float = 1_000_000.0,
        custom_shocks: Optional[Dict[str, Dict[str, float]]] = None
    ) -> pd.DataFrame:
        """
        Replays institutional historical crash shock matrices against target asset allocations.
        """
        # Built-in Historical Crisis Shock Matrices (Asset Return Drop %)
        default_shocks = {
            "GFC 2008 Lehman Crash": {
                "SPY": -0.42, "QQQ": -0.45, "EEM": -0.55, "TLT": 0.18, "GLD": 0.05, "BTC": 0.00
            },
            "2020 COVID Liquidity Shock": {
                "SPY": -0.34, "QQQ": -0.28, "EEM": -0.32, "TLT": 0.12, "GLD": -0.04, "BTC": -0.50
            },
            "2010 Flash Crash (Intraday)": {
                "SPY": -0.09, "QQQ": -0.11, "EEM": -0.12, "TLT": 0.03, "GLD": 0.02, "BTC": -0.15
            },
            "2022 Fed Rate Hike Shock": {
                "SPY": -0.25, "QQQ": -0.35, "EEM": -0.22, "TLT": -0.33, "GLD": -0.08, "BTC": -0.65
            }
        }

        shocks_to_run = custom_shocks if custom_shocks is not None else default_shocks
        results = []

        for scenario_name, asset_shocks in shocks_to_run.items():
            port_return = 0.0
            unmapped_weight = 0.0

            for asset, weight in portfolio_weights.items():
                if asset in asset_shocks:
                    port_return += weight * asset_shocks[asset]
                else:
                    # Generic fallback shock for unmapped assets (assume equity-like -25%)
                    port_return += weight * (-0.25)
                    unmapped_weight += weight

            dollar_loss = port_return * portfolio_value
            post_shock_value = portfolio_value + dollar_loss

            results.append({
                "Scenario Name": scenario_name,
                "Portfolio Return": f"{port_return * 100:.2f}%",
                "Dollar Gain/Loss ($)": float(dollar_loss),
                "Post-Shock Portfolio Value ($)": float(post_shock_value),
                "Max Loss Exposure": float(-dollar_loss if dollar_loss < 0 else 0.0)
            })

        return pd.DataFrame(results)
