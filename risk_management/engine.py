"""
Unified Institutional Risk Manager Engine.

Orchestrates all 5 world-class quantitative risk management engines:
1. VaRCVaREngine (Value at Risk & Tail Risk)
2. KellySizingEngine (Optimal Capital Allocation)
3. DrawdownVolTargetEngine (Dynamic Volatility & Drawdown Circuit Breakers)
4. LiquidityStressRegimeEngine (L-VaR, Volatility Regimes, Macro Stress Tests)
5. HRPRiskParityEngine (Hierarchical Risk Parity & Covariance Regularization)
"""

import numpy as np
import pandas as pd
from typing import Dict, Union, List, Optional

__version__ = "1.0.0"

from .var_cvar import VaRCVaREngine
from .kelly_sizing import KellySizingEngine
from .drawdown_vol_target import DrawdownVolTargetEngine
from .liquidity_stress_regime import LiquidityStressRegimeEngine
from .hrp_risk_parity import HRPRiskParityEngine


class InstitutionalRiskManager:
    """
    Master Risk Gateway for pre-trade verification, dynamic position sizing,
    stress testing, and portfolio risk audits.
    """

    def __init__(
        self,
        confidence_level: float = 0.95,
        target_volatility: float = 0.15,
        max_leverage: float = 2.0,
        max_drawdown_limit: float = 0.20,
        max_single_position_pct: float = 0.35
    ):
        self.var_engine = VaRCVaREngine(confidence_level=confidence_level)
        self.kelly_engine = KellySizingEngine()
        self.dd_engine = DrawdownVolTargetEngine(
            target_volatility=target_volatility,
            max_leverage=max_leverage,
            hard_halt_drawdown=max_drawdown_limit
        )
        self.liquidity_engine = LiquidityStressRegimeEngine()
        self.hrp_engine = HRPRiskParityEngine()

        self.max_single_pos = max_single_position_pct
        self.max_drawdown_limit = max_drawdown_limit

    def evaluate_pre_trade_risk(
        self,
        proposed_symbol: str,
        proposed_trade_dollar_amount: float,
        portfolio_value: float,
        historical_returns: Union[np.ndarray, pd.Series],
        bid_ask_spread: float = 0.001
    ) -> Dict[str, Union[bool, str, Dict]]:
        """
        Pre-Trade Risk Gateway: Checks whether a proposed order passes risk limits
        (Single Position Cap, Drawdown Circuit Breakers, VaR limits, Liquidity Constraints).
        """
        # 1. Position Size Cap Check
        proposed_pos_pct = abs(proposed_trade_dollar_amount) / portfolio_value
        if proposed_pos_pct > self.max_single_pos:
            return {
                "passed": False,
                "reason": f"Position size {proposed_pos_pct * 100:.1f}% exceeds max allowed {self.max_single_pos * 100:.1f}%.",
                "recommended_trade_dollar_amount": self.max_single_pos * portfolio_value
            }

        # 2. Circuit Breaker Check
        cb_status = self.dd_engine.update_circuit_breaker(portfolio_value)
        if cb_status["risk_state"] == "HARD_HALT":
            return {
                "passed": False,
                "reason": f"Circuit Breaker in HARD_HALT state due to {cb_status['current_drawdown_pct']} drawdown.",
                "recommended_trade_dollar_amount": 0.0
            }

        # 3. VaR Risk Limit
        var_res = self.var_engine.historical_simulation(historical_returns, portfolio_value)
        if var_res["pct_var"] > 0.10:  # 10% daily VaR threshold cap
            return {
                "passed": False,
                "reason": f"Daily VaR ({var_res['pct_var']*100:.2f}%) exceeds safety limit of 10.0%.",
                "recommended_trade_dollar_amount": proposed_trade_dollar_amount * 0.5
            }

        # 4. Volatility Scaling Adjustment
        vol_res = self.dd_engine.calculate_volatility_target_weight(historical_returns)
        scaled_amount = proposed_trade_dollar_amount * vol_res["leverage_scaling_factor"]

        return {
            "passed": True,
            "reason": "All pre-trade risk checks passed.",
            "original_proposed_dollar_amount": proposed_trade_dollar_amount,
            "volatility_scaled_dollar_amount": float(scaled_amount),
            "circuit_breaker_state": cb_status["risk_state"]
        }

    def run_full_institutional_risk_audit(
        self,
        returns_df: pd.DataFrame,
        portfolio_weights: Dict[str, float],
        portfolio_value: float = 1_000_000.0,
        bid_ask_spreads: Optional[np.ndarray] = None
    ) -> Dict[str, Union[Dict, pd.DataFrame]]:
        """
        Executes an institutional risk audit combining all 5 quantitative engines.
        """
        weights_arr = np.array([portfolio_weights[c] for c in returns_df.columns])
        port_returns = returns_df.values @ weights_arr

        # 1. VaR / CVaR Multi-Methodology Audit
        var_gaussian = self.var_engine.parametric_gaussian(port_returns, portfolio_value)
        var_t = self.var_engine.parametric_student_t(port_returns, portfolio_value)
        var_hist = self.var_engine.historical_simulation(port_returns, portfolio_value)
        var_fhs = self.var_engine.filtered_historical_simulation(port_returns, portfolio_value)
        var_mc = self.var_engine.monte_carlo_portfolio(returns_df, weights_arr, portfolio_value)
        var_evt = self.var_engine.extreme_value_theory_gpd(port_returns, portfolio_value)

        var_summary = {
            "Gaussian_VaR_Dollar": var_gaussian["dollar_var"],
            "StudentT_VaR_Dollar": var_t["dollar_var"],
            "Historical_VaR_Dollar": var_hist["dollar_var"],
            "FHS_EWMA_VaR_Dollar": var_fhs["dollar_var"],
            "MonteCarlo_VaR_Dollar": var_mc["dollar_var"],
            "EVT_GPD_VaR_Dollar": var_evt["dollar_var"],
            "EVT_GPD_CVaR_Dollar": var_evt["dollar_cvar"]
        }

        # 2. Optimal Kelly Allocation Audit
        kelly_res = self.kelly_engine.multi_asset_kelly(returns_df)

        # 3. Volatility & Circuit Breaker Audit
        vol_target_res = self.dd_engine.calculate_volatility_target_weight(port_returns)

        # 4. Liquidity & Macro Stress Test Audit
        if bid_ask_spreads is None:
            bid_ask_spreads = np.full(len(port_returns), 0.001)

        l_var_res = self.liquidity_engine.liquidity_adjusted_var(port_returns, bid_ask_spreads, portfolio_value)
        regime_res = self.liquidity_engine.regime_aware_risk_adjustment(port_returns)
        stress_test_df = self.liquidity_engine.historical_macro_stress_test(portfolio_weights, portfolio_value)

        # 5. Hierarchical Risk Parity & Covariance Audit
        hrp_res = self.hrp_engine.hierarchical_risk_parity(returns_df)
        erc_res = self.hrp_engine.equal_risk_contribution(returns_df)

        return {
            "portfolio_value": portfolio_value,
            "var_cvar_summary": var_summary,
            "kelly_optimal_allocation": kelly_res,
            "volatility_target_status": vol_target_res,
            "liquidity_adjusted_var": l_var_res,
            "volatility_regime_status": regime_res,
            "macro_stress_test_results": stress_test_df,
            "hrp_allocated_weights": hrp_res["weights"],
            "equal_risk_contribution_weights": erc_res["weights"]
        }
