"""
Unit Test Suite for Institutional Risk Management Suite.
Verifies all 5 quantitative risk management engines and the master gateway.
"""

import unittest
import numpy as np
import pandas as pd

from risk_management.var_cvar import VaRCVaREngine
from risk_management.kelly_sizing import KellySizingEngine
from risk_management.drawdown_vol_target import DrawdownVolTargetEngine, RiskState
from risk_management.liquidity_stress_regime import LiquidityStressRegimeEngine
from risk_management.hrp_risk_parity import HRPRiskParityEngine
from risk_management.engine import InstitutionalRiskManager


class TestInstitutionalRiskManagement(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        # Generate synthetic asset returns
        n_days = 252
        dates = pd.date_range("2025-01-01", periods=n_days, freq="B")
        spy_ret = np.random.normal(0.0005, 0.012, n_days)
        qqq_ret = 1.2 * spy_ret + np.random.normal(0.0, 0.008, n_days)
        tlt_ret = -0.3 * spy_ret + np.random.normal(0.0, 0.006, n_days)
        gld_ret = 0.1 * spy_ret + np.random.normal(0.0, 0.007, n_days)

        self.returns_df = pd.DataFrame({
            "SPY": spy_ret,
            "QQQ": qqq_ret,
            "TLT": tlt_ret,
            "GLD": gld_ret
        }, index=dates)

        self.single_returns = spy_ret
        self.portfolio_weights = {"SPY": 0.4, "QQQ": 0.3, "TLT": 0.2, "GLD": 0.1}
        self.portfolio_value = 1_000_000.0

    def test_model1_var_cvar_engine(self):
        engine = VaRCVaREngine(confidence_level=0.95, time_horizon_days=1)
        
        # 1. Gaussian
        res_g = engine.parametric_gaussian(self.single_returns, self.portfolio_value)
        self.assertGreater(res_g["dollar_var"], 0)
        self.assertGreater(res_g["dollar_cvar"], res_g["dollar_var"])

        # 2. Student-t
        res_t = engine.parametric_student_t(self.single_returns, self.portfolio_value)
        self.assertGreater(res_t["dollar_var"], 0)

        # 3. Historical
        res_h = engine.historical_simulation(self.single_returns, self.portfolio_value)
        self.assertGreater(res_h["dollar_var"], 0)

        # 4. FHS
        res_fhs = engine.filtered_historical_simulation(self.single_returns, self.portfolio_value)
        self.assertGreater(res_fhs["dollar_var"], 0)

        # 5. Monte Carlo Multi-Asset
        res_mc = engine.monte_carlo_portfolio(self.returns_df, np.array([0.4, 0.3, 0.2, 0.1]), self.portfolio_value)
        self.assertGreater(res_mc["dollar_var"], 0)

        # 6. EVT GPD
        res_evt = engine.extreme_value_theory_gpd(self.single_returns, self.portfolio_value)
        self.assertGreater(res_evt["dollar_var"], 0)

    def test_model2_kelly_sizing_engine(self):
        # 1. Bernoulli
        b_res = KellySizingEngine.bernoulli_kelly(win_rate=0.55, win_loss_ratio=1.5, fractional_multiplier=0.5)
        self.assertGreater(b_res["full_kelly_fraction"], 0)
        self.assertEqual(b_res["allocated_fraction"], b_res["full_kelly_fraction"] * 0.5)

        # 2. Continuous Gaussian
        g_res = KellySizingEngine.continuous_gaussian_kelly(expected_return=0.12, volatility=0.18)
        self.assertGreater(g_res["allocated_fraction"], 0)

        # 3. Merton Jump
        j_res = KellySizingEngine.merton_jump_kelly(
            expected_return=0.10,
            diffusive_volatility=0.15,
            jump_intensity=1.0,
            mean_jump_size=-0.05
        )
        self.assertGreater(j_res["allocated_fraction"], 0)

        # 4. Multi-Asset Kelly
        m_res = KellySizingEngine.multi_asset_kelly(self.returns_df)
        self.assertEqual(len(m_res["allocated_weights"]), 4)

    def test_model3_drawdown_vol_target_engine(self):
        engine = DrawdownVolTargetEngine(
            target_volatility=0.15,
            max_leverage=2.0,
            warning_drawdown=0.08,
            deleveraging_drawdown=0.12,
            hard_halt_drawdown=0.20
        )

        # Volatility Target
        v_res = engine.calculate_volatility_target_weight(self.single_returns)
        self.assertGreater(v_res["leverage_scaling_factor"], 0)

        # State Machine Transitions
        # Normal
        status = engine.update_circuit_breaker(1_000_000.0)
        self.assertEqual(status["risk_state"], "NORMAL")

        # Peak update
        status = engine.update_circuit_breaker(1_200_000.0)
        self.assertEqual(status["peak_equity"], 1_200_000.0)

        # Warning DD (10% drop from 1.2M -> 1.08M)
        status = engine.update_circuit_breaker(1_080_000.0)
        self.assertEqual(status["risk_state"], "WARNING")

        # De-leveraging DD (15% drop from 1.2M -> 1.02M)
        status = engine.update_circuit_breaker(1_020_000.0)
        self.assertEqual(status["risk_state"], "DELEVERAGING")

        # Hard Halt DD (25% drop from 1.2M -> 900k)
        status = engine.update_circuit_breaker(900_000.0)
        self.assertEqual(status["risk_state"], "HARD_HALT")
        self.assertEqual(status["circuit_breaker_leverage_cap"], 0.0)

    def test_model4_liquidity_stress_regime_engine(self):
        spreads = np.full(len(self.single_returns), 0.0015)
        lvar_res = LiquidityStressRegimeEngine.liquidity_adjusted_var(self.single_returns, spreads, self.portfolio_value)
        self.assertGreater(lvar_res["dollar_l_var"], lvar_res["base_dollar_var"])

        regime_res = LiquidityStressRegimeEngine.regime_aware_risk_adjustment(self.single_returns)
        self.assertIn("detected_regime", regime_res)

        stress_df = LiquidityStressRegimeEngine.historical_macro_stress_test(self.portfolio_weights, self.portfolio_value)
        self.assertEqual(len(stress_df), 4)  # 4 built-in scenarios

    def test_model5_hrp_risk_parity_engine(self):
        hrp_engine = HRPRiskParityEngine()
        
        # 1. HRP Weights
        hrp_res = hrp_engine.hierarchical_risk_parity(self.returns_df)
        self.assertAlmostEqual(sum(hrp_res["weights"].values()), 1.0, places=4)

        # 2. Equal Risk Contribution (ERC)
        erc_res = hrp_engine.equal_risk_contribution(self.returns_df)
        self.assertAlmostEqual(sum(erc_res["weights"].values()), 1.0, places=4)

    def test_master_institutional_risk_manager(self):
        manager = InstitutionalRiskManager()
        
        # Pre-trade check pass
        trade_check = manager.evaluate_pre_trade_risk("SPY", 100_000.0, self.portfolio_value, self.single_returns)
        self.assertTrue(trade_check["passed"])

        # Pre-trade check fail (exceeds position limit)
        trade_fail = manager.evaluate_pre_trade_risk("SPY", 500_000.0, self.portfolio_value, self.single_returns)
        self.assertFalse(trade_fail["passed"])

        # Full Audit
        audit = manager.run_full_institutional_risk_audit(self.returns_df, self.portfolio_weights, self.portfolio_value)
        self.assertIn("var_cvar_summary", audit)
        self.assertIn("hrp_allocated_weights", audit)


if __name__ == "__main__":
    unittest.main()
