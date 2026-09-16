"""
Institutional Risk Management Suite for Quantitative Investing & Algorithmic Trading.

Modules:
1. Value at Risk & Expected Shortfall (var_cvar)
2. Kelly Criterion & Position Sizing (kelly_sizing)
3. Dynamic Drawdown Control & Volatility Targeting (drawdown_vol_target)
4. Liquidity-Adjusted Risk & Macro Stress Testing (liquidity_stress_regime)
5. Hierarchical Risk Parity & Covariance Regularization (hrp_risk_parity)
6. Unified Institutional Risk Manager Engine (engine)
"""

from .var_cvar import VaRCVaREngine
from .kelly_sizing import KellySizingEngine
from .drawdown_vol_target import DrawdownVolTargetEngine
from .liquidity_stress_regime import LiquidityStressRegimeEngine
from .hrp_risk_parity import HRPRiskParityEngine
from .engine import InstitutionalRiskManager

__all__ = [
    "VaRCVaREngine",
    "KellySizingEngine",
    "DrawdownVolTargetEngine",
    "LiquidityStressRegimeEngine",
    "HRPRiskParityEngine",
    "InstitutionalRiskManager",
]
