"""
Dynamic Drawdown Control & Volatility Targeting Engine.

Provides institutional risk management mechanisms:
1. Volatility Targeting & Dynamic Leverage Scaling
2. Maximum Drawdown (MDD) Circuit Breaker State Machine
3. ATR Volatility Trailing Stop-Loss & High-Water Mark (HWM) Floor Engine
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Union, Optional
from enum import Enum


class RiskState(Enum):
    NORMAL = "NORMAL"
    WARNING = "WARNING"
    DELEVERAGING = "DELEVERAGING"
    HARD_HALT = "HARD_HALT"
    RECOVERY = "RECOVERY"


class DrawdownVolTargetEngine:
    """
    Real-time dynamic volatility targeting and dynamic drawdown protection circuit breakers.
    """

    def __init__(
        self,
        target_volatility: float = 0.15,
        max_leverage: float = 2.0,
        warning_drawdown: float = 0.10,
        deleveraging_drawdown: float = 0.15,
        hard_halt_drawdown: float = 0.20,
        lookback_window: int = 20
    ):
        """
        Parameters:
        -----------
        target_volatility : float
            Annualized target volatility (e.g. 0.15 for 15%).
        max_leverage : float
            Maximum allowable portfolio leverage scaling multiplier.
        warning_drawdown : float
            Drawdown threshold (e.g. 0.10 for 10%) triggering WARNING state.
        deleveraging_drawdown : float
            Drawdown threshold (e.g. 0.15 for 15%) triggering DELEVERAGING state (cuts leverage by 50%).
        hard_halt_drawdown : float
            Drawdown threshold (e.g. 0.20 for 20%) triggering HARD_HALT state (flattens all positions to cash).
        lookback_window : int
            Rolling window size for calculating realized volatility.
        """
        self.target_vol = target_volatility
        self.max_leverage = max_leverage
        self.warning_dd = warning_drawdown
        self.deleveraging_dd = deleveraging_drawdown
        self.hard_halt_dd = hard_halt_drawdown
        self.lookback_window = lookback_window

        # State tracking
        self.peak_equity = -np.inf
        self.current_state = RiskState.NORMAL
        self.current_drawdown = 0.0

    def calculate_volatility_target_weight(
        self,
        returns: Union[np.ndarray, pd.Series],
        decay_factor: float = 0.94
    ) -> Dict[str, float]:
        """
        Calculates leverage scaling factor to enforce constant target volatility.

        Formula:
        Leverage_Scalar = min(Target_Vol / Realized_Vol_Annualized, Max_Leverage)
        """
        returns_arr = np.asarray(returns, dtype=float)
        if len(returns_arr) < 5:
            realized_vol = 0.15  # Default assumption
        else:
            # EWMA Volatility estimation
            var = np.var(returns_arr[:5])
            for r in returns_arr[5:]:
                var = decay_factor * var + (1.0 - decay_factor) * (r ** 2)
            realized_vol = np.sqrt(var * 252.0)

        realized_vol = max(realized_vol, 1e-4)
        target_scalar = min(self.target_vol / realized_vol, self.max_leverage)

        return {
            "target_volatility": float(self.target_vol),
            "realized_volatility_annualized": float(realized_vol),
            "leverage_scaling_factor": float(target_scalar),
            "max_leverage_cap": float(self.max_leverage)
        }

    def update_circuit_breaker(self, current_portfolio_value: float) -> Dict[str, Union[str, float]]:
        """
        Updates peak equity, evaluates portfolio drawdown, and transitions circuit breaker risk states.
        """
        if current_portfolio_value > self.peak_equity:
            self.peak_equity = current_portfolio_value

        drawdown = (self.peak_equity - current_portfolio_value) / self.peak_equity
        self.current_drawdown = float(drawdown)

        # State Machine Transition Logic
        if drawdown >= self.hard_halt_dd:
            self.current_state = RiskState.HARD_HALT
            leverage_cap = 0.0  # Flatten all positions
        elif drawdown >= self.deleveraging_dd:
            self.current_state = RiskState.DELEVERAGING
            leverage_cap = 0.5  # Cut risk in half
        elif drawdown >= self.warning_dd:
            self.current_state = RiskState.WARNING
            leverage_cap = 0.8  # Soft risk reduction
        else:
            if self.current_state in [RiskState.HARD_HALT, RiskState.DELEVERAGING, RiskState.WARNING]:
                self.current_state = RiskState.RECOVERY
            else:
                self.current_state = RiskState.NORMAL
            leverage_cap = 1.0

        return {
            "current_portfolio_value": float(current_portfolio_value),
            "peak_equity": float(self.peak_equity),
            "current_drawdown": float(self.current_drawdown),
            "current_drawdown_pct": f"{self.current_drawdown * 100:.2f}%",
            "risk_state": self.current_state.value,
            "circuit_breaker_leverage_cap": float(leverage_cap)
        }

    @staticmethod
    def atr_trailing_stop(
        high_prices: Union[np.ndarray, pd.Series],
        low_prices: Union[np.ndarray, pd.Series],
        close_prices: Union[np.ndarray, pd.Series],
        atr_period: int = 14,
        atr_multiplier: float = 3.0
    ) -> pd.DataFrame:
        """
        Computes Chandelier Exit ATR Trailing Stop-Loss levels for long position management.

        Formula:
        TR_t = max(H_t - L_t, |H_t - C_{t-1}|, |L_t - C_{t-1}|)
        ATR_t = RollingMean(TR_t, atr_period)
        Trailing_Stop_t = max(Trailing_Stop_{t-1}, High_Peak_t - atr_multiplier * ATR_t)
        """
        df = pd.DataFrame({
            'high': high_prices,
            'low': low_prices,
            'close': close_prices
        }).copy()

        df['prev_close'] = df['close'].shift(1)
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                np.abs(df['high'] - df['prev_close']),
                np.abs(df['low'] - df['prev_close'])
            )
        )

        df['atr'] = df['tr'].rolling(window=atr_period, min_periods=1).mean()
        df['highest_high'] = df['high'].cummax()
        df['raw_stop'] = df['highest_high'] - atr_multiplier * df['atr']

        # Enforce non-decreasing trailing stop
        stops = np.zeros(len(df))
        curr_stop = -np.inf
        for i in range(len(df)):
            raw_s = df['raw_stop'].iloc[i]
            if not np.isnan(raw_s):
                curr_stop = max(curr_stop, raw_s)
            stops[i] = curr_stop

        df['trailing_stop'] = stops
        df['exit_signal'] = df['close'] < df['trailing_stop']

        return df[['close', 'highest_high', 'atr', 'trailing_stop', 'exit_signal']]
