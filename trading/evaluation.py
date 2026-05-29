"""Evaluation helpers for walk-forward paper-trading reports."""

from __future__ import annotations

import pandas as pd


class WalkForwardReport:
    """Compare strategy equity against buy-and-hold baseline."""

    def __init__(self, equity_history, price_history):
        self.equity_history = pd.DataFrame(equity_history)
        self.price_history = pd.DataFrame(price_history)

    def compute(self):
        if self.equity_history.empty:
            raise ValueError("equity_history is empty.")
        if self.price_history.empty:
            raise ValueError("price_history is empty.")

        equity_df = self.equity_history.copy()
        price_df = self.price_history.copy()
        if "timestamp" not in equity_df.columns or "timestamp" not in price_df.columns:
            raise ValueError("equity_history and price_history must both include timestamp columns.")

        equity_df["timestamp"] = pd.to_datetime(equity_df["timestamp"])
        price_df["timestamp"] = pd.to_datetime(price_df["timestamp"])
        merged = (
            equity_df[["timestamp", "equity"]]
            .merge(
                price_df[["timestamp", "close"]],
                on="timestamp",
                how="inner",
            )
            .sort_values("timestamp")
            .reset_index(drop=True)
        )
        if merged.empty:
            raise ValueError("equity_history and price_history do not overlap by timestamp.")
        equity = merged["equity"].astype(float)
        prices = merged["close"].astype(float)
        strategy_return = equity.iloc[-1] / equity.iloc[0] - 1.0
        baseline_return = prices.iloc[-1] / prices.iloc[0] - 1.0

        strategy_dd = self._max_drawdown(equity)
        baseline_dd = self._max_drawdown(prices)
        return {
            "strategy_return": strategy_return,
            "baseline_return": baseline_return,
            "excess_return": strategy_return - baseline_return,
            "strategy_max_drawdown": strategy_dd,
            "baseline_max_drawdown": baseline_dd,
            "observations": int(min(len(equity), len(prices))),
        }

    @staticmethod
    def _max_drawdown(series):
        peak = series.expanding().max()
        return ((series - peak) / peak).min()
