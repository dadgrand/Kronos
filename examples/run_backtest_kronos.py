# run_backtest_kronos.py
"""Conservative historical backtest helper for Kronos prediction files.

This script is intentionally limited: it can backtest only predictions whose
target timestamps overlap realized historical prices. Future-only forecasts are
not a backtest and are rejected instead of being marked to model-generated prices.
"""

import json
import os
import glob
import warnings

import numpy as np
import pandas as pd

from trading.paper import BUY, SELL, MarketBar, PaperBroker, PaperRiskManager

warnings.filterwarnings("ignore")


class KronosBacktester:
    """Historical long/flat backtester for prediction CSVs."""

    REQUIRED_PREDICTION_COLUMNS = {
        "symbol",
        "prediction_asof",
        "execution_timestamp",
        "target_timestamp",
        "features_cutoff",
        "horizon",
        "model_version",
        "model_hash",
        "predicted_close",
    }

    def __init__(
        self,
        data_dir,
        model_dir,
        initial_capital=100000,
        commission_rate=0.001,
        slippage_rate=0.0005,
        max_position_fraction=0.25,
        min_cash_fraction=0.01,
        max_drawdown_stop=0.2,
        max_participation_rate=0.1,
        default_bar_volume=1_000_000,
        allow_short=False,
    ):
        if initial_capital <= 0:
            raise ValueError("initial_capital must be positive.")
        if not 0 <= commission_rate < 1:
            raise ValueError("commission_rate must be in [0, 1).")
        if not 0 <= slippage_rate < 1:
            raise ValueError("slippage_rate must be in [0, 1).")
        if not 0 < max_position_fraction <= 1:
            raise ValueError("max_position_fraction must be in (0, 1].")
        if not 0 <= min_cash_fraction < 1:
            raise ValueError("min_cash_fraction must be in [0, 1).")
        if not 0 < max_drawdown_stop < 1:
            raise ValueError("max_drawdown_stop must be in (0, 1).")
        if not 0 < max_participation_rate <= 1:
            raise ValueError("max_participation_rate must be in (0, 1].")
        if default_bar_volume <= 0:
            raise ValueError("default_bar_volume must be positive.")
        if allow_short:
            raise ValueError("Short backtests require an explicit margin, borrow, and locate model.")

        self.data_dir = data_dir
        self.model_dir = model_dir
        self.initial_capital = float(initial_capital)
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        self.max_position_fraction = max_position_fraction
        self.min_cash_fraction = min_cash_fraction
        self.max_drawdown_stop = max_drawdown_stop
        self.max_participation_rate = max_participation_rate
        self.default_bar_volume = default_bar_volume
        self.allow_short = False

    def load_historical_data(self, stock_code):
        csv_file = os.path.join(self.data_dir, f"{stock_code}_stock_data.csv")
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"Historical data file does not exist: {csv_file}")

        df = pd.read_csv(csv_file, encoding="utf-8-sig")
        column_mapping = {
            "date": "date",
            "timestamp": "date",
            "timestamps": "date",
            "日期": "date",
            "开盘价": "open",
            "最高价": "high",
            "最低价": "low",
            "收盘价": "close",
            "成交量": "volume",
            "成交额": "amount",
        }
        for old_col, new_col in column_mapping.items():
            if old_col in df.columns:
                df = df.rename(columns={old_col: new_col})

        required_cols = {"date", "close"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Historical data is missing required columns: {sorted(missing)}")
        if "open" not in df.columns:
            df["open"] = df["close"]

        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        df[["open", "close"]] = df[["open", "close"]].apply(pd.to_numeric, errors="coerce")
        df = df.dropna(subset=["open", "close"])

        print(f"Loaded historical rows: {len(df)}")
        print(f"Historical range: {df.index.min()} to {df.index.max()}")
        return df

    def load_predictions(self, stock_code):
        pred_files = [
            os.path.join(self.model_dir, f"{stock_code}_kronos_predictions.csv"),
            os.path.join(self.model_dir, f"{stock_code}_kronos_predictions.json"),
            os.path.join(self.model_dir, f"{stock_code}_detailed_predictions.csv"),
            os.path.join(self.model_dir, f"{stock_code}_detailed_predictions.json"),
            os.path.join(self.model_dir, f"{stock_code}_predictions.csv"),
            os.path.join(self.model_dir, f"{stock_code}_predictions.json"),
        ]
        pred_files.extend(
            sorted(
                glob.glob(os.path.join(self.model_dir, "prediction_*.json")),
                key=os.path.getmtime,
                reverse=True,
            )
        )

        pred_df = None
        for pred_file in pred_files:
            if os.path.exists(pred_file):
                if pred_file.endswith(".json"):
                    with open(pred_file, "r", encoding="utf-8") as f:
                        payload = json.load(f)
                    pred_df = pd.DataFrame(payload.get("prediction_results", payload))
                else:
                    pred_df = pd.read_csv(pred_file, encoding="utf-8-sig")
                print(f"Loaded prediction file: {pred_file}")
                break

        if pred_df is None:
            raise FileNotFoundError(f"No prediction file found in: {self.model_dir}")

        column_mapping = {
            "date": "target_timestamp",
            "timestamp": "target_timestamp",
            "timestamps": "target_timestamp",
            "日期": "target_timestamp",
            "target": "target_timestamp",
            "target_date": "target_timestamp",
            "execution_time": "execution_timestamp",
            "execution_date": "execution_timestamp",
            "trade_timestamp": "execution_timestamp",
            "asof": "prediction_asof",
            "generated_at": "prediction_asof",
            "feature_cutoff": "features_cutoff",
            "预测收盘价": "predicted_close",
            "收盘价": "predicted_close",
            "close": "predicted_close",
        }
        for old_col, new_col in column_mapping.items():
            if old_col in pred_df.columns and new_col not in pred_df.columns:
                pred_df = pred_df.rename(columns={old_col: new_col})

        missing = self.REQUIRED_PREDICTION_COLUMNS - set(pred_df.columns)
        if missing:
            raise ValueError(
                "Prediction data is missing the strict as-of contract columns: "
                f"{sorted(missing)}"
            )

        pred_df["symbol"] = pred_df["symbol"].astype(str)
        if stock_code and not (pred_df["symbol"] == str(stock_code)).all():
            raise ValueError(f"Prediction symbols must all match requested stock_code={stock_code}.")

        for col in ("prediction_asof", "execution_timestamp", "target_timestamp", "features_cutoff"):
            pred_df[col] = pd.to_datetime(pred_df[col])

        pred_df["predicted_close"] = pd.to_numeric(pred_df["predicted_close"], errors="coerce")
        pred_df = self.validate_prediction_contract(pred_df)
        pred_df = pred_df.set_index("execution_timestamp", drop=False).sort_index()
        pred_df = pred_df.dropna(subset=["predicted_close"])

        print(f"Loaded prediction rows: {len(pred_df)}")
        print(f"Prediction range: {pred_df.index.min()} to {pred_df.index.max()}")
        return pred_df

    def validate_prediction_contract(self, pred_df):
        missing = self.REQUIRED_PREDICTION_COLUMNS - set(pred_df.columns)
        if missing:
            raise ValueError(
                "Prediction data is missing the strict as-of contract columns: "
                f"{sorted(missing)}"
            )

        checked = pred_df.copy()
        if "target_timestamp" not in checked.columns:
            checked["target_timestamp"] = checked.index
        if "execution_timestamp" not in checked.columns:
            checked["execution_timestamp"] = checked.index

        for col in ("prediction_asof", "execution_timestamp", "target_timestamp", "features_cutoff"):
            checked[col] = pd.to_datetime(checked[col])

        checked["predicted_close"] = pd.to_numeric(checked["predicted_close"], errors="coerce")
        checked = checked.dropna(subset=["predicted_close"])

        if checked.empty:
            raise ValueError("Prediction data is empty after parsing numeric predictions.")
        if checked["execution_timestamp"].duplicated().any():
            raise ValueError("Prediction data contains duplicate execution_timestamp values.")
        if not (checked["prediction_asof"] < checked["execution_timestamp"]).all():
            raise ValueError("Every prediction_asof must be strictly before execution_timestamp.")
        if not (checked["execution_timestamp"] <= checked["target_timestamp"]).all():
            raise ValueError("execution_timestamp must be less than or equal to target_timestamp.")
        if not (checked["execution_timestamp"] == checked["target_timestamp"]).all():
            raise ValueError(
                "This helper supports one-bar forecasts only: execution_timestamp must equal target_timestamp. "
                "Use an event-driven backtester for multi-bar horizons."
            )
        if not (checked["features_cutoff"] <= checked["prediction_asof"]).all():
            raise ValueError("features_cutoff must be less than or equal to prediction_asof.")
        if not (checked["features_cutoff"] < checked["execution_timestamp"]).all():
            raise ValueError("features_cutoff must be strictly before execution_timestamp.")

        for col in ("symbol", "horizon", "model_version", "model_hash"):
            values = checked[col].astype(str).str.strip()
            placeholders = {"", "unknown", "none", "nan", "unversioned"}
            if values.str.lower().isin(placeholders).any():
                raise ValueError(f"{col} must be populated with non-placeholder values.")

        horizon_delta = pd.to_timedelta(checked["horizon"], errors="coerce")
        if horizon_delta.isna().any() or (horizon_delta <= pd.Timedelta(0)).any():
            raise ValueError("horizon must be a positive pandas-compatible Timedelta string, e.g. '1D'.")
        realized_horizon = checked["target_timestamp"] - checked["prediction_asof"]
        if not (horizon_delta == realized_horizon).all():
            raise ValueError("horizon must equal target_timestamp - prediction_asof for every row.")

        return checked

    def align_data(self, hist_df, pred_df):
        """Return only dates where realized market prices and predictions coexist."""
        common_index = hist_df.index.intersection(pred_df.index).sort_values()
        if common_index.empty:
            raise ValueError(
                "Cannot run a historical backtest without overlapping actual and prediction dates. "
                "Future-only forecasts can be inspected, but they cannot produce realized PnL."
            )

        print(
            f"Aligned {len(common_index)} rows from {common_index.min()} to {common_index.max()} "
            "using realized market prices only."
        )
        aligned_predictions = pred_df.loc[common_index].copy()
        if "execution_timestamp" not in aligned_predictions.columns:
            aligned_predictions["execution_timestamp"] = aligned_predictions.index
        aligned_predictions = self.validate_prediction_contract(aligned_predictions)

        return hist_df.loc[common_index].copy(), aligned_predictions.set_index("execution_timestamp", drop=False).sort_index()

    def calculate_trading_signals(self, hist_df, pred_df, threshold=0.02):
        """Build target positions from target-date forecasts without using target close as input."""
        if threshold < 0:
            raise ValueError("threshold must be non-negative.")

        full_hist_df = hist_df.sort_index().copy()
        hist_df, pred_df = self.align_data(hist_df, pred_df)
        combined = pd.DataFrame(index=hist_df.index)
        combined["open"] = hist_df["open"].astype(float)
        combined["actual"] = hist_df["close"].astype(float)
        combined["volume"] = (
            pd.to_numeric(hist_df["volume"], errors="coerce")
            if "volume" in hist_df.columns
            else float(self.default_bar_volume)
        )
        combined["symbol"] = pred_df["symbol"].astype(str)
        combined["predicted"] = pred_df["predicted_close"].astype(float)
        reference_index = pd.DatetimeIndex(pred_df["prediction_asof"])
        reference_close = full_hist_df["close"].reindex(reference_index)
        if reference_close.isna().any():
            missing = sorted(
                pd.Timestamp(timestamp).isoformat()
                for timestamp in reference_index[reference_close.isna()]
            )
            raise ValueError(f"Missing prediction_asof reference close for: {missing}")
        combined["reference_close"] = reference_close.to_numpy(dtype=float)
        combined = combined.replace([np.inf, -np.inf], np.nan).dropna()

        if combined.empty:
            raise ValueError("No aligned rows remain after applying prediction_asof reference closes.")

        combined["pred_return"] = combined["predicted"] / combined["reference_close"] - 1.0

        short_signal = -1 if self.allow_short else 0
        combined["signal"] = np.where(
            combined["pred_return"] > threshold,
            1,
            np.where(combined["pred_return"] < -threshold, short_signal, 0),
        )
        combined["position"] = combined["signal"].astype(int)
        return combined

    def run_backtest(self, combined_df):
        """Run the strategy through the shared paper broker execution path."""
        required_cols = {"symbol", "open", "actual", "volume", "position"}
        missing = required_cols - set(combined_df.columns)
        if missing:
            raise ValueError(f"combined_df is missing required columns: {sorted(missing)}")

        risk_manager = PaperRiskManager(
            initial_equity=self.initial_capital,
            max_drawdown=self.max_drawdown_stop,
            min_cash_fraction=self.min_cash_fraction,
        )
        broker = PaperBroker(
            initial_cash=self.initial_capital,
            commission_rate=self.commission_rate,
            slippage_rate=self.slippage_rate,
            max_participation_rate=self.max_participation_rate,
            risk_manager=risk_manager,
        )
        trades = []
        results = []

        for date, row in combined_df.iterrows():
            symbol = str(row["symbol"])
            open_price = float(row["open"])
            close_price = float(row["actual"])
            volume = float(row["volume"])
            signal = int(row["position"])

            if (
                not np.isfinite(open_price)
                or not np.isfinite(close_price)
                or not np.isfinite(volume)
                or open_price <= 0
                or close_price <= 0
                or volume < 0
            ):
                continue

            reference_open = {symbol: open_price}
            equity_at_open = broker.equity(reference_open)
            if equity_at_open <= 0:
                raise ValueError("Portfolio equity became non-positive.")

            risk_manager.update(equity_at_open)
            if risk_manager.halted:
                signal = 0

            current_position = broker.positions.get(symbol)
            current_shares = current_position.quantity if current_position else 0
            broker.cancel_open_orders(symbol=symbol, reason="replaced by latest target")
            if signal > 0:
                target_notional = max(0.0, equity_at_open * self.max_position_fraction)
                estimated_buy_price = open_price * (1 + self.slippage_rate)
                target_shares = int(target_notional / (estimated_buy_price * (1 + self.commission_rate)))
            else:
                target_shares = 0

            delta = target_shares - current_shares
            if delta > 0:
                broker.submit_order(symbol, BUY, delta, date, reference_prices=reference_open)
            elif delta < 0:
                broker.submit_order(symbol, SELL, abs(delta), date, reference_prices=reference_open)

            bar = MarketBar(
                symbol=symbol,
                timestamp=pd.Timestamp(date),
                open=open_price,
                high=max(open_price, close_price),
                low=min(open_price, close_price),
                close=close_price,
                volume=volume,
            )
            fills = broker.process_bar(bar, reference_prices=reference_open)
            for fill in fills:
                trades.append(
                    {
                        "date": fill.timestamp,
                        "action": fill.side,
                        "price": fill.price,
                        "shares": fill.quantity,
                        "commission": fill.commission,
                        "cash": broker.cash,
                        "order_id": fill.order_id,
                        "target_position": target_shares,
                    }
                )

            portfolio_value = broker.equity({symbol: close_price})
            risk_manager.update(portfolio_value)
            position = broker.positions.get(symbol)
            results.append(
                {
                    "date": date,
                    "capital": portfolio_value,
                    "cash": broker.cash,
                    "position": position.quantity if position else 0,
                    "price": close_price,
                    "trading_halted": risk_manager.halted,
                }
            )

        if not results:
            raise ValueError("Backtest produced no valid rows.")

        backtest_results = pd.DataFrame(results).set_index("date")
        backtest_results["returns"] = backtest_results["capital"].pct_change().fillna(0.0)
        return backtest_results, trades

    def calculate_metrics(self, backtest_results, trades):
        returns = backtest_results["returns"].replace([np.inf, -np.inf], np.nan).dropna()
        total_return = (backtest_results["capital"].iloc[-1] - self.initial_capital) / self.initial_capital
        annual_return = (1 + total_return) ** (252 / max(len(returns), 1)) - 1
        volatility = returns.std() * np.sqrt(252) if len(returns) > 1 else 0.0
        risk_free_rate = 0.03
        sharpe_ratio = (annual_return - risk_free_rate) / volatility if volatility > 0 else 0.0

        equity_curve = backtest_results["capital"] / self.initial_capital
        peak = equity_curve.expanding().max()
        max_drawdown = ((equity_curve - peak) / peak).min()

        closed_trade_returns = []
        entry_price = None
        for trade in trades:
            if trade["action"] == "BUY" and entry_price is None:
                entry_price = trade["price"]
            elif trade["action"] == "SELL" and entry_price:
                closed_trade_returns.append((trade["price"] - entry_price) / entry_price)
                entry_price = None

        win_rate = (
            len([ret for ret in closed_trade_returns if ret > 0]) / len(closed_trade_returns)
            if closed_trade_returns
            else 0.0
        )
        avg_trade_return = float(np.mean(closed_trade_returns)) if closed_trade_returns else 0.0

        return {
            "total_return": total_return,
            "annual_return": annual_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "avg_trade_return": avg_trade_return,
            "trade_count": len(trades),
            "final_capital": backtest_results["capital"].iloc[-1],
        }

    def plot_backtest_results(self, backtest_results, metrics, stock_code, output_dir):
        import matplotlib.pyplot as plt

        plt.rcParams["font.sans-serif"] = ["SimHei"]
        plt.rcParams["axes.unicode_minus"] = False

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))

        ax1.plot(backtest_results.index, backtest_results["capital"], linewidth=2, label="Strategy equity")
        ax1.axhline(y=self.initial_capital, color="red", linestyle="--", label="Initial capital")
        ax1.set_ylabel("Capital")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_title(f"{stock_code} Kronos historical backtest")

        cumulative_returns = (1 + backtest_results["returns"].fillna(0)).cumprod()
        ax2.plot(backtest_results.index, cumulative_returns, linewidth=2, label="Strategy return")
        benchmark_returns = (1 + backtest_results["price"].pct_change().fillna(0)).cumprod()
        ax2.plot(backtest_results.index, benchmark_returns, linewidth=2, label="Buy and hold", alpha=0.7)
        ax2.set_ylabel("Cumulative return")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        peak = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - peak) / peak
        ax3.fill_between(backtest_results.index, drawdown, 0, alpha=0.3, color="red", label="Drawdown")
        ax3.set_ylabel("Drawdown")
        ax3.set_xlabel("Date")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        metrics_text = (
            f"Total return: {metrics['total_return']:.2%}\n"
            f"Annual return: {metrics['annual_return']:.2%}\n"
            f"Sharpe: {metrics['sharpe_ratio']:.2f}\n"
            f"Max drawdown: {metrics['max_drawdown']:.2%}\n"
            f"Win rate: {metrics['win_rate']:.2%}\n"
            f"Trades: {metrics['trade_count']}\n"
            f"Final capital: {metrics['final_capital']:,.0f}"
        )
        ax1.text(
            0.02,
            0.98,
            metrics_text,
            transform=ax1.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
        )

        plt.tight_layout()
        os.makedirs(output_dir, exist_ok=True)
        chart_file = os.path.join(output_dir, f"{stock_code}_backtest_results.png")
        plt.savefig(chart_file, dpi=300, bbox_inches="tight")
        print(f"Backtest chart saved: {chart_file}")
        plt.show()

    def run_complete_backtest(self, stock_code, output_dir, threshold=0.02, raise_on_error=True):
        print(f"Starting historical backtest for {stock_code}")
        print("=" * 50)

        try:
            hist_df = self.load_historical_data(stock_code)
            pred_df = self.load_predictions(stock_code)
            combined_df = self.calculate_trading_signals(hist_df, pred_df, threshold)
            backtest_results, trades = self.run_backtest(combined_df)
            metrics = self.calculate_metrics(backtest_results, trades)
            self.plot_backtest_results(backtest_results, metrics, stock_code, output_dir)

            print("\n" + "=" * 70)
            print(f"{stock_code} backtest report")
            print("=" * 70)
            for key, value in metrics.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")

            print(f"\nTrades ({len(trades)} total):")
            for i, trade in enumerate(trades[-10:], 1):
                print(
                    f"  {i}: {trade['date'].strftime('%Y-%m-%d')} "
                    f"{trade['action']} {trade['shares']} shares @ {trade['price']:.2f}"
                )

            return metrics, backtest_results, trades
        except Exception as exc:
            if raise_on_error:
                raise
            print(f"Backtest failed: {exc}")
            import traceback

            traceback.print_exc()
            return None, None, None


def main():
    backtest_config = {
        "stock_code": "000831",
        "data_dir": r"D:\lianghuajiaoyi\Kronos\examples\data",
        "model_dir": r"D:\lianghuajiaoyi\Kronos\examples\yuce",
        "output_dir": r"D:\lianghuajiaoyi\Kronos\examples\backtest",
        "initial_capital": 100000,
        "threshold": 0.02,
    }

    print("Kronos historical backtest")
    print("=" * 50)
    print(f"Stock code: {backtest_config['stock_code']}")
    print(f"Initial capital: {backtest_config['initial_capital']:,.0f}")
    print(f"Threshold: {backtest_config['threshold']:.1%}")
    print()

    backtester = KronosBacktester(
        data_dir=backtest_config["data_dir"],
        model_dir=backtest_config["model_dir"],
        initial_capital=backtest_config["initial_capital"],
    )
    metrics, _, _ = backtester.run_complete_backtest(
        stock_code=backtest_config["stock_code"],
        output_dir=backtest_config["output_dir"],
        threshold=backtest_config["threshold"],
    )

    if metrics:
        print(f"\nBacktest completed. Results saved to: {backtest_config['output_dir']}")


if __name__ == "__main__":
    main()
