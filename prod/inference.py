"""Inference adapters for the production shadow scheduler."""

from __future__ import annotations

import os

import pandas as pd

from .config import ProdConfig
from .ledger import stable_hash


MODEL_IDS = {
    "kronos-mini": ("NeoQuasar/Kronos-mini", "NeoQuasar/Kronos-Tokenizer-2k", 2048),
    "kronos-small": ("NeoQuasar/Kronos-small", "NeoQuasar/Kronos-Tokenizer-base", 512),
    "kronos-base": ("NeoQuasar/Kronos-base", "NeoQuasar/Kronos-Tokenizer-base", 512),
}


class KronosModelPredictor:
    """Lazy Kronos adapter that predicts the next close from closed OHLCV bars."""

    def __init__(self, config: ProdConfig, *, sample_count=1, temperature=1.0, top_p=0.9):
        if config.model_key not in MODEL_IDS:
            raise ValueError(f"Unsupported Kronos model_key: {config.model_key}")
        if str(config.device).lower().startswith("cuda"):
            if not config.cuda_visible_devices:
                raise ValueError("CUDA inference requires explicit cuda_visible_devices.")
            os.environ["CUDA_VISIBLE_DEVICES"] = str(config.cuda_visible_devices)
        self.config = config
        self.sample_count = int(sample_count)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.model_version = MODEL_IDS[config.model_key][0]
        self.model_hash = stable_hash(
            {
                "model_key": config.model_key,
                "model_revision": config.model_revision,
                "tokenizer_revision": config.tokenizer_revision,
            }
        )
        self._predictor = None

    def predict_close(self, bars) -> float:
        predictor = self._load()
        model_id, _tokenizer_id, max_context = MODEL_IDS[self.config.model_key]
        frame = pd.DataFrame(
            [
                {
                    "open": bar.open,
                    "high": bar.high,
                    "low": bar.low,
                    "close": bar.close,
                    "volume": bar.volume,
                    "amount": bar.quote_volume or 0.0,
                    "timestamps": pd.Timestamp(bar.period_end),
                }
                for bar in bars[-max_context:]
            ]
        )
        if frame.empty:
            raise ValueError("Cannot run Kronos inference without closed bars.")
        frame = frame.sort_values("timestamps").reset_index(drop=True)
        interval = frame["timestamps"].diff().dropna().median()
        if pd.isna(interval):
            interval = pd.Timedelta(minutes=1)
        y_timestamp = pd.Series([frame["timestamps"].iloc[-1] + interval])
        prediction = predictor.predict(
            df=frame[["open", "high", "low", "close", "volume", "amount"]],
            x_timestamp=frame["timestamps"],
            y_timestamp=y_timestamp,
            pred_len=1,
            T=self.temperature,
            top_p=self.top_p,
            sample_count=self.sample_count,
        )
        return float(prediction["close"].iloc[-1])

    def _load(self):
        if self._predictor is not None:
            return self._predictor
        from model import Kronos, KronosPredictor, KronosTokenizer

        model_id, tokenizer_id, max_context = MODEL_IDS[self.config.model_key]
        tokenizer = KronosTokenizer.from_pretrained(tokenizer_id, revision=self.config.tokenizer_revision)
        model = Kronos.from_pretrained(model_id, revision=self.config.model_revision)
        self._predictor = KronosPredictor(model, tokenizer, max_context=max_context)
        return self._predictor


def build_predictor(config: ProdConfig, kind="hold"):
    if kind == "hold":
        from .runtime import HoldLastClosePredictor

        return HoldLastClosePredictor()
    if kind == "kronos":
        return KronosModelPredictor(config)
    raise ValueError("predictor kind must be hold or kronos.")
