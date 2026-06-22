import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ONE_STEP_COLUMNS = [
    "wall_time",
    "next_wall_time",
    "label",
    "symbol",
    "price",
    "next_price",
    "direction",
    "one_step_return_pct",
]


def parse_bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.strip().str.lower().isin({"1", "true", "yes", "y"})


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def last_numeric(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.iloc[-1])


def summarize_counts(df: pd.DataFrame, group_col: str, value_col: str, limit: int) -> pd.DataFrame:
    if df.empty or group_col not in df or value_col not in df:
        return pd.DataFrame(columns=[group_col, "count"])
    return (
        df.dropna(subset=[value_col])
        .groupby(group_col, as_index=False)
        .size()
        .rename(columns={"size": "count"})
        .sort_values("count", ascending=False)
        .head(limit)
    )


def one_step_opportunity(candidate_log: pd.DataFrame) -> pd.DataFrame:
    if candidate_log.empty:
        return pd.DataFrame(columns=ONE_STEP_COLUMNS)

    rows = []
    snapshots = list(candidate_log.groupby("wall_time", sort=True))
    for (wall_time, group), (next_wall_time, next_group) in zip(snapshots, snapshots[1:]):
        next_prices = dict(zip(next_group["symbol"], pd.to_numeric(next_group["price"], errors="coerce")))
        for label, rank_col, direction in [
            ("top_short", "short_rank", -1.0),
            ("top_long", "long_rank", 1.0),
            ("selected", "is_selected_candidate", 0.0),
        ]:
            if label == "selected":
                selected = group[parse_bool_series(group["is_selected_candidate"])]
                if selected.empty:
                    continue
                row = selected.iloc[0]
                weight = float(row.get("candidate_weight", 0.0) or 0.0)
                direction = np.sign(weight) if weight != 0 else 0.0
            else:
                ranked = group[pd.to_numeric(group[rank_col], errors="coerce") == 1]
                if ranked.empty:
                    continue
                row = ranked.iloc[0]

            symbol = row["symbol"]
            price = float(row["price"]) if pd.notna(row["price"]) else np.nan
            next_price = next_prices.get(symbol, np.nan)
            if not np.isfinite(price) or not np.isfinite(next_price) or price <= 0 or direction == 0:
                realized_return = np.nan
            else:
                realized_return = direction * (next_price / price - 1.0)
            rows.append(
                {
                    "wall_time": wall_time,
                    "next_wall_time": next_wall_time,
                    "label": label,
                    "symbol": symbol,
                    "price": price,
                    "next_price": next_price,
                    "direction": direction,
                    "one_step_return_pct": realized_return * 100.0 if np.isfinite(realized_return) else np.nan,
                }
            )
    return pd.DataFrame(rows, columns=ONE_STEP_COLUMNS)


def analyze_run(run_dir: Path, output_dir: Path, top_n: int) -> dict:
    candidate_path = run_dir / "candidate_log.csv"
    live_path = run_dir / "live_log.csv"
    summary_path = run_dir / "summary.json"
    if not candidate_path.exists():
        raise FileNotFoundError(f"missing candidate log: {candidate_path}")
    if not live_path.exists():
        raise FileNotFoundError(f"missing live log: {live_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_log = pd.read_csv(candidate_path)
    live_log = pd.read_csv(live_path)
    summary = load_json(summary_path)

    for column in ["price", "score", "short_rank", "long_rank", "candidate_weight"]:
        if column in candidate_log:
            candidate_log[column] = pd.to_numeric(candidate_log[column], errors="coerce")
    candidate_log["tradable_bool"] = parse_bool_series(candidate_log["tradable"])
    candidate_log["has_live_price_bool"] = parse_bool_series(candidate_log["has_live_price"])
    candidate_log["is_selected_candidate_bool"] = parse_bool_series(candidate_log["is_selected_candidate"])

    polls = candidate_log.groupby("wall_time", sort=True)
    selected = candidate_log[candidate_log["is_selected_candidate_bool"]]
    short_rank1 = candidate_log[candidate_log["short_rank"] == 1]
    long_rank1 = candidate_log[candidate_log["long_rank"] == 1]

    match_rows = []
    if "candidate_symbol" in live_log:
        candidate_wall_times = set(candidate_log["wall_time"].dropna().astype(str))
        live_candidates = live_log.dropna(subset=["candidate_symbol"])[["wall_time", "candidate_symbol"]].copy()
        live_candidates = live_candidates[live_candidates["wall_time"].astype(str).isin(candidate_wall_times)]
        selected_by_poll = selected[["wall_time", "symbol"]].rename(columns={"symbol": "logged_selected_symbol"})
        merged = live_candidates.merge(selected_by_poll, on="wall_time", how="left")
        if not merged.empty:
            merged["matches_candidate_log"] = merged["candidate_symbol"] == merged["logged_selected_symbol"]
            match_rows = merged.to_dict(orient="records")
        else:
            merged = pd.DataFrame(
                columns=["wall_time", "candidate_symbol", "logged_selected_symbol", "matches_candidate_log"]
            )
        merged.to_csv(output_dir / "candidate_consistency.csv", index=False)

    opportunities = one_step_opportunity(candidate_log)
    opportunities.to_csv(output_dir / "one_step_opportunities.csv", index=False)

    top_short_counts = summarize_counts(short_rank1, "symbol", "short_rank", top_n)
    top_long_counts = summarize_counts(long_rank1, "symbol", "long_rank", top_n)
    selected_counts = summarize_counts(selected, "symbol", "is_selected_candidate", top_n)
    top_short_counts.to_csv(output_dir / "top_short_rank1_counts.csv", index=False)
    top_long_counts.to_csv(output_dir / "top_long_rank1_counts.csv", index=False)
    selected_counts.to_csv(output_dir / "selected_candidate_counts.csv", index=False)

    one_step_summary = {}
    if not opportunities.empty:
        one_step_summary = (
            opportunities.groupby("label")["one_step_return_pct"]
            .agg(["count", "mean", "min", "max"])
            .reset_index()
            .to_dict(orient="records")
        )

    final_equity = summary.get("final_equity_after_close")
    final_return_pct = summary.get("final_return_pct_after_close")
    if final_equity is None and "equity" in live_log:
        final_equity = last_numeric(live_log["equity"])
    if final_return_pct is None and "return_pct" in live_log:
        final_return_pct = last_numeric(live_log["return_pct"])

    result = {
        "run_dir": str(run_dir),
        "live_log_rows": int(len(live_log)),
        "candidate_log_rows": int(len(candidate_log)),
        "candidate_polls": int(candidate_log["wall_time"].nunique()),
        "unique_symbols": int(candidate_log["symbol"].nunique()),
        "avg_symbols_per_poll": float(polls.size().mean()) if len(candidate_log) else 0.0,
        "price_coverage": float(candidate_log["has_live_price_bool"].mean()) if len(candidate_log) else 0.0,
        "tradable_rate": float(candidate_log["tradable_bool"].mean()) if len(candidate_log) else 0.0,
        "selected_candidate_rows": int(len(selected)),
        "final_equity_after_close": final_equity,
        "final_return_pct_after_close": final_return_pct,
        "candidate_consistency_matches": int(
            sum(bool(row.get("matches_candidate_log")) for row in match_rows)
        ),
        "candidate_consistency_checks": int(len(match_rows)),
        "one_step_rows": int(len(opportunities)),
        "candidate_consistency": match_rows,
        "one_step_summary": one_step_summary,
    }
    (output_dir / "summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    readme = [
        "# Live Candidate Log Analysis",
        "",
        f"- Run dir: `{run_dir}`",
        f"- Candidate polls: {result['candidate_polls']}",
        f"- Candidate rows: {result['candidate_log_rows']}",
        f"- Unique symbols: {result['unique_symbols']}",
        f"- Price coverage: {result['price_coverage']:.2%}",
        f"- Tradable rate: {result['tradable_rate']:.2%}",
        f"- Selected candidate rows: {result['selected_candidate_rows']}",
        f"- Candidate consistency: {result['candidate_consistency_matches']}/{result['candidate_consistency_checks']}",
        f"- One-step opportunity rows: {result['one_step_rows']}",
        f"- Final return after close: {float(final_return_pct or 0.0):.4f}%",
        "",
        "Generated files:",
        "",
        "- `summary.json`",
        "- `candidate_consistency.csv`",
        "- `one_step_opportunities.csv`",
        "- `top_short_rank1_counts.csv`",
        "- `top_long_rank1_counts.csv`",
        "- `selected_candidate_counts.csv`",
        "",
    ]
    (output_dir / "README.md").write_text("\n".join(readme), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze live paper candidate_log.csv artifacts.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-n", type=int, default=20)
    args = parser.parse_args()

    result = analyze_run(args.run_dir, args.output_dir, args.top_n)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
