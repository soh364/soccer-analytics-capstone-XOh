"""
Step 4: Normalization & Transformation
- Log/log1p transform for right-skewed metrics (skew > 1.0)
- Rank-based normalization for metrics with negatives or bounded distributions
- Z-score for symmetric metrics
- All normalization is computed PER SEASON to avoid cross-season contamination
- Adds normalized columns as {metric_key}_norm alongside originals
- decay_weight and shrinkage_flag columns are preserved unchanged
"""
import math
import polars as pl
import numpy as np
from typing import Dict
from player_metrics_config import PLAYER_METRICS

# Transform method per metric key
# log    → log(x + epsilon) for strictly positive values
# log1p  → log(1 + x) for zero-inclusive values
# rank   → percentile rank within season (0-1), handles negatives and bounded
# zscore → standard z-score, for symmetric distributions
TRANSFORM_CONFIG = {
    "finishing_quality":   "rank",
    "xg_volume":           "log",
    "progressive_passes":  "log1p",
    "progressive_carries": "log1p",
    "packing":             "log1p",
    "xg_chain":            "log",
    "team_involvement":    "rank",
    "xg_buildup":          "log",
    "network_centrality":  "rank",
    "defensive_actions":   "log",
    "high_turnovers":      "log1p",
    "pressure_volume":     "rank",
    "pressure_success":    "rank",
}

EPSILON = 1e-6  # floor for log transform to avoid log(0)


def _rank_norm(series: pl.Series) -> pl.Series:
    """Percentile rank within series, scaled to [0, 1]."""
    n = series.len()
    if n <= 1:
        return pl.Series([0.5] * n)
    ranks = series.rank(method="average")
    return (ranks - 1) / (n - 1)


def _zscore(series: pl.Series) -> pl.Series:
    """Standard z-score. Returns zeros if std is zero."""
    mean = series.mean()
    std = series.std()
    if std is None or std == 0:
        return pl.Series([0.0] * series.len())
    return (series - mean) / std


def _log_transform(series: pl.Series, use_log1p: bool = False) -> pl.Series:
    """Log transform, then rank-normalize within the season."""
    if use_log1p:
        transformed = (series + 1).log(base=math.e)
    else:
        transformed = (series.clip(lower_bound=EPSILON)).log(base=math.e)
    return _rank_norm(transformed)


def _normalize_series(series: pl.Series, method: str) -> pl.Series:
    """Apply the specified normalization method to a series."""
    import math
    null_mask = series.is_null()
    filled = series.fill_null(strategy="mean") if null_mask.any() else series

    if method == "rank":
        normed = _rank_norm(filled)
    elif method == "zscore":
        normed = _zscore(filled)
    elif method == "log":
        normed = _log_transform(filled, use_log1p=False)
    elif method == "log1p":
        normed = _log_transform(filled, use_log1p=True)
    else:
        raise ValueError(f"Unknown transform method: {method}")

    # Restore nulls
    result = normed.to_list()
    null_list = null_mask.to_list()
    return pl.Series([None if null_list[i] else result[i] for i in range(len(result))])


def normalize_file(
    df: pl.DataFrame,
    filename: str,
    verbose: bool = True,
) -> pl.DataFrame:

    # Pre-processing: compute derived metrics before normalization
    if filename == "xg__player__totals.csv":
        if "goals" in df.columns and "xg" in df.columns:
            df = df.with_columns(
                (pl.col("goals") / pl.col("xg").clip(lower_bound=0.1))
                .alias("goals_per_xg")
            )
    # Find which metrics map to this file
    metrics_for_file = {
        k: v for k, v in PLAYER_METRICS.items()
        if v["file"] == filename
    }

    if not metrics_for_file:
        if verbose:
            print(f"  [norm] {filename}: no metrics configured, passing through")
        return df

    if "season_name" not in df.columns:
        if verbose:
            print(f"  [norm] {filename}: no season_name, normalizing globally")
        seasons = [None]
    else:
        seasons = df["season_name"].unique().to_list()

    norm_cols = {}  # col_name → list of normed values in row order

    for metric_key, cfg in metrics_for_file.items():
        col = cfg["column"]
        method = TRANSFORM_CONFIG.get(metric_key, "rank")
        norm_col = f"{metric_key}_norm"

        if col not in df.columns:
            if verbose:
                print(f"  [norm] {filename}: column '{col}' not found, skipping")
            continue

        # Per-season normalization
        # Build index → normed_value mapping
        normed_values = [None] * len(df)

        for season in seasons:
            if season is None:
                mask = pl.Series([True] * len(df))
            else:
                mask = df["season_name"] == season

            indices = mask.arg_true().to_list()
            if not indices:
                continue

            season_series = df[col].gather(indices)
            season_normed = _normalize_series(season_series, method)

            for idx, val in zip(indices, season_normed.to_list()):
                normed_values[idx] = val

        norm_cols[norm_col] = normed_values

    # Add all norm columns at once
    for norm_col, values in norm_cols.items():
        df = df.with_columns(pl.Series(norm_col, values, dtype=pl.Float64))

    if verbose:
        norm_col_names = list(norm_cols.keys())
        print(f"  [norm] {filename}: added {len(norm_col_names)} norm columns: {norm_col_names}")

    return df


def normalize_all(
    decayed: Dict[str, pl.DataFrame],
    verbose: bool = True,
) -> Dict[str, pl.DataFrame]:
    """
    Apply per-season normalization to all decayed DataFrames.

    Args:
        decayed: dict of {filename: DataFrame} from decay step
        verbose: print per-file summary

    Returns:
        dict of {filename: DataFrame} with {metric_key}_norm columns added
    """
    if verbose:
        print("\n" + "=" * 60)
        print("STEP 4: NORMALIZATION")
        print("Method: per-season | log/log1p → rank | rank | zscore")
        print("=" * 60)

    normalized = {}
    for filename, df in decayed.items():
        normalized[filename] = normalize_file(df, filename, verbose=verbose)

    return normalized


def normalization_summary(normalized: Dict[str, pl.DataFrame]) -> None:
    """Print a concise summary including norm column stats."""
    print("\n" + "=" * 60)
    print("NORMALIZATION SUMMARY")
    print("=" * 60)
    for filename, df in normalized.items():
        norm_cols = [c for c in df.columns if c.endswith("_norm")]
        print(f"  {filename}: {len(norm_cols)} norm columns")
        for col in norm_cols:
            s = df[col].drop_nulls()
            print(f"    {col}: min={s.min():.3f}, median={s.median():.3f}, "
                  f"max={s.max():.3f}, nulls={df[col].null_count()}")