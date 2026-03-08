"""
Step 2: Filtering
- Hard threshold: exclude player-seasons below minimum sample (default 450 min)
- Shrinkage flag: mark player-seasons between 270-450 min for Bayesian shrinkage later
- Handles files without minutes_played via filter_column/filter_threshold overrides in config
"""

import polars as pl
from typing import Dict
from player_metrics_config import PLAYER_METRICS

# Default thresholds
HARD_THRESHOLD_MINUTES = 450
SHRINKAGE_FLOOR_MINUTES = 270


def _get_filter_spec(metric_key: str) -> tuple[str, int]:
    """Return (filter_column, filter_threshold) for a given metric key."""
    cfg = PLAYER_METRICS[metric_key]
    col = cfg.get("filter_column", "minutes_played")
    threshold = cfg.get("filter_threshold", HARD_THRESHOLD_MINUTES)
    return col, threshold


def filter_file(
    df: pl.DataFrame,
    filename: str,
    hard_threshold: int = HARD_THRESHOLD_MINUTES,
    shrinkage_floor: int = SHRINKAGE_FLOOR_MINUTES,
    verbose: bool = True,
) -> pl.DataFrame:
    """
    Filter a single aggregated DataFrame.

    Adds a `shrinkage_flag` column (bool) to rows that pass the hard threshold
    but fall below the shrinkage floor — used in Step 5.

    Returns only rows that meet the hard threshold.
    """
    # Find which metrics use this file
    metrics_for_file = {
        k: v for k, v in PLAYER_METRICS.items()
        if v["file"] == filename
    }

    if not metrics_for_file:
        if verbose:
            print(f"  [filter] {filename}: no metrics configured, passing through")
        return df.with_columns(pl.lit(False).alias("shrinkage_flag"))

    # All metrics in the same file share the same filter_column/threshold
    # (validated by config design — pick from first metric)
    first_metric = next(iter(metrics_for_file.values()))
    filter_col = first_metric.get("filter_column", "minutes_played")
    threshold = first_metric.get("filter_threshold", hard_threshold)

    # Derive shrinkage floor in same units
    # For minutes: floor = SHRINKAGE_FLOOR_MINUTES
    # For matches proxy (threshold=5): floor = 3
    # For passes proxy (threshold=50): floor = 30
    if filter_col == "minutes_played":
        floor = shrinkage_floor
    else:
        # Scale floor proportionally to threshold
        floor = int(round(shrinkage_floor / hard_threshold * threshold))

    if filter_col not in df.columns:
        if verbose:
            print(f"  [filter] {filename}: '{filter_col}' column not found, passing through")
        return df.with_columns(pl.lit(False).alias("shrinkage_flag"))

    n_before = len(df)

    # Flag rows in the shrinkage zone [floor, threshold)
    shrinkage_flag = (
        (pl.col(filter_col) >= floor) &
        (pl.col(filter_col) < threshold)
    )

    df = df.with_columns(shrinkage_flag.alias("shrinkage_flag"))

    # Hard filter: keep only rows >= threshold
    df = df.filter(pl.col(filter_col) >= threshold)

    n_after = len(df)
    n_shrink = df["shrinkage_flag"].sum()

    if verbose:
        print(f"  [filter] {filename}:")
        print(f"           filter_col={filter_col}, threshold={threshold}, floor={floor}")
        print(f"           {n_before:,} → {n_after:,} rows kept "
              f"({n_before - n_after:,} dropped), {n_shrink:,} flagged for shrinkage")

    return df


def filter_all(
    aggregated: Dict[str, pl.DataFrame],
    hard_threshold: int = HARD_THRESHOLD_MINUTES,
    shrinkage_floor: int = SHRINKAGE_FLOOR_MINUTES,
    verbose: bool = True,
) -> Dict[str, pl.DataFrame]:
    """
    Apply filtering to all aggregated DataFrames.

    Args:
        aggregated: dict of {filename: DataFrame} from aggregation.py
        hard_threshold: minimum minutes (or equivalent) to include in scoring
        shrinkage_floor: minimum minutes to retain with shrinkage flag
        verbose: print per-file summary

    Returns:
        dict of {filename: DataFrame} with rows below hard_threshold removed
        and shrinkage_flag column added
    """
    if verbose:
        print("\n" + "=" * 60)
        print("STEP 2: FILTERING")
        print(f"Hard threshold: {hard_threshold} min  |  Shrinkage floor: {shrinkage_floor} min")
        print("=" * 60)

    filtered = {}
    total_before = 0
    total_after = 0

    for filename, df in aggregated.items():
        before = len(df)
        total_before += before

        result = filter_file(df, filename, hard_threshold, shrinkage_floor, verbose)
        filtered[filename] = result

        after = len(result)
        total_after += after

    if verbose:
        print(f"\n  TOTAL: {total_before:,} → {total_after:,} rows "
              f"({total_before - total_after:,} dropped across all files)")

    return filtered


def filtering_summary(filtered: Dict[str, pl.DataFrame]) -> None:
    """Print a concise summary of filtered DataFrames."""
    print("\n" + "=" * 60)
    print("FILTERING SUMMARY")
    print("=" * 60)
    for filename, df in filtered.items():
        n_players = df["player"].n_unique() if "player" in df.columns else "N/A"
        n_shrink = int(df["shrinkage_flag"].sum()) if "shrinkage_flag" in df.columns else 0
        seasons = sorted(df["season_name"].unique().to_list()) if "season_name" in df.columns else []
        print(f"  {filename}")
        print(f"    {len(df):,} rows | {n_players} players | {n_shrink} shrinkage-flagged | seasons: {seasons}")