"""
Step 5: Bayesian Shrinkage
- For rows with shrinkage_flag=True (180-270 min), shrink norm values
  toward the positional mean for that metric and season
- Shrinkage factor λ is proportional to minutes deficit:
    λ = 1 - (minutes - floor) / (threshold - floor)
  so a player at 180 min gets λ=1.0 (full shrinkage to mean)
  and a player at 269 min gets λ~0.0 (minimal shrinkage)
- Position is the most common position per player per season from lineups.parquet
- Positions are grouped into 8 archetypes before computing means
"""

import polars as pl
import pandas as pd
from pathlib import Path
from typing import Dict, Optional

from player_metrics_config import PLAYER_METRICS
HARD_THRESHOLD = 270
SHRINKAGE_FLOOR = 180

# StatsBomb position → archetype grouping
POSITION_MAP = {
    "Goalkeeper":                  "GK",
    "Left Back":                   "FB",
    "Right Back":                  "FB",
    "Left Wing Back":              "FB",
    "Right Wing Back":             "FB",
    "Left Center Back":            "CB",
    "Right Center Back":           "CB",
    "Center Back":                 "CB",
    "Left Defensive Midfield":     "DM",
    "Right Defensive Midfield":    "DM",
    "Center Defensive Midfield":   "DM",
    "Left Midfield":               "CM",
    "Right Midfield":              "CM",
    "Left Center Midfield":        "CM",
    "Right Center Midfield":       "CM",
    "Center Midfield":             "CM",
    "Left Attacking Midfield":     "AM",
    "Right Attacking Midfield":    "AM",
    "Center Attacking Midfield":   "AM",
    "Left Wing":                   "W",
    "Right Wing":                  "W",
    "Left Center Forward":         "FW",
    "Right Center Forward":        "FW",
    "Center Forward":              "FW",
    "Secondary Striker":           "FW",
}


def build_position_lookup(lineups_path: str | Path) -> pl.DataFrame:
    """
    Build a player × season → archetype lookup from lineups.parquet.

    Uses the most common position per player per season.
    Season name normalisation matches the pipeline convention:
      single-year codes (2022, 2023, 2024) are kept as-is.

    Returns a Polars DataFrame with columns:
      player, season_name, position_archetype
    """
    lineups = pd.read_parquet(lineups_path, engine='fastparquet')

    # Keep only rows with both player and position
    lineups = lineups.dropna(subset=["player_name", "position_name"])

    # Map granular positions to archetypes
    lineups["archetype"] = lineups["position_name"].map(POSITION_MAP)
    lineups = lineups.dropna(subset=["archetype"])

    # We need season info — join from matches if available, otherwise
    # derive from match_id. lineups.parquet has match_id but not season_name.
    # Load matches to get season_name per match_id.
    lineups_path = Path(lineups_path)
    matches_path = lineups_path.parent / "matches.parquet"

    if matches_path.exists():
        matches = pd.read_parquet(matches_path)[["match_id", "season_name"]]
        lineups = lineups.merge(matches, on="match_id", how="left")
    else:
        # Fallback: no season info, use a single dummy season
        lineups["season_name"] = "unknown"

    # Most common archetype per player per season
    lookup = (
        lineups
        .groupby(["player_name", "season_name", "archetype"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .drop_duplicates(subset=["player_name", "season_name"])
        [["player_name", "season_name", "archetype"]]
        .rename(columns={"player_name": "player"})
    )

    return pl.from_pandas(lookup)


def _compute_shrinkage_lambda(
    minutes: float,
    floor: int = SHRINKAGE_FLOOR,
    threshold: int = HARD_THRESHOLD,
) -> float:
    """
    Compute shrinkage factor λ ∈ [0, 1].
    λ=1 at floor (maximum shrinkage), λ=0 at threshold (no shrinkage).
    """
    if minutes >= threshold:
        return 0.0
    if minutes <= floor:
        return 1.0
    return 1.0 - (minutes - floor) / (threshold - floor)


def apply_shrinkage(
    normalised: Dict[str, pl.DataFrame],
    lineups_path: str | Path,
    hard_threshold: int = HARD_THRESHOLD,
    shrinkage_floor: int = SHRINKAGE_FLOOR,
    verbose: bool = True,
) -> Dict[str, pl.DataFrame]:
    """
    Apply Bayesian shrinkage to flagged rows in all normalised DataFrames.

    For each norm column in each file:
    - Compute the positional mean per season per archetype
    - For shrinkage_flag=True rows, blend toward that mean
      proportional to how far below the threshold they sit

    Args:
        normalised: dict of {filename: DataFrame} from normalisation step
        lineups_path: path to lineups.parquet
        hard_threshold: minutes threshold (default 270)
        shrinkage_floor: minimum minutes floor (default 180)
        verbose: print per-file summary

    Returns:
        dict of {filename: DataFrame} with norm columns adjusted in-place
        for flagged rows, plus position_archetype column added
    """
    if verbose:
        print("\n" + "=" * 60)
        print("STEP 5: BAYESIAN SHRINKAGE")
        print(f"Threshold: {hard_threshold} min | Floor: {shrinkage_floor} min")
        print("=" * 60)

    # Build position lookup
    position_lookup = build_position_lookup(lineups_path)
    if verbose:
        print(f"  Position lookup: {len(position_lookup):,} player-season entries")
        print(f"  Archetypes: {sorted(position_lookup['archetype'].unique().to_list())}")

    shrunk = {}

    for filename, df in normalised.items():
        # Find norm columns for this file
        norm_cols = [
            f"{k}_norm" for k, v in PLAYER_METRICS.items()
            if v["file"] == filename and f"{k}_norm" in df.columns
        ]

        if not norm_cols:
            shrunk[filename] = df
            continue

        # Join position archetype
        join_keys = ["player", "season_name"] if "season_name" in df.columns else ["player"]
        df = df.join(
            position_lookup.rename({"archetype": "position_archetype"}),
            on=join_keys,
            how="left"
        )

        # Players with no position matched → assign "CM" as neutral fallback
        df = df.with_columns(
            pl.col("position_archetype").fill_null("CM")
        )

        # Check if shrinkage_flag exists
        if "shrinkage_flag" not in df.columns:
            if verbose:
                print(f"  [shrink] {filename}: no shrinkage_flag column, skipping")
            shrunk[filename] = df
            continue

        n_flagged = int(df["shrinkage_flag"].sum())
        if n_flagged == 0:
            if verbose:
                print(f"  [shrink] {filename}: 0 flagged rows, skipping")
            shrunk[filename] = df
            continue

        # Get minutes column for λ computation
        minutes_col = None
        for cfg in PLAYER_METRICS.values():
            if cfg["file"] == filename:
                fc = cfg.get("filter_column", "minutes_played")
                if fc in df.columns:
                    minutes_col = fc
                    break

        # Process each norm column
        df_pd = df.to_pandas()

        for norm_col in norm_cols:
            if norm_col not in df_pd.columns:
                continue

            # Compute positional mean per season per archetype
            group_cols = ["season_name", "position_archetype"] if "season_name" in df_pd.columns else ["position_archetype"]
            pos_means = (
                df_pd[~df_pd["shrinkage_flag"]]  # use only non-flagged rows for mean
                .groupby(group_cols)[norm_col]
                .mean()
                .reset_index()
                .rename(columns={norm_col: f"{norm_col}_pos_mean"})
            )

            df_pd = df_pd.merge(pos_means, on=group_cols, how="left")

            # Apply shrinkage to flagged rows
            flagged_mask = df_pd["shrinkage_flag"]

            if minutes_col and minutes_col in df_pd.columns:
                lambdas = df_pd.loc[flagged_mask, minutes_col].apply(
                    lambda m: _compute_shrinkage_lambda(m, shrinkage_floor, hard_threshold)
                )
            else:
                # No minutes info — use fixed λ=0.5
                lambdas = pd.Series([0.5] * flagged_mask.sum(), index=df_pd[flagged_mask].index)

            raw = df_pd.loc[flagged_mask, norm_col]
            pos_mean = df_pd.loc[flagged_mask, f"{norm_col}_pos_mean"].fillna(0.5)

            df_pd.loc[flagged_mask, norm_col] = (1 - lambdas) * raw + lambdas * pos_mean

            # Drop helper column
            df_pd = df_pd.drop(columns=[f"{norm_col}_pos_mean"])

        if verbose:
            print(f"  [shrink] {filename}: {n_flagged} rows shrunk across {len(norm_cols)} norm columns")

        shrunk[filename] = pl.from_pandas(df_pd)

    return shrunk


def shrinkage_summary(shrunk: Dict[str, pl.DataFrame]) -> None:
    """Print a concise summary of shrinkage results."""
    print("\n" + "=" * 60)
    print("SHRINKAGE SUMMARY")
    print("=" * 60)
    for filename, df in shrunk.items():
        norm_cols = [c for c in df.columns if c.endswith("_norm")]
        n_flagged = int(df["shrinkage_flag"].sum()) if "shrinkage_flag" in df.columns else 0
        archetypes = sorted(df["position_archetype"].unique().to_list()) if "position_archetype" in df.columns else []
        print(f"  {filename}")
        print(f"    {len(df):,} rows | {n_flagged} shrinkage-flagged | archetypes: {archetypes}")
        for col in norm_cols:
            s = df[col].drop_nulls()
            print(f"    {col}: min={s.min():.3f}, median={s.median():.3f}, max={s.max():.3f}")