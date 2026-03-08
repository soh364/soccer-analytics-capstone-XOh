"""
Step 7 & 8: Intra-Archetype Percentile Scoring + Composite Score Assembly

Order of operations:
1. Join all files on (player, season_name) to build a unified player-season table
2. Decay-weighted collapse: for each metric, compute weighted mean across seasons
   → one row per player
3. Intra-archetype percentile rank for each metric (0-100)
4. Trait category scores: mean of metric percentiles within each category
5. Composite score: mean of 4 trait category scores (equal category weights)

Category weights (equal):
  Mobility_Intensity   25%  → defensive_actions, high_turnovers, pressure_volume, pressure_success
  Progression          25%  → progressive_passes, progressive_carries, packing
  Control              25%  → network_centrality
  Final_Third_Output   25%  → finishing_quality, xg_volume, xg_chain, xg_buildup, team_involvement
"""

import polars as pl
import pandas as pd
import numpy as np
from typing import Dict, Optional
from player_metrics_config import PLAYER_METRICS, TRAIT_CATEGORIES

# Metric key → norm column name
METRIC_NORM_COLS = {k: f"{k}_norm" for k in PLAYER_METRICS}

# Which file each norm column lives in
METRIC_FILE = {k: v["file"] for k, v in PLAYER_METRICS.items()}

# Join keys used across files
JOIN_KEYS = ["player", "team", "season_name", "position_archetype",
             "archetype_label", "decay_weight", "shrinkage_flag"]

SEASON_NORMALISE = {
    "2022": "2021/2022",
    "2023": "2022/2023",
    "2024": "2023/2024",
}

def _normalise_seasons(df: pd.DataFrame) -> pd.DataFrame:
    """Map single-year tournament codes to club season format."""
    if "season_name" in df.columns:
        df = df.copy()
        df["season_name"] = df["season_name"].replace(SEASON_NORMALISE)
    return df


def _build_unified_table(segmented: Dict[str, pl.DataFrame]) -> pd.DataFrame:
    """
    Join all segmented files into a single player × season table.

    Each file contributes its norm columns. Files are joined on
    (player, season_name) with outer joins to preserve all players.
    decay_weight and position_archetype are taken from the first file
    that has them for a given player-season.
    """
    base = None

    for filename, df in segmented.items():
        # Get norm columns for this file
        norm_cols = [
            f"{k}_norm" for k, v in PLAYER_METRICS.items()
            if v["file"] == filename and f"{k}_norm" in df.columns
        ]

        if not norm_cols:
            continue

        # Select only what we need
        keep_cols = ["player", "team", "season_name", "position_archetype",
             "archetype_label", "decay_weight", "shrinkage_flag"] + norm_cols
        
        # Keep only columns that exist
        keep_cols = [c for c in keep_cols if c in df.columns]
        sub = df.select(keep_cols).to_pandas()
        sub = _normalise_seasons(sub) 

        if base is None:
            base = sub
        else:
            # Outer join on player + season_name
            # For overlapping metadata cols, keep from left (base)
            meta_cols = ["position_archetype", "archetype_label",
                         "decay_weight", "shrinkage_flag"]
            right_cols = ["player", "team", "season_name"] + norm_cols
            right_cols = [c for c in right_cols if c in sub.columns]

            # to
            join_on = [c for c in ["player", "team", "season_name"] if c in base.columns and c in sub.columns]
            base = base.merge(
                sub[right_cols + [c for c in ["team"] if c in sub.columns and c not in right_cols]],
                on=join_on,
                how="outer"
            )

            # Fill missing metadata from right side
            for col in meta_cols:
                if col in base.columns and f"{col}_x" in base.columns:
                    base[col] = base[f"{col}_x"].fillna(base.get(f"{col}_y", None))
                    base = base.drop(columns=[c for c in [f"{col}_x", f"{col}_y"] if c in base.columns])

    return base


def _decay_collapse(unified: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse player × season rows into a single row per player
    using decay-weighted mean for each norm column.

    decay_weight per season: 2023/24=1.0, 2022/23=0.75, 2021/22=0.50
    """
    norm_cols = [c for c in unified.columns if c.endswith("_norm")]

    # Ensure decay_weight exists
    if "decay_weight" not in unified.columns:
        unified["decay_weight"] = 1.0

    # For each norm column, compute weighted mean ignoring nulls
    def weighted_mean(group):
        result = {}
        for col in norm_cols:
            vals = group[col]
            weights = group["decay_weight"]
            mask = vals.notna()
            if mask.sum() == 0:
                result[col] = np.nan
            else:
                result[col] = np.average(vals[mask], weights=weights[mask])
        return pd.Series(result)

    # Take modal value for categoricals
    def modal(series):
        mode = series.mode()
        return mode.iloc[0] if len(mode) > 0 else series.iloc[0]

    # Group by player
    meta_agg = (
        unified.groupby("player")
        .agg(
            team=("team", modal) if "team" in unified.columns else ("player", "first"),
            position_archetype=("position_archetype", modal),
            archetype_label=("archetype_label", modal),
            seasons_present=("season_name", lambda x: sorted(x.dropna().unique().tolist())),
            decay_weight_max=("decay_weight", "max"),
        )
        .reset_index()
    )

    norm_agg = (
        unified.groupby("player")
        .apply(weighted_mean)
        .reset_index()
    )

    collapsed = meta_agg.merge(norm_agg, on="player", how="left")
    return collapsed


def _intra_archetype_percentile(
    collapsed: pd.DataFrame,
    norm_col: str,
) -> pd.Series:
    """
    Compute intra-archetype percentile rank (0-100) for a norm column.
    Each player is ranked only against players of the same archetype.
    """
    result = pd.Series(np.nan, index=collapsed.index)

    for archetype in collapsed["position_archetype"].dropna().unique():
        mask = collapsed["position_archetype"] == archetype
        vals = collapsed.loc[mask, norm_col]
        valid = vals.notna()

        if valid.sum() <= 1:
            result.loc[mask & valid] = 50.0
            continue

        # Percentile rank within archetype
        ranks = vals[valid].rank(pct=True) * 100
        result.loc[mask & valid] = ranks.values

    return result


def _compute_category_score(
    scored: pd.DataFrame,
    category: str,
    metric_keys: list,
) -> pd.Series:
    """
    Compute trait category score as mean of available metric percentiles.
    For packing specifically, only include if the player has sufficient
    packing data (packing_pct is not NaN AND packing_norm > 0).
    Returns NaN if no metrics are available for a player.
    """
    percentile_cols = [f"{k}_pct" for k in metric_keys if f"{k}_pct" in scored.columns]

    if not percentile_cols:
        return pd.Series(np.nan, index=scored.index)

    # For packing: null out packing_pct where packing_norm is effectively zero
    # (player had no meaningful packing data — tournament coverage gap)
    result_df = scored[percentile_cols].copy()
    if "packing_pct" in result_df.columns and "packing_norm" in scored.columns:
        no_packing = scored["packing_norm"].isna() | (scored["packing_norm"] == 0)
        result_df.loc[no_packing, "packing_pct"] = np.nan

    return result_df.mean(axis=1, skipna=True)

def build_scores(
    segmented: Dict[str, pl.DataFrame],
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Build the full scoring table from segmented DataFrames.

    Returns a DataFrame with one row per player containing:
    - Player metadata (player, team, position_archetype, archetype_label)
    - seasons_present: list of seasons the player appeared in
    - {metric}_norm: decay-weighted norm score for each metric
    - {metric}_pct: intra-archetype percentile (0-100)
    - {category}_score: trait category score (mean of metric percentiles)
    - composite_score: mean of 4 category scores (0-100)
    """
    if verbose:
        print("\n" + "=" * 60)
        print("STEP 7 & 8: SCORING + COMPOSITE ASSEMBLY")
        print("=" * 60)

    # Step 1: Build unified table
    if verbose:
        print("  [1/4] Building unified player-season table...")
    unified = _build_unified_table(segmented)
    if verbose:
        print(f"         {len(unified):,} player-season rows, "
              f"{unified['player'].nunique():,} unique players")

    # Step 2: Decay-weighted collapse
    if verbose:
        print("  [2/4] Collapsing seasons with decay weighting...")
    collapsed = _decay_collapse(unified)
    if verbose:
        print(f"         {len(collapsed):,} players after collapse")

    # Step 3: Intra-archetype percentile per metric
    if verbose:
        print("  [3/4] Computing intra-archetype percentiles...")

    norm_cols = [c for c in collapsed.columns if c.endswith("_norm")]
    for norm_col in norm_cols:
        metric_key = norm_col.replace("_norm", "")
        pct_col = f"{metric_key}_pct"
        collapsed[pct_col] = _intra_archetype_percentile(collapsed, norm_col)

    if verbose:
        pct_cols = [c for c in collapsed.columns if c.endswith("_pct")]
        print(f"         {len(pct_cols)} percentile columns computed")

    # Step 4: Trait category scores + composite
    if verbose:
        print("  [4/4] Computing category scores and composite...")

    category_scores = []
    for category, metric_keys in TRAIT_CATEGORIES.items():
        score_col = f"{category}_score"
        collapsed[score_col] = _compute_category_score(collapsed, category, metric_keys)
        category_scores.append(score_col)
        if verbose:
            valid = collapsed[score_col].notna().sum()
            median = collapsed[score_col].median()
            print(f"         {category}: {valid:,} players, median={median:.1f}")

    # Composite: mean of available category scores
    collapsed["composite_score"] = collapsed[category_scores].mean(axis=1, skipna=True)

    if verbose:
        print(f"\n  Composite score: {len(collapsed):,} players")
        print(f"  min={collapsed['composite_score'].min():.1f}, "
              f"median={collapsed['composite_score'].median():.1f}, "
              f"max={collapsed['composite_score'].max():.1f}")

    # Sort by composite score descending
    collapsed = collapsed.sort_values("composite_score", ascending=False).reset_index(drop=True)
    collapsed.index = collapsed.index + 1  # 1-based rank
    collapsed.index.name = "rank"

    # Require at least 2 out of 4 category scores to be non-NaN
    min_categories = 3
    category_coverage = collapsed[category_scores].notna().sum(axis=1)
    collapsed = collapsed[category_coverage >= min_categories].copy()

    return collapsed


def scoring_summary(scored: pd.DataFrame, top_n: int = 20) -> None:
    """Print top N players by composite score."""
    print("\n" + "=" * 60)
    print(f"TOP {top_n} PLAYERS — COMPOSITE SCORE")
    print("=" * 60)

    display_cols = ["player", "position_archetype", "composite_score",
                    "Mobility_Intensity_score", "Progression_score",
                    "Control_score", "Final_Third_Output_score"]
    display_cols = [c for c in display_cols if c in scored.columns]

    print(scored[display_cols].head(top_n).to_string())

    print("\n" + "=" * 60)
    print("ARCHETYPE BREAKDOWN — MEDIAN COMPOSITE SCORE")
    print("=" * 60)
    breakdown = (
        scored.groupby("position_archetype")["composite_score"]
        .agg(["count", "median", "mean", "std"])
        .round(1)
        .sort_values("median", ascending=False)
    )
    print(breakdown.to_string())