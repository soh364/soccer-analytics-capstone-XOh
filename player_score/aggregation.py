"""
aggregation.py
--------------
Aggregates raw player metric files from match-level to player-season level.

Granularity map:
  - Match-level  : xg_chain, xg_buildup, network_centrality,
                   defensive__profile, defensive__pressures
  - Season-level : xg__player__totals, progression__player__profile
  - Special      : packing (tournament snapshot, no minutes filtering)

Output: dict[filename -> pl.DataFrame] at player x season granularity.
"""

import polars as pl
from typing import Optional

# ---------------------------------------------------------------------------
# Column-level aggregation rules
# ---------------------------------------------------------------------------

# Columns that should be SUMMED when aggregating match -> season
SUM_COLS = {
    # xg_chain
    "possessions_with_shot", "xg_chain", "shots_in_chain", "goals_in_chain",
    "total_touches_in_chains",
    # xg_buildup
    "possessions_with_buildup", "xg_buildup", "total_touches_in_buildup",
    # network_centrality
    "passes_made", "passes_received", "total_pass_involvement",
    # defensive__profile
    "tackles", "interceptions", "pressures", "blocks", "clearances",
    "ball_recoveries", "fouls_committed", "duels_won", "total_duels",
    "defensive_third_actions", "middle_third_actions", "attacking_third_actions",
    "high_turnovers", "total_defensive_actions",
    # defensive__pressures
    "total_pressures", "counterpresses", "pressure_regains", "high_pressures",
    # shared
    "minutes_played",
}

# Rate / p90 columns that need WEIGHTED MEAN by minutes_played
WEIGHTED_MEAN_COLS = {
    "xg_chain_per90",
    "team_involvement_pct",
    "xg_buildup_per90",
    "avg_xg_per_possession",       # xg_buildup
    "network_involvement_pct",
    "degree_centrality",
    "pressures_per_90",
    "pressure_success_pct",
    # defensive profile rates
    "duel_win_pct",
    "pressure_pct",
    "tackle_interception_pct",
    "high_turnover_pct",
    "attacking_third_pct",
}

# Categorical columns — take MODE (most frequent value across matches)
MODE_COLS = {
    "defensive_profile",
    "unique_receivers",
    "unique_passers",
}

# Columns to DROP entirely after aggregation
DROP_COLS = {"match_id"}


# ---------------------------------------------------------------------------
# Core aggregation helpers
# ---------------------------------------------------------------------------

def _weighted_mean_expr(col: str) -> pl.Expr:
    """Weighted mean of `col` by minutes_played."""
    return (
        (pl.col(col) * pl.col("minutes_played")).sum()
        / pl.col("minutes_played").sum()
    ).alias(col)


def _mode_expr(col: str) -> pl.Expr:
    """Most frequent value (mode) for a categorical column."""
    return pl.col(col).mode().first().alias(col)

def build_team_league_map(matches: pl.DataFrame) -> pl.DataFrame:
    home = matches.select([
        pl.col("home_team").alias("team"),
        pl.col("competition").alias("competition_name"), 
        pl.col("season_name"),
    ])
    away = matches.select([
        pl.col("away_team").alias("team"),
        pl.col("competition").alias("competition_name"), 
        pl.col("season_name"),
    ])
    return pl.concat([home, away]).unique()


def aggregate_match_to_season(
    df: pl.DataFrame,
    filename: str,
) -> pl.DataFrame:
    """
    Aggregate a match-level DataFrame to player x season level.

    Groups by (player, team, season_name, season_year).
    Applies sum / weighted-mean / mode rules per column.
    Drops match_id and any unrecognised columns with a warning.
    """
    group_keys = ["player", "team", "season_name", "season_year"]

    # Validate minutes column exists for weighting
    if "minutes_played" not in df.columns:
        raise ValueError(
            f"[{filename}] Missing 'minutes_played' — cannot aggregate match-level file."
        )

    present_cols = set(df.columns) - set(group_keys) - DROP_COLS

    agg_exprs = []
    unhandled = []

    for col in present_cols:
        if col in SUM_COLS:
            agg_exprs.append(pl.col(col).sum().alias(col))
        elif col in WEIGHTED_MEAN_COLS:
            agg_exprs.append(_weighted_mean_expr(col))
        elif col in MODE_COLS:
            agg_exprs.append(_mode_expr(col))
        else:
            unhandled.append(col)

    if unhandled:
        print(
            f"  ⚠️  [{filename}] Unhandled columns (dropped): {unhandled}"
        )

    aggregated = df.group_by(group_keys).agg(agg_exprs)
    return aggregated


# ---------------------------------------------------------------------------
# File-specific handlers
# ---------------------------------------------------------------------------

def _handle_season_level(df: pl.DataFrame, filename: str) -> pl.DataFrame:
    """
    Season-level files need no aggregation.
    Standardise column names and ensure season_year exists.
    """
    # progression__player__profile uses 'total_mins' — rename for consistency
    if "total_mins" in df.columns:
        df = df.rename({"total_mins": "minutes_played"})

    if "season_year" not in df.columns and "season_name" in df.columns:
        df = df.with_columns(
            pl.col("season_name")
            .map_elements(_parse_season_year, return_dtype=pl.Int32)
            .alias("season_year")
        )

    return df


def _handle_packing(df: pl.DataFrame) -> pl.DataFrame:
    """
    Packing is a tournament snapshot — single-year season_name (e.g. '2022').
    No minutes column, no threshold filtering.
    Aggregated to player level (already roughly there, but may have duplicates).
    """
    group_keys = ["player", "team", "season_name", "season_year"]

    # Keep only the columns we need
    keep = group_keys + [
        "total_passes", "total_opponents_packed",
        "avg_packing_per_pass", "progressive_passes",
        "avg_packing_progressive",
    ]
    existing_keep = [c for c in keep if c in df.columns]
    df = df.select(existing_keep)

    # Aggregate in case of duplicate player rows
    agg_exprs = [
        pl.col("total_passes").sum(),
        pl.col("total_opponents_packed").sum(),
        pl.col("progressive_passes").sum(),
        (
            (pl.col("avg_packing_per_pass") * pl.col("total_passes")).sum()
            / pl.col("total_passes").sum()
        ).alias("avg_packing_per_pass"),
        (
            (pl.col("avg_packing_progressive") * pl.col("progressive_passes")).sum()
            / pl.col("progressive_passes").sum()
        ).alias("avg_packing_progressive"),
    ]

    present_agg = [e for e in agg_exprs if any(
        c in df.columns for c in [str(e)]
    )]

    # Safe column-by-column aggregation
    safe_agg = []
    for col, agg_fn in [
        ("total_passes", "sum"),
        ("total_opponents_packed", "sum"),
        ("progressive_passes", "sum"),
    ]:
        if col in df.columns:
            safe_agg.append(pl.col(col).sum().alias(col))

    # Weighted means for avg columns
    if "avg_packing_per_pass" in df.columns and "total_passes" in df.columns:
        safe_agg.append(
            (
                (pl.col("avg_packing_per_pass") * pl.col("total_passes")).sum()
                / pl.col("total_passes").sum()
            ).alias("avg_packing_per_pass")
        )
    if "avg_packing_progressive" in df.columns and "progressive_passes" in df.columns:
        safe_agg.append(
            (
                (pl.col("avg_packing_progressive") * pl.col("progressive_passes")).sum()
                / pl.col("progressive_passes").sum()
            ).alias("avg_packing_progressive")
        )

    available_group_keys = [k for k in group_keys if k in df.columns]
    return df.group_by(available_group_keys).agg(safe_agg)


# ---------------------------------------------------------------------------
# Routing logic
# ---------------------------------------------------------------------------

# Files that are already at season level — no aggregation needed
SEASON_LEVEL_FILES = {
    "xg__player__totals.csv",
    "progression__player__profile.csv",
}

# Files that are match-level — need aggregation
MATCH_LEVEL_FILES = {
    "advanced__player__xg_chain.csv",
    "advanced__player__xg_buildup.csv",
    "advanced__player__network_centrality.csv",
    "defensive__player__profile.csv",
    "defensive__player__pressures.csv",
}

PACKING_FILE = "advanced__player__packing.csv"


def _parse_season_year(season_name: str) -> int:
    """'2023/2024' -> 2024, '2022' -> 2022"""
    parts = str(season_name).replace("-", "/").split("/")
    return max(int(p) for p in parts if p.isdigit())


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def aggregate_all(
    raw_data: dict[str, pl.DataFrame],
    matches: pl.DataFrame,
    verbose: bool = True,
) -> dict[str, pl.DataFrame]:
    """
    Takes the raw dict from PlayerDataLoader and returns a clean dict
    with every file at player x season granularity.

    Parameters
    ----------
    raw_data : dict returned by load_player_data_for_scoring()
    matches  : matches DataFrame with competition_name and season_name
    verbose  : print summary per file

    Returns
    -------
    dict[filename -> aggregated pl.DataFrame]
    """
    team_league_map_pl = build_team_league_map(matches)

    aggregated = {}

    if verbose:
        print(f"\n{'='*70}")
        print("AGGREGATION: match-level -> player x season")
        print(f"{'='*70}")

    for filename, df in raw_data.items():

        if verbose:
            print(f"\n  [{filename}]")

        try:
            if filename in SEASON_LEVEL_FILES:
                result = _handle_season_level(df, filename)
                if verbose:
                    print(f"    Season-level — no aggregation needed")

            elif filename == PACKING_FILE:
                result = _handle_packing(df)
                if verbose:
                    print(f"    Packing (tournament snapshot) — deduplicated")

            elif filename in MATCH_LEVEL_FILES:
                result = aggregate_match_to_season(df, filename)
                if verbose:
                    print(f"    Match-level — aggregated to player x season")

            else:
                if "match_id" in df.columns:
                    print(f"    ⚠️  Unknown file, attempting match-level aggregation")
                    result = aggregate_match_to_season(df, filename)
                else:
                    print(f"    ⚠️  Unknown file, passing through unchanged")
                    result = df

            # Join competition_name before storing
            if "team" in result.columns and "season_name" in result.columns:
                result = result.join(
                    team_league_map_pl.select(["team", "season_name", "competition_name"]),
                    on=["team", "season_name"],
                    how="left"
                )

            aggregated[filename] = result

            if verbose:
                n_players = result["player"].n_unique() if "player" in result.columns else "N/A"
                seasons = (
                    sorted(result["season_name"].unique().to_list())
                    if "season_name" in result.columns else "N/A"
                )
                print(f"    → {len(result):,} rows | {n_players} unique players | seasons: {seasons}")

        except Exception as e:
            print(f"    ❌ ERROR: {e}")
            aggregated[filename] = df

    if verbose:
        print(f"\n{'='*70}")
        print(f"DONE — {len(aggregated)} files aggregated")
        print(f"{'='*70}")

    return aggregated