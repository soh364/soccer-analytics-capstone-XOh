"""
tc_data.py
──────────
Data loading, feature engineering, and team-level aggregation
for the 2026 World Cup Tactical Clustering analysis.

Depends on:
    eda/analysis/data_loader.py  →  load_tournament_data_8d()
"""

import sys
import functools
from pathlib import Path

import polars as pl
import pandas as pd
import numpy as np

# ── Path setup ────────────────────────────────────────────────────────────────
# Notebook lives in  <project_root>/tactical_clustering/
# data_loader lives in  <project_root>/eda/analysis/
def _add_eda_to_path():
    notebook_dir = Path(__file__).resolve()
    project_root = notebook_dir.parent
    eda_root     = project_root / 'eda'
    if str(eda_root) not in sys.path:
        sys.path.insert(0, str(eda_root))

_add_eda_to_path()
from analysis.data_loader import load_tournament_data_8d


# ── Constants ─────────────────────────────────────────────────────────────────
CORE_METRICS = [
    'ppda',
    'defensive_line_height',
    'field_tilt_pct',
    'possession_pct',
    'progressive_carry_pct',
    'epr',
    'npxg',
    'avg_xg_per_buildup_possession',
]

CLUSTER_FEATURES = CORE_METRICS + ['ppda_std', 'possession_pct_std']


# ── Functions ─────────────────────────────────────────────────────────────────
def load_and_merge(tournament_key: str = 'men_tourn_2022_24',
                   verbose: bool = True) -> pl.DataFrame:
    """
    Load all 8 metric files and merge into a single match-level DataFrame.

    Returns
    -------
    pl.DataFrame
        Columns: [match_id, team, ppda, defensive_line_height, field_tilt_pct,
                  possession_pct, progressive_carries, progressive_passes,
                  epr, npxg, avg_xg_per_buildup_possession]
    """
    metrics_raw = load_tournament_data_8d(tournament_key, verbose=verbose)

    dfs = list(metrics_raw.values())
    metrics = functools.reduce(
        lambda a, b: a.join(b, on=['match_id', 'team'], how='left'),
        dfs
    )

    # Derive progressive_carry_pct — carries as % of total progressive actions
    metrics = metrics.with_columns(
        (pl.col('progressive_carries') /
         (pl.col('progressive_carries') + pl.col('progressive_passes')) * 100
        ).alias('progressive_carry_pct')
    )

    if verbose:
        print(f'Merged shape      : {metrics.shape}')
        print(f'All metrics found : {all(c in metrics.columns for c in CORE_METRICS)}')

    return metrics


def aggregate_to_team_level(metrics: pl.DataFrame) -> pl.DataFrame:
    """
    Collapse match-level rows to one row per team.

    Computes:
    - Mean of all 8 CORE_METRICS
    - Std dev of PPDA and possession_pct (captures tactical volatility)
    - Match count per team

    Returns
    -------
    pl.DataFrame  shape (n_teams, 12)
    """
    team_metrics = (
        metrics
        .group_by('team')
        .agg([
            *[pl.col(c).mean().alias(c) for c in CORE_METRICS],
            pl.col('ppda').std().alias('ppda_std'),
            pl.col('possession_pct').std().alias('possession_pct_std'),
            pl.col('match_id').count().alias('n_matches'),
        ])
        .sort('team')
    )
    return team_metrics


def run_data_quality_audit(team_metrics: pl.DataFrame) -> None:
    """
    Print a full data quality report: shape, nulls, duplicates,
    low-match teams, and 3-sigma outliers per CORE_METRIC.
    """
    print('=== SHAPE ===')
    print(team_metrics.shape)

    print('\n=== NULL CHECK ===')
    print(team_metrics.null_count())

    print('\n=== DUPLICATE TEAMS ===')
    print(f'Total rows   : {len(team_metrics)}')
    print(f'Unique teams : {team_metrics["team"].n_unique()}')

    print('\n=== MATCH COUNT DISTRIBUTION ===')
    for threshold in [1, 2]:
        label = 'only 1 match' if threshold == 1 else '2 matches (low confidence)'
        subset = team_metrics.filter(pl.col('n_matches') == threshold).select(['team', 'n_matches'])
        print(f'Teams with {label}: {len(subset)}')
        if len(subset):
            print(subset)

    print('\n=== 3-SIGMA OUTLIER CHECK ===')
    found_any = False
    for col in CORE_METRICS:
        mean = team_metrics[col].mean()
        std  = team_metrics[col].std()
        outliers = team_metrics.filter(
            (pl.col(col) > mean + 3 * std) | (pl.col(col) < mean - 3 * std)
        ).select(['team', col, 'n_matches'])
        if len(outliers):
            found_any = True
            print(f'\n{col} — {len(outliers)} outlier(s):')
            print(outliers)
    if not found_any:
        print('No 3-sigma outliers found.')


def load_pipeline(tournament_key: str = 'men_tourn_2022_24',
                  verbose: bool = True, exclude_teams=None):
    """
    Convenience wrapper: load → merge → aggregate → audit.

    Returns
    -------
    (metrics, team_metrics) : tuple of pl.DataFrame
    """
    metrics      = load_and_merge(tournament_key, verbose=verbose)
    team_metrics = aggregate_to_team_level(metrics)
    print(f'\nTeam-level shape  : {team_metrics.shape}')
    print(f'Teams             : {team_metrics["team"].n_unique()}')
    run_data_quality_audit(team_metrics)
    return metrics, team_metrics
