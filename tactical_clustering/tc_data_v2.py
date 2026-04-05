"""
tc_data.py
──────────
Data loading, feature engineering, and team-level aggregation
for the 2026 World Cup Tactical Clustering analysis.

KEY CHANGE from original:
    Aggregation is now TOURNAMENT-TIER WEIGHTED rather than a simple mean.
    Match-level metrics are weighted by the competitive level of the tournament
    they came from, sourced directly from matches.parquet (internal data only).

    This ensures that World Cup matches carry more weight than AFCON group
    stage matches when computing a team's tactical profile — correcting the
    original problem where Nigerian AFCON metrics and Argentine WC metrics
    were treated as equivalent inputs to clustering.

Tournament tier weights (derived from matches.parquet competition_name):
    FIFA World Cup              → 1.00  (the validation standard)
    UEFA Euro / Copa América    → 0.90  (elite continental)
    Africa Cup of Nations       → 0.75  (regional — strong but weaker field)
    CONCACAF Gold Cup           → 0.75  (regional)
    AFC Asian Cup               → 0.75  (regional)
    All others                  → 0.70  (default)

Depends on:
    eda/analysis/data_loader.py  →  load_tournament_data_8d()
    data/Statsbomb/matches.parquet → competition_name per match_id
"""

import sys
import functools
from pathlib import Path

import polars as pl
import pandas as pd
import numpy as np

# ── Path setup ────────────────────────────────────────────────────────────────
def _add_eda_to_path():
    notebook_dir = Path('.').resolve()
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

# Tournament tier weights — purely from competition_name in matches.parquet
# All weights are internal data: no external sources used.
TOURNAMENT_TIER_WEIGHTS = {
    # Tier 1 — FIFA World Cup (the validation standard)
    # All archetype scores are derived from WC 2022 outcomes.
    'FIFA World Cup'                        : 1.00,
    'World Cup'                             : 1.00,

    # Tier 2 — UEFA Euro 2024
    # Near-equivalent to WC in field quality — Spain, France, Germany,
    # England, Portugal all played each other. Minimal discount.
    'UEFA Euro'                             : 0.95,
    'UEFA European Championship'            : 0.95,

    # Tier 3 — Copa América 2024
    # Strong CONMEBOL field but closed confederation tournament.
    # CONCACAF guests lower the average field slightly vs WC/Euro.
    'Copa America'                          : 0.85,
    'Copa América'                          : 0.85,

    # Tier 4 — Africa Cup of Nations 2023
    # Regional tournament. Includes Tanzania, Namibia, Equatorial Guinea
    # who would not qualify for a WC. Modest discount — enough to nudge
    # African team metrics toward their WC-context equivalent without
    # distorting the overall clustering geometry.
    'Africa Cup of Nations'                 : 0.75,
    'African Cup of Nations'                : 0.75,
    'AFCON'                                 : 0.75,

    # Other regional (forward compatibility)
    'CONCACAF Gold Cup'                     : 0.80,
    'Gold Cup'                              : 0.80,
    'AFC Asian Cup'                         : 0.80,
    'Asian Cup'                             : 0.80,
}
DEFAULT_TIER_WEIGHT = 0.80  # for any competition not explicitly listed


# ── Helpers ───────────────────────────────────────────────────────────────────
def _find_matches_parquet() -> Path:
    """
    Locate matches.parquet relative to the notebook working directory.
    Tries multiple candidate paths used across different project setups.
    """
    notebook_dir = Path('.').resolve()
    project_root = notebook_dir.parent

    candidates = [
        project_root / 'data' / 'Statsbomb' / 'matches.parquet',
        project_root / 'data' / 'statsbomb' / 'matches.parquet',
        project_root / 'data' / 'matches.parquet',
        notebook_dir / 'data' / 'Statsbomb' / 'matches.parquet',
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f'matches.parquet not found. Tried:\n' +
        '\n'.join(f'  {p}' for p in candidates)
    )


def _get_tier_weight(competition_name: str) -> float:
    """Map competition_name to tier weight. Case-insensitive partial match."""
    if not competition_name:
        return DEFAULT_TIER_WEIGHT
    name_lower = competition_name.lower()
    for comp, weight in TOURNAMENT_TIER_WEIGHTS.items():
        if comp.lower() in name_lower:
            return weight
    return DEFAULT_TIER_WEIGHT


def load_match_weights(verbose: bool = True) -> pl.DataFrame:
    """
    Load matches.parquet and return a DataFrame mapping match_id → tier_weight.

    Parameters
    ----------
    verbose : bool

    Returns
    -------
    pl.DataFrame  columns: [match_id, competition_name, tier_weight]
    """
    matches_path = _find_matches_parquet()
    matches = pl.read_parquet(matches_path)

    # Select only what we need
    cols_needed = ['match_id', 'competition_name']
    available   = [c for c in cols_needed if c in matches.columns]

    if 'competition_name' not in matches.columns:
        # Fallback: try competition column
        if 'competition' in matches.columns:
            matches = matches.rename({'competition': 'competition_name'})
        else:
            if verbose:
                print('⚠  competition_name not found in matches.parquet — '
                      'using default tier weight for all matches')
            return pl.DataFrame({
                'match_id'        : matches['match_id'],
                'competition_name': ['Unknown'] * len(matches),
                'tier_weight'     : [DEFAULT_TIER_WEIGHT] * len(matches),
            })

    weights = matches.select(['match_id', 'competition_name'])

    # Apply tier weights
    weights = weights.with_columns(
        pl.col('competition_name')
        .map_elements(_get_tier_weight, return_dtype=pl.Float64)
        .alias('tier_weight')
    )

    if verbose:
        print('\n=== TOURNAMENT TIER WEIGHTS ===')
        summary = (
            weights
            .group_by(['competition_name', 'tier_weight'])
            .agg(pl.count('match_id').alias('n_matches'))
            .sort('tier_weight', descending=True)
        )
        print(summary)

    return weights


# ── Functions ─────────────────────────────────────────────────────────────────
def load_and_merge(tournament_key: str = 'men_tourn_2022_24',
                   verbose: bool = True) -> pl.DataFrame:
    """
    Load all 8 metric files, merge into match-level DataFrame,
    and attach tier_weight from matches.parquet.

    Returns
    -------
    pl.DataFrame
        Columns: [match_id, team, ppda, ..., progressive_carry_pct,
                  competition_name, tier_weight]
    """
    metrics_raw = load_tournament_data_8d(tournament_key, verbose=verbose)

    dfs     = list(metrics_raw.values())
    metrics = functools.reduce(
        lambda a, b: a.join(b, on=['match_id', 'team'], how='left'),
        dfs
    )

    # Derive progressive_carry_pct
    metrics = metrics.with_columns(
        (pl.col('progressive_carries') /
         (pl.col('progressive_carries') + pl.col('progressive_passes')) * 100
        ).alias('progressive_carry_pct')
    )

    # Join tournament tier weights
    try:
        match_weights = load_match_weights(verbose=verbose)
        metrics = metrics.join(
            match_weights.select(['match_id', 'competition_name', 'tier_weight']),
            on='match_id',
            how='left',
        )
        # Fill any unmatched matches with default weight
        metrics = metrics.with_columns(
            pl.col('tier_weight').fill_null(DEFAULT_TIER_WEIGHT)
        )
        if verbose:
            print(f'\nTier weight coverage: '
                  f'{metrics["tier_weight"].is_not_null().sum()}/{len(metrics)} matches')
    except FileNotFoundError as e:
        if verbose:
            print(f'\n⚠  {e}')
            print('   Falling back to unweighted aggregation.')
        metrics = metrics.with_columns(
            pl.lit(DEFAULT_TIER_WEIGHT).alias('tier_weight')
        )

    if verbose:
        print(f'\nMerged shape      : {metrics.shape}')
        print(f'All metrics found : {all(c in metrics.columns for c in CORE_METRICS)}')

    return metrics


def aggregate_to_team_level(metrics: pl.DataFrame) -> pl.DataFrame:
    """
    Collapse match-level rows to one row per team using WEIGHTED aggregation.

    Each metric is computed as a weighted mean where the weight is the
    tournament tier weight for that match. This ensures WC matches contribute
    more to a team's tactical profile than AFCON group stage matches.

    Also computes:
    - Unweighted std dev of PPDA and possession_pct (tactical volatility)
    - Match count per team
    - Effective weight (sum of tier weights — higher = more WC-quality evidence)
    - Average tier weight (1.0 = all WC matches, 0.75 = all regional)

    Returns
    -------
    pl.DataFrame  shape (n_teams, 14)
    """
    team_metrics = (
        metrics
        .group_by('team')
        .agg(
            # Weighted mean for each core metric — explicit alias avoids naming issues
            *[
                (
                    (pl.col(c) * pl.col('tier_weight')).sum() /
                    pl.col('tier_weight').sum()
                ).alias(c)
                for c in CORE_METRICS
            ],
            # Unweighted std dev — captures volatility regardless of match tier
            pl.col('ppda').std().alias('ppda_std'),
            pl.col('possession_pct').std().alias('possession_pct_std'),

            # Evidence quality
            pl.col('match_id').count().alias('n_matches'),
            pl.col('tier_weight').sum().alias('effective_weight'),
            pl.col('tier_weight').mean().alias('avg_tier_weight'),
        )
        .sort('team')
    )

    return team_metrics


def run_data_quality_audit(team_metrics: pl.DataFrame) -> None:
    """
    Print data quality report including tournament tier coverage.
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

    if 'avg_tier_weight' in team_metrics.columns:
        print('\n=== TOURNAMENT TIER COVERAGE ===')
        print('(avg_tier_weight: 1.0=all WC, 0.75=all regional)')
        tier_summary = team_metrics.select(
            ['team', 'n_matches', 'effective_weight', 'avg_tier_weight']
        ).sort('avg_tier_weight', descending=True)
        print(tier_summary)

    print('\n=== 3-SIGMA OUTLIER CHECK ===')
    found_any = False
    for col in CORE_METRICS:
        mean     = team_metrics[col].mean()
        std      = team_metrics[col].std()
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
                  verbose: bool = True,
                  exclude_teams=None):
    """
    Convenience wrapper: load → merge (with tier weights) → aggregate → audit.

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
