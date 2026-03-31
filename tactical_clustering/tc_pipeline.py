"""
tc_pipeline.py
──────────────
Single entry point for the tactical clustering pipeline.

Usage (from anywhere in the project):
    from tactical_clustering.tc_pipeline import get_team_archetypes
    archetype_df = get_team_archetypes()

Returns a clean DataFrame ready for the Section V composite scorer with
columns:
    team, archetype, archetype_score, gmm_confidence, n_matches,
    ppda, possession_pct, defensive_line_height, field_tilt_pct,
    npxg, epr, progressive_carry_pct, avg_xg_per_buildup_possession

Team name reconciliation:
    Names in the StatsBomb dataset are mapped to the canonical roster names
    used in rosters_2026.py. Mismatches are flagged in the summary output.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment

warnings.filterwarnings('ignore')

# ── Path resolution — works whether called from inside or outside the folder ──
_THIS_DIR    = Path(__file__).resolve().parent        # tactical_clustering/
_PROJECT_ROOT = _THIS_DIR.parent                       # project root
_EDA_ROOT    = _PROJECT_ROOT / 'eda'

for _p in [str(_THIS_DIR), str(_EDA_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tc_data         import load_pipeline, CORE_METRICS, CLUSTER_FEATURES
from tc_preprocessing import cap_and_scale
from tc_clustering   import (fit_kmeans_with_scaler, ARCHETYPE_MAP,
                              N_CLUSTERS)


# ── Constants ─────────────────────────────────────────────────────────────────

# Archetype → composite score for Section V
ARCHETYPE_SCORE = {
    'High Press / High Output' : 85,
    'Possession Dominant'      : 75,
    'Compact Transition'       : 65,
    'Mid-Block Reactive'       : 60,
    'Moderate Possession'      : 50,
    'Low Intensity'            : 40,
}

# Teams excluded from clustering (extreme PPDA outliers, not in WC 2026)
EXCLUDE_TEAMS = ['Georgia', 'Slovenia']

# Canonical roster country names from rosters_2026.py
ROSTER_COUNTRIES = [
    'Spain', 'Argentina', 'England', 'France', 'Germany', 'Brazil',
    'Netherlands', 'Portugal', 'United States', 'Mexico', 'Uruguay',
    'Croatia', 'Belgium', 'Japan', 'Switzerland', 'Ecuador', 'Turkey',
    'Senegal', 'Canada', 'Nigeria', 'Serbia', 'Austria', 'Poland',
    'Ghana', 'Colombia', 'Italy', 'Morocco', 'Australia', 'DR Congo',
    'Côte d\'Ivoire', 'Cape Verde', 'Egypt', 'South Africa', 'Mali',
    'Cameroon', 'Czech Republic', 'Denmark', 'Ukraine', 'South Korea',
    'Iran', 'Saudi Arabia', 'Hungary', 'Qatar', 'Algeria', 'Tunisia',
    'Haiti', 'Norway', 'New Zealand', 'Uzbekistan', 'Jordan',
    'Curaçao', 'Panama', 'Scotland', 'Paraguay',
]

# StatsBomb → roster name mapping for known mismatches
TEAM_NAME_MAP = {
    'Congo DR'         : 'DR Congo',
    'Côte d\'Ivoire'   : 'Côte d\'Ivoire',   # should already match
    'Cape Verde Islands': 'Cape Verde',
    'United States'    : 'United States',     # already matches
    'South Korea'      : 'South Korea',       # already matches
}

# Output columns in exact order required by composite scorer
OUTPUT_COLUMNS = [
    'team', 'archetype', 'archetype_score', 'gmm_confidence', 'n_matches',
    'ppda', 'possession_pct', 'defensive_line_height', 'field_tilt_pct',
    'npxg', 'epr', 'progressive_carry_pct', 'avg_xg_per_buildup_possession',
]


# ── Internal helpers ──────────────────────────────────────────────────────────
def _compute_gmm_confidence(X: np.ndarray,
                             kmeans_labels: np.ndarray,
                             n_components: int = N_CLUSTERS,
                             random_state: int = 42) -> np.ndarray:
    """
    Fit a GMM and return the max posterior probability for each team
    as a confidence score (0–1). Aligns GMM components to KMeans clusters
    via the Hungarian algorithm so confidence is comparable to KMeans labels.
    """
    gmm = GaussianMixture(
        n_components=n_components, covariance_type='full',
        random_state=random_state, n_init=20,
    )
    gmm.fit(X)

    proba      = gmm.predict_proba(X)           # shape (n_teams, n_components)
    gmm_labels = gmm.predict(X)

    # Align GMM component indices to KMeans cluster indices
    cm               = confusion_matrix(kmeans_labels, gmm_labels,
                                        labels=list(range(n_components)))
    row_ind, col_ind = linear_sum_assignment(-cm)
    gmm_to_kmeans    = {g: k for k, g in zip(row_ind, col_ind)}

    # Reorder probability columns to match KMeans cluster order
    aligned_proba = np.zeros_like(proba)
    for gmm_idx, km_idx in gmm_to_kmeans.items():
        aligned_proba[:, km_idx] = proba[:, gmm_idx]

    # Confidence = probability of the assigned KMeans cluster
    confidence = aligned_proba[np.arange(len(kmeans_labels)), kmeans_labels]
    return np.round(confidence, 4)


def _reconcile_team_names(df: pd.DataFrame) -> tuple[pd.DataFrame, list, list]:
    """
    Apply TEAM_NAME_MAP to align StatsBomb names to roster canonical names.
    Returns (updated_df, matched_list, unmatched_list).
    """
    df = df.copy()
    df['team'] = df['team'].replace(TEAM_NAME_MAP)

    matched   = [t for t in df['team'] if t in ROSTER_COUNTRIES]
    unmatched = [t for t in df['team'] if t not in ROSTER_COUNTRIES]

    return df, matched, unmatched


# ── Main pipeline function ────────────────────────────────────────────────────
def get_team_archetypes(tournament_key: str = 'men_tourn_2022_24',
                        verbose: bool = True) -> pd.DataFrame:
    """
    Run the full tactical clustering pipeline and return a clean DataFrame
    ready for the Section V composite scorer.

    Steps:
        1. Load & aggregate tournament metrics
        2. Exclude Georgia & Slovenia (extreme outliers, not in WC 2026)
        3. Cap PPDA + EPR at 95th percentile
        4. Scale with StandardScaler
        5. Fit KMeans (k=6, n_init=20)
        6. Compute GMM confidence scores
        7. Reconcile team names against roster_2026.py canonical list
        8. Return clean DataFrame with exact output columns

    Parameters
    ----------
    tournament_key : str   passed to load_tournament_data_8d()
    verbose        : bool  print summary when True

    Returns
    -------
    pd.DataFrame with columns:
        team, archetype, archetype_score, gmm_confidence, n_matches,
        ppda, possession_pct, defensive_line_height, field_tilt_pct,
        npxg, epr, progressive_carry_pct, avg_xg_per_buildup_possession
    """
    # ── 1. Load & aggregate ───────────────────────────────────────────────────
    _, team_metrics = load_pipeline(tournament_key=tournament_key,
                                    verbose=False)

    # ── 2. Exclude outliers ───────────────────────────────────────────────────
    team_metrics_filtered = team_metrics.filter(
        ~pl.col('team').is_in(EXCLUDE_TEAMS)
    )

    # ── 3 & 4. Cap + scale ────────────────────────────────────────────────────
    # cap_info contains the exact thresholds used — reuse below so metric
    # values in the output DataFrame match the notebook session exactly.
    X, teams, scaler, cap_info = cap_and_scale(
        team_metrics_filtered, ppda_pct=0.95, epr_pct=0.95
    )

    # ── 5. Fit KMeans ─────────────────────────────────────────────────────────
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=20)
    labels = kmeans.fit_predict(X)

    # ── 6. GMM confidence ─────────────────────────────────────────────────────
    gmm_confidence = _compute_gmm_confidence(X, labels)

    # ── 7. Build results DataFrame ────────────────────────────────────────────
    # Use cap_info thresholds — not recomputed — so values are consistent.
    capped_pd = team_metrics_filtered.with_columns([
        pl.col('ppda').clip(upper_bound=cap_info['ppda_cap']).alias('ppda'),
        pl.col('epr').clip(upper_bound=cap_info['epr_cap']).alias('epr'),
    ]).to_pandas()

    results = capped_pd.copy()
    results['cluster']        = labels
    results['archetype']      = results['cluster'].map(ARCHETYPE_MAP)
    results['archetype_score'] = results['archetype'].map(ARCHETYPE_SCORE)
    results['gmm_confidence'] = gmm_confidence

    # ── 8. Team name reconciliation ───────────────────────────────────────────
    results, matched, unmatched = _reconcile_team_names(results)

    # ── 9. Select & order output columns ─────────────────────────────────────
    # Keep only columns that exist (n_matches from aggregation)
    available = [c for c in OUTPUT_COLUMNS if c in results.columns]
    output    = results[available].copy()

    # Round metric columns
    metric_cols = [c for c in OUTPUT_COLUMNS if c not in
                   ['team', 'archetype', 'archetype_score', 'gmm_confidence', 'n_matches']]
    for col in metric_cols:
        if col in output.columns:
            output[col] = output[col].round(4)

    output = output.sort_values('team').reset_index(drop=True)

    # ── Summary ───────────────────────────────────────────────────────────────
    if verbose:
        print('\n' + '=' * 55)
        print('TACTICAL CLUSTERING PIPELINE — SUMMARY')
        print('=' * 55)
        print(f'Teams clustered    : {len(output)}')
        print(f'Excluded           : {EXCLUDE_TEAMS}')
        print(f'\nArchetype distribution:')
        dist = output['archetype'].value_counts()
        for arch, count in dist.items():
            score = ARCHETYPE_SCORE.get(arch, '?')
            print(f'  [{score:>2}] {arch:<30} {count} teams')
        print(f'\nTeam name reconciliation:')
        print(f'  Matched to roster  : {len(matched)}/{len(output)}')
        if unmatched:
            print(f'  ⚠ Not in roster   : {unmatched}')
            print(f'    → These teams will not join with rosters_2026.py')
            print(f'    → Add to TEAM_NAME_MAP in tc_pipeline.py to fix')
        else:
            print(f'  ✓ All names matched')
        print(f'\nGMM confidence:')
        print(f'  Mean  : {output["gmm_confidence"].mean():.3f}')
        print(f'  Min   : {output["gmm_confidence"].min():.3f}')
        low_conf = output[output['gmm_confidence'] < 0.5]
        if len(low_conf):
            print(f'  ⚠ Low confidence (<0.5): {low_conf["team"].tolist()}')
        print('=' * 55)

    return output


# ── CSV export ────────────────────────────────────────────────────────────────
def export_archetypes_csv(output_path: str = None,
                          tournament_key: str = 'men_tourn_2022_24') -> pd.DataFrame:
    """
    Run get_team_archetypes() and save the result to CSV.

    Parameters
    ----------
    output_path    : str  path to save CSV; defaults to
                          tactical_clustering/outputs/team_archetypes.csv
    tournament_key : str

    Returns
    -------
    pd.DataFrame  same as get_team_archetypes()
    """
    df = get_team_archetypes(tournament_key=tournament_key, verbose=True)

    if output_path is None:
        out_dir     = _THIS_DIR / 'outputs'
        out_dir.mkdir(exist_ok=True)
        output_path = out_dir / 'team_archetypes.csv'

    df.to_csv(output_path, index=False)
    print(f'\nCSV saved → {output_path}')

    return df


# ── CLI entry point ───────────────────────────────────────────────────────────
if __name__ == '__main__':
    export_archetypes_csv()
