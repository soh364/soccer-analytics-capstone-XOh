"""
tc_pipeline.py
──────────────
Single entry point for the tactical clustering pipeline.

Usage (from anywhere in the project):
    from tactical_clustering.tc_pipeline import get_team_archetypes
    archetype_df = get_team_archetypes()

── Final Score Model ────────────────────────────────────────────────────────

The final_score column is a 0–100 score representing a team's tactical
readiness contribution for Section V. It is computed in three layers:

LAYER 1 — Base Archetype Score (0–100)
    The starting point. Derived by rescaling WC 2022 average outcome ranks
    to a 0–100 range, capped at 95 to leave headroom. Higher = the style
    this team plays historically produces better WC results.

    High Press / High Output : 95   (avg WC 2022 outcome rank 3.11 — highest)
    Possession Dominant      : 82
    Compact Transition       : 70
    Mid-Block Reactive       : 63
    Moderate Possession      : 48
    Low Intensity            : 35   (avg WC 2022 outcome rank 1.14 — lowest)

LAYER 2 — Blend Score (boundary team adjustment)
    For teams where the model is uncertain about their archetype, we blend
    the base scores of their top two candidate archetypes weighted by GMM
    probability. This avoids penalising boundary teams — instead it asks
    which two archetypes they sit between and takes a weighted average.

    blend_score = (gmm_confidence × base_score)
                + ((1 - gmm_confidence) × second_archetype_score)

    Exception: if gmm_confidence < 0.10, GMM is numerically unstable for
    this team. blend_score = archetype_score directly (no blending).

    Example: Spain is 23% Possession Dominant (82) and 77% High Press /
    High Output (95) → blend score ≈ 92. Argentina is 100% High Press /
    High Output → blend score = 95. Tunisia has confidence 0.002 →
    blend_score = 48 (archetype_score, no blending applied).

LAYER 3 — Evidence Discount
    Scales down teams whose style profile is based on limited evidence.
    Both factors are derived purely from internal data — no external sources.

    sample_weight      = 0.60 + 0.40 × (min(n_matches, 9) / 9)
                         (0.60 for 3 matches → 1.00 for 9+ matches)
                         Floor at 0.60 prevents over-penalising thin samples.

    wc_presence_weight = 0.80 + 0.20 × (outcome_rank / 7)  for WC 2022 teams
                       = 0.75                               for non-WC 2022 teams

                         Outcome rank scale: 1=Group Stage → 7=Winner
                         Winner        (rank 7) → 1.00
                         Runner-up     (rank 6) → 0.97
                         Third         (rank 5) → 0.94
                         Fourth        (rank 4) → 0.91
                         Quarter-final (rank 3) → 0.89
                         Round of 16   (rank 2) → 0.86
                         Group stage   (rank 1) → 0.83
                         Not in WC 2022         → 0.75

                         This ensures Argentina (Winner) scores above Canada
                         (Group stage exit) despite both playing the same style.

LAYER 4 — Intra-Cluster Quality Adjustment
    Within each archetype, teams are ranked by attacking output quality using
    npxG (0.7 weight) and inverse EPR (0.3 weight). This separates teams that
    share the same style but differ in how effectively they execute it.

    quality_metric   = npxG × 0.7 + (1/EPR) × 30
    intra_rank       = percentile rank within archetype (0–1)
    quality_adjustment = (intra_rank - 0.5) × 8 × sample_weight
                         Range: -4 to +4 (dampened by sample size)

    In plain English: Germany plays the same style as Namibia but creates
    far better chances. The adjustment rewards Germany (+5) and discounts
    Namibia (-4) within the same archetype cluster.

FINAL FORMULA:
    base = clip(blend_score × sample_weight × wc_presence_weight, 0, 100)
    final_score = clip(base + quality_adjustment, 0, 100)

In plain English: Start with the score your style earns historically, adjust
for whether you genuinely belong to that style or sit between two, scale down
if we do not have enough evidence to trust the profile fully, then fine-tune
based on how well you execute that style compared to your archetype peers.

── Output columns ────────────────────────────────────────────────────────────
    team, archetype, archetype_score,
    second_archetype, second_archetype_score,
    blend_score, gmm_confidence,
    sample_weight, wc_presence_weight, final_score,
    n_matches,
    ppda, possession_pct, defensive_line_height, field_tilt_pct,
    npxg, epr, progressive_carry_pct, avg_xg_per_buildup_possession
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

# ── Path resolution ───────────────────────────────────────────────────────────
_THIS_DIR     = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent
_EDA_ROOT     = _PROJECT_ROOT / 'eda'

for _p in [str(_THIS_DIR), str(_EDA_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tc_data           import load_pipeline, CLUSTER_FEATURES
from tc_preprocessing  import cap_and_scale
from tc_clustering     import ARCHETYPE_MAP, N_CLUSTERS
from tc_outcome_validation import WC2022_OUTCOMES, OUTCOME_RANK


# ── Constants ─────────────────────────────────────────────────────────────────

# Base archetype scores (0–100), derived from WC 2022 avg outcome ranks
# rescaled to [35, 95]. Formula per archetype:
#   score = (avg_rank - min_rank) / (max_rank - min_rank) * (95 - 35) + 35
# Raw avg ranks: HP/HO=3.11, CT=3.00, PD=2.00, MBR=1.67, LI=1.33, MP=1.14
ARCHETYPE_SCORE = {
    'High Press / High Output' : 95,
    'Possession Dominant'      : 82,
    'Compact Transition'       : 70,
    'Mid-Block Reactive'       : 63,
    'Moderate Possession'      : 48,
    'Low Intensity'            : 35,
}

EXCLUDE_TEAMS = ['Georgia', 'Slovenia']

# WC 2022 outcome-based presence weights
# Maps each WC 2022 team to their wc_presence_weight using outcome rank
# Formula: 0.80 + 0.20 × (outcome_rank / 7)
# Non-WC 2022 teams get 0.75
WC2022_PRESENCE_WEIGHT = {
    team: round(0.80 + 0.20 * (OUTCOME_RANK[outcome] / 7), 4)
    for team, outcome in WC2022_OUTCOMES.items()
}

ROSTER_COUNTRIES = [
    'Spain', 'Argentina', 'England', 'France', 'Germany', 'Brazil',
    'Netherlands', 'Portugal', 'United States', 'Mexico', 'Uruguay',
    'Croatia', 'Belgium', 'Japan', 'Switzerland', 'Ecuador', 'Turkey',
    'Senegal', 'Canada', 'Nigeria', 'Serbia', 'Austria', 'Poland',
    'Ghana', 'Colombia', 'Italy', 'Morocco', 'Australia', 'DR Congo',
    "Côte d'Ivoire", 'Cape Verde', 'Egypt', 'South Africa', 'Mali',
    'Cameroon', 'Czech Republic', 'Denmark', 'Ukraine', 'South Korea',
    'Iran', 'Saudi Arabia', 'Hungary', 'Qatar', 'Algeria', 'Tunisia',
    'Haiti', 'Norway', 'New Zealand', 'Uzbekistan', 'Jordan',
    'Curaçao', 'Panama', 'Scotland', 'Paraguay',
]

TEAM_NAME_MAP = {
    'Congo DR'          : 'DR Congo',
    'Cape Verde Islands': 'Cape Verde',
}

OUTPUT_COLUMNS = [
    'team', 'archetype', 'archetype_score',
    'second_archetype', 'second_archetype_score',
    'blend_score', 'gmm_confidence',
    'sample_weight', 'wc_presence_weight', 'quality_adjustment', 'final_score',
    'n_matches',
    'ppda', 'possession_pct', 'defensive_line_height', 'field_tilt_pct',
    'npxg', 'epr', 'progressive_carry_pct', 'avg_xg_per_buildup_possession',
]


# ── Internal helpers ──────────────────────────────────────────────────────────

def _compute_gmm_probabilities(X: np.ndarray,
                                kmeans_labels: np.ndarray,
                                n_components: int = N_CLUSTERS,
                                random_state: int = 42) -> np.ndarray:
    """
    Fit GMM and return full aligned probability matrix (n_teams, n_clusters).
    Columns are aligned to KMeans cluster integers via Hungarian algorithm.
    """
    gmm = GaussianMixture(
        n_components=n_components, covariance_type='full',
        random_state=random_state, n_init=20,
    )
    gmm.fit(X)

    proba      = gmm.predict_proba(X)
    gmm_labels = gmm.predict(X)

    cm               = confusion_matrix(kmeans_labels, gmm_labels,
                                        labels=list(range(n_components)))
    row_ind, col_ind = linear_sum_assignment(-cm)
    gmm_to_kmeans    = {g: k for k, g in zip(row_ind, col_ind)}

    aligned = np.zeros_like(proba)
    for gmm_idx, km_idx in gmm_to_kmeans.items():
        aligned[:, km_idx] = proba[:, gmm_idx]

    return np.round(aligned, 4)


def _compute_final_scores(results: pd.DataFrame,
                           gmm_proba: np.ndarray) -> pd.DataFrame:
    """
    Add blend_score, second_archetype, second_archetype_score,
    sample_weight, wc_presence_weight, and final_score to results.
    """
    df = results.copy().reset_index(drop=True)

    top_cluster = df['cluster'].values

    # Top confidence = probability of assigned cluster
    top_conf = gmm_proba[np.arange(len(df)), top_cluster]

    # Second best cluster = highest probability excluding top
    proba_copy = gmm_proba.copy()
    proba_copy[np.arange(len(df)), top_cluster] = -1
    second_cluster = np.argmax(proba_copy, axis=1)
    second_conf    = gmm_proba[np.arange(len(df)), second_cluster]

    df['gmm_confidence']         = np.round(top_conf, 4)
    df['second_archetype']       = [ARCHETYPE_MAP.get(int(c), 'Unknown')
                                     for c in second_cluster]
    df['second_archetype_score'] = df['second_archetype'].map(ARCHETYPE_SCORE)

    # Blend score — weighted average of top and second archetype scores
    # If gmm_confidence < 0.10, GMM is numerically unstable for this team
    # (not a genuine boundary case). Use archetype_score directly to avoid
    # inflating scores via a spurious second archetype assignment.
    prob_sum = top_conf + second_conf
    prob_sum = np.where(prob_sum > 0, prob_sum, 1.0)  # avoid div by zero

    raw_blend = (
        (top_conf * df['archetype_score'] +
         second_conf * df['second_archetype_score']) / prob_sum
    ).round(1)

    df['blend_score'] = np.where(
        top_conf >= 0.10,
        raw_blend,
        df['archetype_score'].astype(float)
    ).round(1)

    # Sample weight — from n_matches, floor at 0.60 so even 3-match teams
    # retain meaningful weight. Formula: 0.60 + 0.40 × (min(n,9) / 9)
    # Range: 0.60 (3 matches) → 1.00 (9+ matches)
    df['sample_weight'] = (0.60 + 0.40 * (df['n_matches'].clip(upper=9) / 9)).round(4)

    # WC presence weight — outcome-based for WC 2022 teams, 0.75 for others
    # Formula: 0.80 + 0.20 × (outcome_rank / 7) for WC 2022 participants
    # Winner → 1.00, Group stage → 0.83, Not in WC 2022 → 0.75
    df['wc_presence_weight'] = df['team'].apply(
        lambda t: WC2022_PRESENCE_WEIGHT.get(t, 0.75)
    )

    # Base score before quality adjustment — clipped to [0, 100]
    base_score = (
        df['blend_score'] *
        df['sample_weight'] *
        df['wc_presence_weight']
    ).clip(0, 100)

    # ── Layer 4: Intra-cluster quality adjustment ─────────────────────────
    # Rank teams within each archetype by attacking output quality.
    # Uses npxG (0.7) and inverse EPR (0.3) — the two highest F-stat metrics
    # from ANOVA that vary meaningfully within clusters.
    # Adjustment is dampened by sample_weight to prevent thin-sample inflation.
    epr_clipped = df['epr'].clip(upper=261.47)
    df['quality_metric'] = (
        df['npxg'] * 0.7 +
        (1 / epr_clipped) * 30
    )
    df['intra_rank'] = df.groupby('archetype')['quality_metric'].rank(pct=True)
    df['quality_adjustment'] = (
        (df['intra_rank'] - 0.5) * 8 * df['sample_weight']
    ).round(1)

    # Final score — base + quality adjustment, clipped to [0, 100]
    df['final_score'] = (base_score + df['quality_adjustment']).clip(0, 100).round(1)

    # Clean up helper columns not needed in output
    df.drop(columns=['quality_metric', 'intra_rank'], inplace=True)

    return df


def _reconcile_team_names(df: pd.DataFrame) -> tuple[pd.DataFrame, list, list]:
    """Apply TEAM_NAME_MAP and check against ROSTER_COUNTRIES."""
    df = df.copy()
    df['team'] = df['team'].replace(TEAM_NAME_MAP)
    matched   = [t for t in df['team'] if t in ROSTER_COUNTRIES]
    unmatched = [t for t in df['team'] if t not in ROSTER_COUNTRIES]
    return df, matched, unmatched


# ── Main pipeline function ────────────────────────────────────────────────────

def get_team_archetypes(tournament_key: str = 'men_tourn_2022_24',
                        verbose: bool = True) -> pd.DataFrame:
    """
    Run the full tactical clustering pipeline and return a clean DataFrame.

    Parameters
    ----------
    tournament_key : str   passed to load_tournament_data_8d()
    verbose        : bool  print summary when True

    Returns
    -------
    pd.DataFrame with OUTPUT_COLUMNS
    """
    # 1. Load & aggregate
    _, team_metrics = load_pipeline(tournament_key=tournament_key,
                                    verbose=False)

    # 2. Exclude outliers
    team_metrics_filtered = team_metrics.filter(
        ~pl.col('team').is_in(EXCLUDE_TEAMS)
    )

    # 3. Cap + scale
    X, teams, scaler, cap_info = cap_and_scale(
        team_metrics_filtered, ppda_pct=0.95, epr_pct=0.95
    )

    # 4. Fit KMeans
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=20)
    labels = kmeans.fit_predict(X)

    # 5. Full GMM probability matrix
    gmm_proba = _compute_gmm_probabilities(X, labels)

    # 6. Build base results DataFrame from capped metrics
    capped_pd = team_metrics_filtered.with_columns([
        pl.col('ppda').clip(upper_bound=cap_info['ppda_cap']).alias('ppda'),
        pl.col('epr').clip(upper_bound=cap_info['epr_cap']).alias('epr'),
    ]).to_pandas()

    results = capped_pd.copy()
    results['cluster']         = labels
    results['archetype']       = results['cluster'].map(ARCHETYPE_MAP)
    results['archetype_score'] = results['archetype'].map(ARCHETYPE_SCORE)

    # 7. Compute blend score and final score
    results = _compute_final_scores(results, gmm_proba)

    # 8. Team name reconciliation
    results, matched, unmatched = _reconcile_team_names(results)

    # 9. Select and order output columns
    available = [c for c in OUTPUT_COLUMNS if c in results.columns]
    output    = results[available].copy()

    metric_cols = [
        'ppda', 'possession_pct', 'defensive_line_height', 'field_tilt_pct',
        'npxg', 'epr', 'progressive_carry_pct', 'avg_xg_per_buildup_possession',
    ]
    for col in metric_cols:
        if col in output.columns:
            output[col] = output[col].round(4)

    output = output.sort_values('team').reset_index(drop=True)

    # Summary
    if verbose:
        print('\n' + '=' * 60)
        print('TACTICAL CLUSTERING PIPELINE — SUMMARY')
        print('=' * 60)
        print(f'Teams clustered : {len(output)}')
        print(f'Excluded        : {EXCLUDE_TEAMS}')

        print(f'\nArchetype distribution:')
        for arch, count in output['archetype'].value_counts().items():
            score = ARCHETYPE_SCORE.get(arch, '?')
            print(f'  [{score:>2}] {arch:<30} {count} teams')

        print(f'\nFinal score summary (0–100):')
        print(f'  Mean : {output["final_score"].mean():.1f}')
        print(f'  Max  : {output["final_score"].max():.1f}  '
              f'({output.loc[output["final_score"].idxmax(), "team"]})')
        print(f'  Min  : {output["final_score"].min():.1f}  '
              f'({output.loc[output["final_score"].idxmin(), "team"]})')

        print(f'\nTop 10 by final score:')
        top10 = output.nlargest(10, 'final_score')[
            ['team', 'archetype', 'blend_score',
             'sample_weight', 'wc_presence_weight', 'final_score']
        ]
        print(top10.to_string(index=False))

        print(f'\nTeam name reconciliation:')
        print(f'  Matched to roster : {len(matched)}/{len(output)}')
        if unmatched:
            print(f'  ⚠ Not in roster  : {unmatched}')
        else:
            print(f'  ✓ All names matched')

        low_conf = output[output['gmm_confidence'] < 0.5]
        if len(low_conf):
            print(f'\nBoundary teams (gmm_confidence < 0.5):')
            print(f'  {low_conf["team"].tolist()}')
        print('=' * 60)

    return output


# ── CSV export ────────────────────────────────────────────────────────────────

def export_archetypes_csv(output_path: str = None,
                          tournament_key: str = 'men_tourn_2022_24') -> pd.DataFrame:
    """Run get_team_archetypes() and save to CSV."""
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
