"""
tc_clustering.py
────────────────
K-Means fitting, archetype labelling, GMM cross-validation,
and sensitivity analysis for the tactical clustering pipeline.

Why k=6?
─────────
Initial runs with k=4 produced a single cluster of 16 teams spanning
Argentina, Spain and Germany at the elite end down to Belgium, Mexico
and Canada. That cluster is analytically indefensible for a 2026
readiness framework — it assigns the same tactical profile to the
world champions and a group-stage exit.

A multi-metric decision matrix (Silhouette, Calinski-Harabasz, Davies-
Bouldin, GMM ARI) showed k=2 statistically dominates, reflecting the
true finding that teams exist on a tactical continuum. Among analytically
meaningful k values, k=6 achieves the highest GMM ARI (0.455) — the
cluster structure is more reproducible across different modelling
assumptions than at k=4 or k=5.

Why 69 teams, not 71?
──────────────────────
Georgia (PPDA 31.53) and Slovenia (PPDA 23.08) sit more than 4 standard
deviations above the dataset median and form a 2-team artifact cluster
at k=6 regardless of cap threshold. Tightening the cap to absorb them
dragged Japan (PPDA 17.11) — a tactically meaningful outlier — into the
same group, distorting its archetype assignment. As neither nation
qualified for the 2026 World Cup, exclusion is additionally justified
by the framework's primary purpose of 2026 readiness assessment.

On archetype naming:
────────────────────
Archetypes are named after tactical STYLE, not quality. Two teams can
share an archetype while being separated by player quality, squad depth,
and managerial experience (e.g. Argentina and Namibia are both
'High Press / High Output' by style metrics alone). Quality
differentiation happens in Section IV (player scoring) and Section V
(readiness score). The archetype is one input into that score, not the
score itself.

ARCHETYPE_MAP and ARCHETYPE_COLORS are the single source of truth —
import from this module, never redefine elsewhere.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score,
    confusion_matrix,
    adjusted_rand_score,
)
from scipy.optimize import linear_sum_assignment


# ── Constants ─────────────────────────────────────────────────────────────────
# Cluster integers assigned by KMeans — inspected via centroids.
# Labels reflect STYLE only, not quality tier.
# DO NOT change labels without re-running centroid inspection.

N_CLUSTERS = 6   # single source of truth — update here if k changes

ARCHETYPE_MAP = {
    0: 'Possession Dominant',       # highest possession, organised press, lower output
    1: 'Low Intensity',             # low press, low possession, limited tactical identity
    2: 'High Press / High Output',  # aggressive press + efficient possession + high npxG
    3: 'Compact Transition',        # low possession, compact defence, high npxG via transition
    4: 'Mid-Block Reactive',        # moderate press, defensively organised, lower conversion
    5: 'Moderate Possession',       # moderate possession, high EPR — sterile possession
}

ARCHETYPE_COLORS = {
    'High Press / High Output' : '#1a6eb5',   # deep blue
    'Possession Dominant'      : '#5bafd6',   # light blue
    'Compact Transition'       : '#27ae60',   # green
    'Mid-Block Reactive'       : '#f39c12',   # amber
    'Moderate Possession'      : '#e67e22',   # orange
    'Low Intensity'            : '#e74c3c',   # red
}


# ── KMeans fitting ────────────────────────────────────────────────────────────
def fit_kmeans_with_scaler(X: np.ndarray,
                           teams: list,
                           team_metrics_capped,
                           scaler,
                           n_clusters: int = N_CLUSTERS,
                           random_state: int = 42,
                           n_init: int = 20) -> tuple:
    """
    Fit KMeans on the scaled feature matrix and return labelled results.

    Uses the already-fitted scaler from cap_and_scale() to correctly
    back-transform cluster centroids to original metric scale.

    Parameters
    ----------
    X                   : np.ndarray   scaled feature matrix (n_teams, n_features)
    teams               : list[str]    team names in same row order as X
    team_metrics_capped : pl.DataFrame capped (unscaled) team metrics
    scaler              : StandardScaler fitted on capped data
    n_clusters          : int          default = N_CLUSTERS
    random_state        : int
    n_init              : int          20 initialisations for stability

    Returns
    -------
    (kmeans, results, centroid_df) : KMeans, pd.DataFrame, pd.DataFrame
        results     — one row per team, includes cluster + archetype columns
        centroid_df — centroids back-transformed to original metric scale
    """
    from tc_data import CLUSTER_FEATURES

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=n_init)
    labels = kmeans.fit_predict(X)

    results = team_metrics_capped.to_pandas()
    results['cluster']   = labels
    results['archetype'] = results['cluster'].map(ARCHETYPE_MAP)

    centroid_df = pd.DataFrame(
        scaler.inverse_transform(kmeans.cluster_centers_),
        columns=CLUSTER_FEATURES,
    ).round(3)
    centroid_df.insert(0, 'cluster', range(n_clusters))
    centroid_df['archetype'] = centroid_df['cluster'].map(ARCHETYPE_MAP)

    print('Cluster distribution:')
    print(results['cluster'].value_counts().sort_index())
    print('\nArchetype sizes:')
    print(results['archetype'].value_counts())

    return kmeans, results, centroid_df


def print_centroid_summary(centroid_df: pd.DataFrame) -> None:
    """Print key centroid metrics for archetype interpretation."""
    cols = ['archetype', 'ppda', 'possession_pct',
            'defensive_line_height', 'field_tilt_pct', 'npxg', 'epr']
    print('\nCluster centroids (original scale):')
    print(centroid_df[cols].to_string(index=False))


def print_archetype_teams(results: pd.DataFrame) -> None:
    """Print team lists per archetype, sorted alphabetically."""
    for archetype in ARCHETYPE_MAP.values():
        teams_in = (results[results['archetype'] == archetype]['team']
                    .sort_values().tolist())
        print(f'\n{archetype} ({len(teams_in)} teams):')
        print('  ' + ', '.join(teams_in))


# ── GMM cross-validation ──────────────────────────────────────────────────────
def run_gmm_validation(X: np.ndarray,
                       results: pd.DataFrame,
                       n_components: int = N_CLUSTERS,
                       random_state: int = 42,
                       n_init: int = 20) -> tuple[float, float, pd.DataFrame]:
    """
    Fit a GMM with the same k and compare assignments to KMeans using ARI.

    ARI (Adjusted Rand Index) is used instead of raw agreement rate because:
    - It accounts for chance agreement (raw % inflates with larger k)
    - ARI=1.0 means perfect agreement, ARI=0.0 means chance-level
    - Comparable across different k values (unlike raw %)

    Uses Hungarian algorithm to align GMM cluster labels to KMeans labels
    before generating the human-readable disagreement table.

    Parameters
    ----------
    X            : np.ndarray
    results      : pd.DataFrame  must contain 'cluster' column from KMeans
    n_components : int           default = N_CLUSTERS

    Returns
    -------
    (ari, kmeans_sil, disagreements_df) : float, float, pd.DataFrame
    """
    gmm = GaussianMixture(
        n_components=n_components, random_state=random_state,
        n_init=n_init, covariance_type='full',
    )
    gmm_labels = gmm.fit_predict(X)

    # ARI — primary agreement metric
    ari = adjusted_rand_score(results['cluster'].values, gmm_labels)

    # Align labels via Hungarian algorithm for disagreement table
    cm               = confusion_matrix(results['cluster'], gmm_labels)
    row_ind, col_ind = linear_sum_assignment(-cm)
    gmm_to_kmeans    = {col: row for row, col in zip(row_ind, col_ind)}
    gmm_aligned      = pd.Series(gmm_labels).map(gmm_to_kmeans)

    raw_agreement = (results['cluster'].values == gmm_aligned.values).mean()

    # Silhouette comparison
    kmeans_sil = silhouette_score(X, results['cluster'].values)
    gmm_sil    = silhouette_score(X, gmm_labels)

    print(f'KMeans silhouette  : {kmeans_sil:.3f}')
    print(f'GMM    silhouette  : {gmm_sil:.3f}')
    print(f'GMM ARI            : {ari:.3f}  (0=chance, 1=perfect)')
    print(f'Raw agreement      : {raw_agreement:.1%}')

    # Disagreement table
    mask          = results['cluster'].values != gmm_aligned.values
    disagreements = results[mask][['team', 'cluster', 'ppda',
                                   'possession_pct', 'npxg']].copy()
    disagreements['kmeans_archetype'] = disagreements['cluster'].map(ARCHETYPE_MAP)
    disagreements['gmm_archetype']    = gmm_aligned[mask].map(ARCHETYPE_MAP).values

    print(f'\nDisagreements ({mask.sum()} teams):')
    print(disagreements[['team', 'kmeans_archetype', 'gmm_archetype',
                          'ppda', 'possession_pct', 'npxg']].to_string(index=False))

    return ari, kmeans_sil, disagreements


# ── Sensitivity analysis ──────────────────────────────────────────────────────
def run_sensitivity_check(X: np.ndarray,
                          results: pd.DataFrame,
                          anchor_teams: list = None,
                          k_values: list = None,
                          random_state: int = 42,
                          n_init: int = 20) -> pd.DataFrame:
    """
    Test whether anchor teams maintain consistent relative groupings
    across k-1, k, and k+1.

    Stable groupings confirm the cluster structure is real and not an
    artifact of the specific k chosen.

    Parameters
    ----------
    anchor_teams : list[str]  high-profile teams to track across k values
    k_values     : list[int]  defaults to [N_CLUSTERS-1, N_CLUSTERS, N_CLUSTERS+1]

    Returns
    -------
    anchor_df : pd.DataFrame  anchor teams × k columns
    """
    if anchor_teams is None:
        anchor_teams = [
            'Argentina', 'Spain', 'Germany', 'France',
            'Morocco', 'Japan', 'Brazil', 'England',
            'Netherlands', 'Portugal', 'Italy',
        ]
    if k_values is None:
        k_values = [N_CLUSTERS - 1, N_CLUSTERS, N_CLUSTERS + 1]

    sensitivity = pd.DataFrame({'team': results['team'].values})

    for k in k_values:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
        sensitivity[f'k{k}'] = km.fit_predict(X)

    anchor_df = (sensitivity[sensitivity['team'].isin(anchor_teams)]
                 .reset_index(drop=True))

    print(f'Anchor team stability across k={k_values}:')
    print(anchor_df.to_string(index=False))

    print('\nRelative groupings (teams always clustering together):')
    for k in k_values:
        col    = f'k{k}'
        groups = anchor_df.groupby(col)['team'].apply(list)
        for _, group in groups.items():
            if len(group) > 1:
                print(f'  k={k}: {", ".join(sorted(group))}')

    return anchor_df