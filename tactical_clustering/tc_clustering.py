"""
tc_clustering.py
────────────────
K-Means fitting, archetype labelling, GMM cross-validation,
and sensitivity analysis for the tactical clustering pipeline.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, confusion_matrix
from scipy.optimize import linear_sum_assignment


# ── Constants ─────────────────────────────────────────────────────────────────
# Archetype labels keyed by KMeans cluster integer.
# NOTE: These are assigned AFTER inspecting centroids — do not change without re-running the centroid inspection step.
ARCHETYPE_MAP = {
    0: 'Reactive Disruptors',
    1: 'Proactive Dominant',
    2: 'Possession Adaptive',
    3: 'Passive Struggling',
}

ARCHETYPE_COLORS = {
    'Proactive Dominant'  : '#3498db',  # blue
    'Possession Adaptive' : '#e67e22',  # orange
    'Reactive Disruptors' : '#2ecc71',  # green
    'Passive Struggling'  : '#e74c3c',  # red
}


# ── KMeans fitting ────────────────────────────────────────────────────────────
def fit_kmeans(X: np.ndarray,
               teams: list,
               team_metrics_capped,           # pl.DataFrame
               n_clusters: int = 4,
               random_state: int = 42,
               n_init: int = 20) -> tuple:
    """
    Fit KMeans on the scaled feature matrix and return labelled results.

    Parameters
    ----------
    X                  : np.ndarray  scaled feature matrix
    teams              : list[str]
    team_metrics_capped: pl.DataFrame  capped (unscaled) team metrics
    n_clusters         : int
    random_state       : int
    n_init             : int  number of initialisations (20 for stability)

    Returns
    -------
    (kmeans, results, centroid_df) : KMeans, pd.DataFrame, pd.DataFrame
        results     — one row per team with cluster + archetype columns
        centroid_df — centroids in original (unscaled) space
    """
    from sklearn.preprocessing import StandardScaler
    from tc_data import CLUSTER_FEATURES

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=n_init)
    labels = kmeans.fit_predict(X)

    results = team_metrics_capped.to_pandas()
    results['cluster']   = labels
    results['archetype'] = results['cluster'].map(ARCHETYPE_MAP)

    # Refit scaler on capped data to back-transform centroids
    from sklearn.preprocessing import StandardScaler
    scaler_tmp = StandardScaler()
    scaler_tmp.fit(team_metrics_capped.select(CLUSTER_FEATURES).to_numpy())

    centroid_df = pd.DataFrame(
        scaler_tmp.inverse_transform(kmeans.cluster_centers_),
        columns=CLUSTER_FEATURES
    ).round(3)
    centroid_df.insert(0, 'cluster', range(n_clusters))
    centroid_df['archetype'] = centroid_df['cluster'].map(ARCHETYPE_MAP)

    # Print distribution
    print('Cluster distribution:')
    print(results['cluster'].value_counts().sort_index())
    print('\nArchetype assignments:')
    for arch in ARCHETYPE_MAP.values():
        t = results[results['archetype'] == arch]['team'].sort_values().tolist()
        print(f'\n  {arch} ({len(t)} teams):')
        print('  ' + ', '.join(t))

    return kmeans, results, centroid_df


def fit_kmeans_with_scaler(X: np.ndarray,
                           teams: list,
                           team_metrics_capped,
                           scaler,
                           n_clusters: int = 4,
                           random_state: int = 42,
                           n_init: int = 20) -> tuple:
    """
    Same as fit_kmeans but uses the already-fitted scaler from
    cap_and_scale() to correctly back-transform centroids.

    Preferred entry point when called from the notebook.
    """
    from tc_data import CLUSTER_FEATURES

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=n_init)
    labels = kmeans.fit_predict(X)

    results = team_metrics_capped.to_pandas()
    results['cluster']   = labels
    results['archetype'] = results['cluster'].map(ARCHETYPE_MAP)

    centroid_df = pd.DataFrame(
        scaler.inverse_transform(kmeans.cluster_centers_),
        columns=CLUSTER_FEATURES
    ).round(3)
    centroid_df.insert(0, 'cluster', range(n_clusters))
    centroid_df['archetype'] = centroid_df['cluster'].map(ARCHETYPE_MAP)

    print('Cluster distribution:')
    print(results['cluster'].value_counts().sort_index())

    return kmeans, results, centroid_df


def print_centroid_summary(centroid_df: pd.DataFrame) -> None:
    """Print key centroid metrics for archetype interpretation."""
    cols = ['archetype', 'ppda', 'possession_pct',
            'defensive_line_height', 'field_tilt_pct', 'npxg', 'epr']
    print('\nCluster centroids (original scale):')
    print(centroid_df[cols].to_string(index=False))


# ── GMM cross-validation ──────────────────────────────────────────────────────
def run_gmm_validation(X: np.ndarray,
                       results: pd.DataFrame,
                       n_components: int = 4,
                       random_state: int = 42,
                       n_init: int = 20) -> tuple[float, pd.DataFrame]:
    """
    Fit a GMM with the same k and compare assignments to KMeans.

    Uses the Hungarian algorithm to align GMM cluster labels to KMeans
    labels before computing agreement (GMM assigns numbers arbitrarily).

    Returns
    -------
    (agreement_rate, disagreements_df) : float, pd.DataFrame
    """
    gmm        = GaussianMixture(n_components=n_components, random_state=random_state,
                                 n_init=n_init, covariance_type='full')
    gmm_labels = gmm.fit_predict(X)

    # Align GMM labels to KMeans labels via Hungarian algorithm
    cm                   = confusion_matrix(results['cluster'], gmm_labels)
    row_ind, col_ind     = linear_sum_assignment(-cm)
    gmm_to_kmeans        = {col: row for row, col in zip(row_ind, col_ind)}
    gmm_aligned          = pd.Series(gmm_labels).map(gmm_to_kmeans)

    agreement = (results['cluster'].values == gmm_aligned.values).mean()

    # Silhouette comparison
    kmeans_sil = silhouette_score(X, results['cluster'].values)
    gmm_sil    = silhouette_score(X, gmm_labels)

    print(f'KMeans silhouette : {kmeans_sil:.3f}')
    print(f'GMM    silhouette : {gmm_sil:.3f}')
    print(f'Agreement rate    : {agreement:.1%}')

    # Disagreement table
    mask          = results['cluster'].values != gmm_aligned.values
    disagreements = results[mask][['team', 'cluster', 'ppda',
                                   'possession_pct', 'npxg']].copy()
    disagreements['kmeans_archetype'] = disagreements['cluster'].map(ARCHETYPE_MAP)
    disagreements['gmm_archetype']    = gmm_aligned[mask].map(ARCHETYPE_MAP).values

    print(f'\nDisagreements ({len(disagreements)} teams):')
    print(disagreements[['team', 'kmeans_archetype', 'gmm_archetype',
                          'ppda', 'possession_pct', 'npxg']].to_string(index=False))

    return agreement, disagreements


# ── Sensitivity analysis ──────────────────────────────────────────────────────
def run_sensitivity_check(X: np.ndarray,
                          results: pd.DataFrame,
                          anchor_teams: list = None,
                          k_values: list = None,
                          random_state: int = 42,
                          n_init: int = 20) -> pd.DataFrame:
    """
    Test whether anchor teams maintain consistent relative groupings
    across different values of k.

    Parameters
    ----------
    anchor_teams : list[str]  high-profile teams to track
    k_values     : list[int]  e.g. [3, 4, 5]

    Returns
    -------
    sensitivity_df : pd.DataFrame  anchor teams × k columns
    """
    if anchor_teams is None:
        anchor_teams = ['Argentina', 'Spain', 'Germany', 'France',
                        'Morocco', 'Japan', 'Brazil', 'England']
    if k_values is None:
        k_values = [3, 4, 5]

    sensitivity = pd.DataFrame({'team': results['team']})

    for k in k_values:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
        sensitivity[f'k{k}'] = km.fit_predict(X)

    anchor_df = sensitivity[sensitivity['team'].isin(anchor_teams)].reset_index(drop=True)
    print('Anchor team stability across k values:')
    print(anchor_df.to_string(index=False))

    return anchor_df
