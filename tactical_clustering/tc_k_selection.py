"""
tc_k_selection.py
─────────────────
K selection analysis for tactical clustering using a multi-metric
decision matrix. Evaluates k values across four independent diagnostics
before selecting the optimal cluster count.

Metrics:
    Silhouette Score       — cluster cohesion (higher = better)
    Calinski-Harabasz (CH) — between/within variance ratio (higher = better)
    Davies-Bouldin (DB)    — average cluster similarity (lower = better)
    GMM ARI                — cross-model stability via Adjusted Rand Index (higher = better)
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    adjusted_rand_score,
)


# ── Multi-metric sweep ────────────────────────────────────────────────────────
def sweep_k(X: np.ndarray,
            k_range: range = range(2, 10),
            random_state: int = 42,
            n_init: int = 20) -> pd.DataFrame:
    """
    Evaluate cluster quality across k values using four independent metrics.

    For each k: fits KMeans (n_init=20), fits GMM, computes ARI
    between the two models, and records all four cluster quality metrics.

    Parameters
    ----------
    X            : np.ndarray  scaled feature matrix (n_teams, n_features)
    k_range      : range       k values to evaluate
    random_state : int
    n_init       : int         KMeans initialisations (20 for stability)

    Returns
    -------
    metrics_df : pd.DataFrame  one row per k, columns = all metrics
    """
    rows = []

    for k in k_range:
        km     = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
        km_lbl = km.fit_predict(X)

        gmm     = GaussianMixture(n_components=k, random_state=random_state,
                                  n_init=n_init, covariance_type='full')
        gmm_lbl = gmm.fit_predict(X)

        sil = silhouette_score(X, km_lbl)
        ch  = calinski_harabasz_score(X, km_lbl)
        db  = davies_bouldin_score(X, km_lbl)
        ari = adjusted_rand_score(km_lbl, gmm_lbl)

        rows.append({'k': k, 'silhouette': round(sil, 3),
                     'calinski_harabasz': round(ch, 1),
                     'davies_bouldin': round(db, 3),
                     'gmm_ari': round(ari, 3),
                     'inertia': round(km.inertia_, 1)})

        print(f'k={k} | Sil: {sil:.3f} | CH: {ch:.1f} | '
              f'DB: {db:.3f} | ARI: {ari:.3f} | Inertia: {km.inertia_:.1f}')

    return pd.DataFrame(rows)


def print_decision_matrix(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """
    Print a ranked decision matrix. Each metric is ranked independently;
    avg_rank aggregates them into a single composite score.

    Rank direction:  Silhouette ↑  |  CH ↑  |  DB ↓  |  ARI ↑
    """
    df = metrics_df.copy()
    df['rank_sil'] = df['silhouette'].rank(ascending=False).astype(int)
    df['rank_ch']  = df['calinski_harabasz'].rank(ascending=False).astype(int)
    df['rank_db']  = df['davies_bouldin'].rank(ascending=True).astype(int)
    df['rank_ari'] = df['gmm_ari'].rank(ascending=False).astype(int)
    df['avg_rank'] = ((df['rank_sil'] + df['rank_ch'] +
                       df['rank_db'] + df['rank_ari']) / 4).round(2)

    display_cols = ['k', 'silhouette', 'calinski_harabasz',
                    'davies_bouldin', 'gmm_ari', 'avg_rank']

    print('\n=== K SELECTION — MULTI-METRIC DECISION MATRIX ===\n')
    print(df[display_cols].to_string(index=False))
    best_k = int(df.loc[df['avg_rank'].idxmin(), 'k'])
    print(f'\n→ Best composite rank : k={best_k}')

    return df


# ── Visualisation ─────────────────────────────────────────────────────────────
def plot_k_selection(metrics_df: pd.DataFrame,
                     selected_k: int = 6,
                     figures_dir: Path = Path('figures')) -> None:
    """
    4-panel plot: one panel per metric with selected_k marked.
    Saves to figures_dir/optimal_k.png.
    """
    figures_dir.mkdir(exist_ok=True)
    k_vals = metrics_df['k'].tolist()

    panels = [
        ('silhouette',        'Silhouette Score',         'go-', 'Higher = better'),
        ('calinski_harabasz', 'Calinski-Harabasz Score',  'bo-', 'Higher = better'),
        ('davies_bouldin',    'Davies-Bouldin Index',     'ro-', 'Lower = better'),
        ('gmm_ari',           'GMM Agreement (ARI)',      'mo-', 'Higher = better'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for ax, (col, title, fmt, note) in zip(axes.flatten(), panels):
        ax.plot(k_vals, metrics_df[col].tolist(), fmt, linewidth=2, markersize=8)
        ax.axvline(x=selected_k, color='black', linestyle='--',
                   alpha=0.7, label=f'k={selected_k} selected')
        ax.set_xlabel('Number of Clusters (k)')
        ax.set_title(f'{title}\n({note})', fontsize=10, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    plt.suptitle('K Selection — Multi-Metric Decision Matrix',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(figures_dir / 'optimal_k.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f'Saved → {figures_dir}/optimal_k.png')