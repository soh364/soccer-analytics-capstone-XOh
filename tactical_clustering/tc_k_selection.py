"""
tc_k_selection.py
─────────────────
K selection analysis for tactical clustering:
- Elbow method (inertia)
- Silhouette score sweep
- Plot generation
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# ── Analysis ──────────────────────────────────────────────────────────────────
def sweep_k(X: np.ndarray,
            k_range: range = range(2, 10),
            random_state: int = 42,
            n_init: int = 20) -> tuple[list, list]:
    """
    Compute inertia and silhouette score for each k in k_range.

    Returns
    -------
    (inertias, silhouette_scores) : list, list
    """
    inertias          = []
    silhouette_scores = []

    for k in k_range:
        km     = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
        labels = km.fit_predict(X)
        inertias.append(km.inertia_)
        silhouette_scores.append(silhouette_score(X, labels))
        print(f'k={k} | Inertia: {km.inertia_:.1f} | Silhouette: {silhouette_score(X, labels):.3f}')

    return inertias, silhouette_scores


# ── Visualisation ─────────────────────────────────────────────────────────────
def plot_k_selection(k_range: range,
                     inertias: list,
                     silhouette_scores: list,
                     selected_k: int = 4,
                     figures_dir: Path = Path('figures')) -> None:
    """
    Plot elbow curve and silhouette scores side by side.
    Saves to figures_dir/optimal_k.png.
    """
    figures_dir.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Elbow
    axes[0].plot(list(k_range), inertias, 'bo-', linewidth=2, markersize=8)
    axes[0].axvline(x=selected_k, color='red', linestyle='--',
                    alpha=0.7, label=f'k={selected_k} selected')
    axes[0].set_xlabel('Number of Clusters (k)')
    axes[0].set_ylabel('Inertia')
    axes[0].set_title('Elbow Method')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Silhouette
    axes[1].plot(list(k_range), silhouette_scores, 'go-', linewidth=2, markersize=8)
    axes[1].axvline(x=selected_k, color='red', linestyle='--',
                    alpha=0.7, label=f'k={selected_k} selected')
    axes[1].set_xlabel('Number of Clusters (k)')
    axes[1].set_ylabel('Silhouette Score')
    axes[1].set_title('Silhouette Score (higher = better)')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.suptitle('Finding Optimal k', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(figures_dir / 'optimal_k.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f'Saved → {figures_dir}/optimal_k.png')
