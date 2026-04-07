"""
tc_visualisation.py
───────────────────
All visualisation functions for the tactical clustering analysis:
- PCA scatter plot
- Archetype radar charts
- Outcome validation bar charts
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from tc_clustering import ARCHETYPE_MAP, ARCHETYPE_COLORS
from tc_data import CLUSTER_FEATURES


# ── PCA scatter ───────────────────────────────────────────────────────────────
def plot_pca_scatter(X: np.ndarray,
                     results: pd.DataFrame,
                     kmeans: KMeans,
                     highlight_teams: list = None,
                     figures_dir: Path = Path('figures'),
                     random_state: int = 42) -> pd.DataFrame:
    """
    Reduce X to 2D via PCA and plot teams coloured by archetype.
    Cluster centroids shown as stars. Key nations labelled.

    Returns results with pca1, pca2 columns added.
    """
    if highlight_teams is None:
        highlight_teams = [
            'Argentina', 'Spain', 'Germany', 'France',
            'Brazil', 'England', 'Morocco', 'Japan',
            'Netherlands', 'Portugal', 'United States',
        ]

    figures_dir.mkdir(exist_ok=True)

    pca   = PCA(n_components=2, random_state=random_state)
    X_pca = pca.fit_transform(X)

    results = results.copy()
    results['pca1'] = X_pca[:, 0]
    results['pca2'] = X_pca[:, 1]

    var1, var2 = pca.explained_variance_ratio_
    print(f'Variance explained — PC1: {var1:.1%}, PC2: {var2:.1%}, Total: {var1+var2:.1%}')

    fig, ax = plt.subplots(figsize=(14, 10))

    for archetype, color in ARCHETYPE_COLORS.items():
        mask = results['archetype'] == archetype
        ax.scatter(
            results.loc[mask, 'pca1'],
            results.loc[mask, 'pca2'],
            c=color, label=archetype,
            s=80, alpha=0.7, edgecolors='white', linewidth=0.5,
        )

    # Labels for key nations
    for _, row in results[results['team'].isin(highlight_teams)].iterrows():
        ax.annotate(
            row['team'], (row['pca1'], row['pca2']),
            fontsize=8, fontweight='bold',
            xytext=(6, 4), textcoords='offset points',
        )

    # Centroid stars
    centroids_pca = pca.transform(kmeans.cluster_centers_)
    for i, (cx, cy) in enumerate(centroids_pca):
        color = ARCHETYPE_COLORS[ARCHETYPE_MAP[i]]
        ax.scatter(cx, cy, c=color, s=300, marker='*',
                   edgecolors='black', linewidth=1.5, zorder=5)

    ax.set_xlabel(f'PC1 ({var1:.1%} variance — Proactive vs Reactive axis)', fontsize=11)
    ax.set_ylabel(f'PC2 ({var2:.1%} variance — Efficiency axis)', fontsize=11)
    ax.set_title('2026 World Cup Tactical Archetypes\nK-Means Clustering (k=6): PCA Projection',
                 fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig(figures_dir / 'tactical_clusters_pca.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f'Saved → {figures_dir}/tactical_clusters_pca.png')

    return results


# ── Radar charts ──────────────────────────────────────────────────────────────

def plot_archetype_radars(kmeans: KMeans,
                          scaler,
                          results: pd.DataFrame,
                          figures_dir: Path = Path('figures')) -> None:
    """
    2×2 grid of radar charts — one per archetype.
    PPDA is inverted so all axes read "higher = better".
    Axes use sqrt-transformed min-max normalisation: relative ordering
    between archetypes is preserved but small values are amplified so
    low-intensity profiles remain legible rather than collapsing to a dot.
    Saves to figures_dir/tactical_archetypes_radar.png.
    """
    figures_dir.mkdir(exist_ok=True)
 
    RADAR_METRICS = [
        'ppda', 'possession_pct', 'defensive_line_height',
        'field_tilt_pct', 'npxg', 'avg_xg_per_buildup_possession',
    ]
    RADAR_LABELS = [
        'Pressing\nIntensity*', 'Possession\n%', 'Line\nHeight',
        'Field\nTilt', 'npxG', 'xG\nBuildup',
    ]
 
    centroid_original = pd.DataFrame(
        scaler.inverse_transform(kmeans.cluster_centers_),
        columns=CLUSTER_FEATURES,
    )
 
    # Invert PPDA — lower raw value = more aggressive pressing
    radar_data         = centroid_original[RADAR_METRICS].copy()
    radar_data['ppda'] = radar_data['ppda'].max() - radar_data['ppda']

    # ── Normalise against full team-level range + apply floor ────────────
    # Root cause of the flat Low Intensity hexagon:
    #   Low Intensity sits at or near the minimum on EVERY axis after PPDA
    #   inversion. When we normalise using only the 6 centroid values,
    #   the range is tiny and LI maps to near-zero everywhere — producing
    #   a flat regular hexagon regardless of floor tricks.
    #
    # Fix: normalise each axis against the FULL team-level range (all 69
    #   teams, not just 6 centroids). This gives each axis a meaningful
    #   scale so the slight variation within Low Intensity's profile is
    #   visible (e.g. it's weakest on npxG, slightly less weak on possession).
    #   Then apply a FLOOR=0.15 so even the absolute minimum sits visibly
    #   off the centre rather than disappearing.
    #
    # Result: Low Intensity still looks clearly smaller than every other
    #   archetype — but you can read its actual shape, not just a dot.
    team_metrics = results[RADAR_METRICS].copy()
    team_metrics['ppda'] = team_metrics['ppda'].max() - team_metrics['ppda']

    global_min = team_metrics.min()
    global_max = team_metrics.max()

    radar_norm_linear = (radar_data - global_min) / (global_max - global_min)

    FLOOR      = 0.15
    radar_norm = FLOOR + (1.0 - FLOOR) * radar_norm_linear

    N      = len(RADAR_METRICS)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
 
    n_archetypes = len(ARCHETYPE_MAP)
    ncols = 3 if n_archetypes >= 5 else 2
    nrows = (n_archetypes + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7*ncols, 5*nrows), subplot_kw=dict(polar=True))
    plt.subplots_adjust(hspace=0.3, wspace=0.2)
    axes      = axes.flatten()
 
    for i, (cluster_id, archetype) in enumerate(ARCHETYPE_MAP.items()):
        ax     = axes[i]
        values = radar_norm.iloc[cluster_id].tolist() + [radar_norm.iloc[cluster_id].tolist()[0]]
        color  = ARCHETYPE_COLORS[archetype]
        n_teams = len(results[results['archetype'] == archetype])
 
        ax.plot(angles, values, color=color, linewidth=2)
        ax.fill(angles, values, color=color, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(RADAR_LABELS, fontsize=9)
        ax.set_ylim(0, 1)   # keep 0 as origin so FLOOR gap is visible
        ax.set_title(f'{archetype}\n({n_teams} teams)',
                     fontsize=10, fontweight='bold', pad=15)
        ax.grid(alpha=0.3)
        ax.set_yticklabels([])
 
    plt.suptitle('Tactical Archetype Profiles: Radar Charts\n(axes scaled against full team range, floor=0.15 for legibility)',
                 fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(figures_dir / 'tactical_archetypes_radar.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f'Saved → {figures_dir}/tactical_archetypes_radar.png')

# ── Outcome validation ────────────────────────────────────────────────────────
def plot_outcome_validation(wc_teams: pd.DataFrame,
                            avg_rank: pd.DataFrame,
                            figures_dir: Path = Path('figures')) -> None:
    """
    Two-panel plot:
    Left  — average outcome rank per archetype (horizontal bar)
    Right — stacked bar of outcome breakdown per archetype

    Saves to figures_dir/outcome_validation.png.
    """
    figures_dir.mkdir(exist_ok=True)

    outcome_order  = ['Winner', 'Runner-up', 'Third', 'Fourth',
                      'Quarter-final', 'Round of 16', 'Group Stage']
    outcome_colors = ['#FFD700', '#C0C0C0', '#CD7F32', '#4ECDC4',
                      '#45B7D1', '#96CEB4', '#DDD']

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left panel — average rank
    colors = [ARCHETYPE_COLORS[a] for a in avg_rank['archetype']]
    axes[0].barh(avg_rank['archetype'], avg_rank['avg_outcome_rank'],
                 color=colors, edgecolor='white')
    axes[0].axvline(x=wc_teams['outcome_rank'].mean(), color='black',
                    linestyle='--', alpha=0.5, label='Overall mean')
    axes[0].set_xlabel('Average Outcome Rank (higher = better)')
    axes[0].set_title('Average Tournament Outcome\nby Tactical Archetype')
    axes[0].legend()
    axes[0].grid(alpha=0.2, axis='x')

    # Right panel — stacked breakdown
    pivot = (
        wc_teams.groupby(['archetype', 'wc2022_outcome'])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=outcome_order, fill_value=0)
    )
    pivot.plot(kind='barh', stacked=True, ax=axes[1],
               color=outcome_colors, edgecolor='white')
    axes[1].set_xlabel('Number of Teams')
    axes[1].set_title('Tournament Outcome Distribution\nby Tactical Archetype')
    axes[1].legend(loc='lower right', fontsize=8)
    axes[1].grid(alpha=0.2, axis='x')

    plt.tight_layout()
    plt.savefig(figures_dir / 'outcome_validation.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f'Saved → {figures_dir}/outcome_validation.png')
