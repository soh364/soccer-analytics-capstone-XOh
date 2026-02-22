"""
Tactical visualization library for cluster analysis and player comparison.
Includes PCA maps, archetype radars, and pizza charts for player profiling.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from mplsoccer import PyPizza
import pandas as pd
import polars as pl
import seaborn as sns

def plot_clustering_validation_compact(optimization_results, figsize=(8, 3.5)):
    """Validation plots for k-means optimization (inertia and silhouette)."""
    navy = '#1a1a2e'
    teal = '#2d6a4f'
    grey = '#999999'

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, dpi=100)
    
    # Elbow curve
    ax1.plot(optimization_results['k'], optimization_results['inertia'], 
             color=navy, marker='o', linewidth=2, markersize=6, 
             markerfacecolor='white', markeredgewidth=2)
    ax1.set_xlabel('Number of Clusters (k)', fontsize=10, color=navy)
    ax1.set_ylabel('Inertia', fontsize=9, color=grey)
    ax1.set_title('Elbow Curve', fontsize=11, weight='bold', color=navy, pad=12)
    
    # Silhouette curve
    ax2.plot(optimization_results['k'], optimization_results['silhouette'], 
             color=teal, marker='o', linewidth=2, markersize=6,
             markerfacecolor='white', markeredgewidth=2)
    ax2.set_xlabel('Number of Clusters (k)', fontsize=10, color=navy)
    ax2.set_ylabel('Silhouette Score', fontsize=9, color=grey)
    ax2.set_title('Silhouette Analysis', fontsize=11, weight='bold', color=navy, pad=12)

    # Formatting
    for ax in [ax1, ax2]:
        ax.grid(True, linestyle='--', alpha=0.2, color=navy)
        ax.set_xticks(optimization_results['k'])
        ax.tick_params(axis='both', which='major', labelsize=9, colors=navy)
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        for spine in ['left', 'bottom']:
            ax.spines[spine].set_color('#e0e0e0')
    
    plt.tight_layout()
    return fig, (ax1, ax2)

def save_figure(fig, filename, dpi=300):
    """Save figure with consistent settings"""
    fig.savefig(f'../figures/{filename}', dpi=dpi, bbox_inches='tight')
    print(f"Saved: outputs/figures/{filename}")

def plot_tactical_pca(profiles_df, dimensions, labels, archetype_names,
                      highlight_teams=None, figsize=(8, 6)):
    """PCA map of tactical space with archetype clusters and team annotations."""
    colors = ['#4895C4', '#A23B72', '#F18F01', '#06A77D']
    
    # Fit PCA
    scaler = StandardScaler()
    scaled = scaler.fit_transform(profiles_df[dimensions])
    pca = PCA(n_components=2)
    coords = pca.fit_transform(scaled)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot clusters
    for cluster_id in range(len(archetype_names)):
        mask = labels == cluster_id
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=colors[cluster_id], alpha=0.6, s=100, edgecolors='white',
                   linewidth=1, label=archetype_names[cluster_id], zorder=2)
        
        # Centroid
        cx, cy = coords[mask, 0].mean(), coords[mask, 1].mean()
        ax.scatter(cx, cy, c=colors[cluster_id], s=200, edgecolors='black',
                   linewidth=1, marker='D', zorder=2, alpha=0.9)
    
    # Annotate teams
    if highlight_teams and 'team' in profiles_df.columns:
        for team in highlight_teams:
            idx = profiles_df.index[profiles_df['team'] == team]
            if len(idx) > 0:
                i = idx[0]
                x, y = coords[i, 0], coords[i, 1]
                ax.annotate(team, (x, y), fontsize=8, weight='bold',
                            xytext=(8, 8), textcoords='offset points',
                            arrowprops=dict(arrowstyle='->', color='#333', lw=1),
                            color='#1a1a2e', zorder=7,
                            bbox=dict(boxstyle='round,pad=0.3', 
                                     facecolor='yellow', alpha=0.7))
    
    # Axes
    loadings = pca.components_
    top_pc1_idx = np.argsort(np.abs(loadings[0]))[-2:]
    top_pc2_idx = np.argsort(np.abs(loadings[1]))[-2:]
    
    top_pc1 = [dimensions[i].replace('_', ' ').title() for i in top_pc1_idx]
    top_pc2 = [dimensions[i].replace('_', ' ').title() for i in top_pc2_idx]
    
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})\n"
                  f"Driven by: {', '.join(top_pc1)}", 
                  fontsize=9, weight='bold')
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})\n"
                  f"Driven by: {', '.join(top_pc2)}", 
                  fontsize=9, weight='bold')
    
    ax.set_title("Tournament Tactical Space (2022-24)", 
                 fontsize=13, weight='bold', pad=15)
    
    # Legend
    legend_labels = [f"{name} (n={sum(labels == i)})" 
                     for i, name in archetype_names.items()]
    handles = [plt.Line2D([0], [0], marker='o', color='w', 
                         markerfacecolor=colors[i], markersize=10, alpha=0.6)
              for i in range(len(archetype_names))]
    ax.legend(handles, legend_labels, fontsize=9, loc='best', 
             framealpha=0.95, edgecolor='gray')
    
    ax.grid(True, alpha=0.2, linestyle='--')
    ax.set_axisbelow(True)
    
    # Variance explained
    total_var = pca.explained_variance_ratio_[:2].sum()
    ax.text(0.02, 0.98, f"Total Variance Explained: {total_var:.1%}",
            transform=ax.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    return fig, pca

def plot_archetype_radars(cluster_centers, dimensions, archetype_names, 
                          profiles_df=None, figsize=(16, 4)):
    """Radar charts for each archetype (one per cluster)."""
    k = len(archetype_names)
    colors = ['#4895C4', '#A23B72', '#F18F01', '#06A77D']

    fig, axes = plt.subplots(1, k, figsize=figsize, subplot_kw=dict(projection='polar'))
    fig.subplots_adjust(wspace=0.3)
    
    if k == 1:
        axes = [axes]
    
    for cluster_id in range(k):
        ax = axes[cluster_id]
        center = cluster_centers.iloc[cluster_id]
        
        # Normalize values
        values = []
        for dim in dimensions:
            if profiles_df is not None:
                min_val = profiles_df[dim].min()
                max_val = profiles_df[dim].max()
                val = (center[dim] - min_val) / (max_val - min_val)
            else:
                val = center[dim]
            values.append(val)
        values.append(values[0])
        
        # Angles
        angles = np.linspace(0, 2 * np.pi, len(dimensions), endpoint=False).tolist()
        angles.append(angles[0])
        
        # Plot
        ax.plot(angles, values, 'o-', linewidth=2, color=colors[cluster_id], markersize=4)
        ax.fill(angles, values, alpha=0.25, color=colors[cluster_id])
        
        # Labels
        labels_clean = [d.replace('_', '\n').title() for d in dimensions]
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels_clean, fontsize=7)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.25, 0.5, 0.75])
        ax.set_yticklabels(['0.25', '0.5', '0.75'], fontsize=5, color='#999')
        ax.grid(True, alpha=0.3)
        
        # Title
        name = archetype_names[cluster_id]
        size = int(center.get('size', 0)) if 'size' in center.index else ''
        title = f"{name}\n(n={size})" if size else name
        ax.set_title(title, fontsize=10, weight='bold', pad=15, color=colors[cluster_id])
    
    plt.tight_layout()
    return fig


def plot_comparison_pizzas(df, p1_name, p2_name, p1_rank=None, p2_rank=None):
    """Side-by-side pizza charts comparing two players."""
    p1_data = df.filter(df['player'] == p1_name).to_pandas().iloc[0]
    p2_data = df.filter(df['player'] == p2_name).to_pandas().iloc[0]

    params = [
        "xg_volume_percentile", "finishing_quality_percentile", "Final_Third_Output",
        "progressive_passes_percentile", "progressive_carries_percentile", "xg_chain_percentile",
        "network_centrality_percentile", "team_involvement_percentile", "xg_buildup_percentile",
        "pressure_volume_percentile", "defensive_actions_percentile", "high_turnovers_percentile"
    ]
    labels = [
        "xG Volume", "Finishing", "Final 3rd\nOutput",
        "Prog. Passes", "Prog. Carries", "xG Chain",
        "Centrality", "Involvement", "xG Build-up",
        "Pressure", "Def. Actions", "High\nTurnovers"
    ]
    
    p1_vals = [round(p1_data[p], 1) for p in params]
    p2_vals = [round(p2_data[p], 1) for p in params]

    fig, axs = plt.subplots(1, 2, figsize=(12, 7), facecolor="white", 
                             subplot_kw=dict(projection='polar'))

    baker = PyPizza(
        params=labels, 
        background_color="white", 
        straight_line_color="#E0E0E0", 
        last_circle_color="#BCBCBC",
        last_circle_lw=2,
        inner_circle_size=20
    )

    colors = ["#1A78CF"]*3 + ["#FF9300"]*3 + ["#4E6111"]*3 + ["#D70232"]*3

    for i, vals in enumerate([p1_vals, p2_vals]):
        baker.make_pizza(
            vals, 
            ax=axs[i], 
            color_blank_space="same", 
            blank_alpha=0.15,
            slice_colors=colors,
            param_location=110,
            kwargs_params=dict(color="#222222", fontsize=8, va="center"),
            kwargs_values=dict(
                color="white", 
                fontsize=8, 
                zorder=3, 
                fontweight='bold',
                bbox=dict(
                    edgecolor="white", 
                    facecolor="#222222", 
                    boxstyle="round,pad=0.2", 
                    lw=1
                )
            )
        )
    
    t1_full = f"#{p1_rank}: {p1_name} ({p1_data['latest_club']})" if p1_rank else f"{p1_name} ({p1_data['latest_club']})"
    axs[0].set_title(t1_full, color="#1A78CF", size=11, weight='bold', pad=35)

    t2_full = f"#{p2_rank}: {p2_name} ({p2_data['latest_club']})" if p2_rank else f"{p2_name} ({p2_data['latest_club']})"
    axs[1].set_title(t2_full, color="#D70232", size=11, weight='bold', pad=35)

    plt.tight_layout(pad=1.5)
    plt.show()

