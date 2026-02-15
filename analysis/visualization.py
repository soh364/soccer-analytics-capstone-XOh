"""
Compact, publication-ready visualizations for tactical analysis.
All functions create multi-panel figures to minimize notebook clutter.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from math import pi
import pandas as pd


# Updated visualization.py - Replace plot_eda_compact function

def plot_eda_compact(profiles_df, dimensions, figsize=(16, 10)):
    """
    Create compact 2x2 EDA figure with clearer visualizations.
    """
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # Top left: Distribution summary (box plots instead of overlapping histograms)
    ax_dist = fig.add_subplot(gs[0, 0])
    
    # Select key dimensions and normalize for visualization
    key_dims = ['possession_dominance', 'offensive_threat', 'press_intensity', 
                'tempo', 'progression_intensity', 'buildup_complexity']
    
    # Normalize each dimension to 0-1 for comparison
    normalized_data = []
    labels = []
    for dim in key_dims:
        values = profiles_df[dim]
        normalized = (values - values.min()) / (values.max() - values.min())
        normalized_data.append(normalized)
        labels.append(dim.replace('_', ' ').title())
    
    bp = ax_dist.boxplot(normalized_data, labels=labels, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('#6BAED6')
        patch.set_alpha(0.7)
    
    ax_dist.set_ylabel('Normalized Value (0-1)', fontsize=10)
    ax_dist.set_title('Key Dimension Distributions', fontsize=12, weight='bold')
    ax_dist.grid(axis='y', alpha=0.3)
    plt.setp(ax_dist.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=8)
    
    # Top right: Correlation matrix WITH LABELS
    ax_corr = fig.add_subplot(gs[0, 1])
    corr_matrix = profiles_df[dimensions].corr()
    
    # Shortened labels for readability
    short_labels = [
        'Poss', 'Terr', 'P.Eff', 'Prog.I', 'Prog.M', 'Build',
        'xG', 'Tempo', 'Press.I', 'Def.L', 'Press.E', 'Counter'
    ]
    
    sns.heatmap(corr_matrix, ax=ax_corr, cmap='coolwarm', center=0, 
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
                xticklabels=short_labels, yticklabels=short_labels,
                annot=False, fmt='.2f')
    ax_corr.set_title('Correlation Matrix', fontsize=12, weight='bold')
    plt.setp(ax_corr.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=8)
    plt.setp(ax_corr.yaxis.get_majorticklabels(), rotation=0, fontsize=8)
    
    # Bottom left: Descriptive stats table
    ax_stats = fig.add_subplot(gs[1, 0])
    ax_stats.axis('off')
    
    stats_text = f"""
DATASET SUMMARY

Teams: {len(profiles_df)}
Dimensions: {len(dimensions)}

Key Tactical Statistics:
- Possession:        {profiles_df['possession_dominance'].mean():.1f}% ± {profiles_df['possession_dominance'].std():.1f}%
- xG per match:      {profiles_df['offensive_threat'].mean():.2f} ± {profiles_df['offensive_threat'].std():.2f}
- Press Intensity:   {profiles_df['press_intensity'].mean():.3f} ± {profiles_df['press_intensity'].std():.3f}
- Tempo:             {profiles_df['tempo'].mean():.2f} ± {profiles_df['tempo'].std():.2f} passes/seq
- Progression:       {profiles_df['progression_intensity'].mean():.1f} ± {profiles_df['progression_intensity'].std():.1f} actions
- Build-up Quality:  {profiles_df['buildup_complexity'].mean():.3f} ± {profiles_df['buildup_complexity'].std():.3f}

Range Analysis:
- Highest possession: {profiles_df['possession_dominance'].max():.1f}%
- Lowest possession:  {profiles_df['possession_dominance'].min():.1f}%
- Most intense press: {profiles_df['press_intensity'].max():.3f}
- Least intense press: {profiles_df['press_intensity'].min():.3f}
    """
    ax_stats.text(0.05, 0.5, stats_text, fontsize=9, verticalalignment='center',
                  family='monospace')
    
    # Bottom right: Top correlations (easier to interpret than PCA)
    ax_top_corr = fig.add_subplot(gs[1, 1])
    ax_top_corr.axis('off')
    
    # Get top positive and negative correlations
    corr_pairs = []
    for i in range(len(dimensions)):
        for j in range(i+1, len(dimensions)):
            corr_pairs.append((dimensions[i], dimensions[j], corr_matrix.iloc[i, j]))
    
    corr_pairs_sorted = sorted(corr_pairs, key=lambda x: abs(x[2]), reverse=True)
    
    corr_text = "STRONGEST CORRELATIONS\n" + "="*40 + "\n\n"
    corr_text += "Positive (teams similar on both):\n"
    for dim1, dim2, corr in corr_pairs_sorted[:5]:
        if corr > 0:
            corr_text += f"• {dim1.replace('_', ' ')[:15]:15s} ↔ {dim2.replace('_', ' ')[:15]:15s}  {corr:+.2f}\n"
    
    corr_text += "\nNegative (trade-offs):\n"
    for dim1, dim2, corr in corr_pairs_sorted:
        if corr < -0.3:
            corr_text += f"• {dim1.replace('_', ' ')[:15]:15s} ↔ {dim2.replace('_', ' ')[:15]:15s}  {corr:+.2f}\n"
    
    ax_top_corr.text(0.05, 0.95, corr_text, fontsize=8.5, verticalalignment='top',
                     family='monospace')
    
    plt.suptitle('Exploratory Data Analysis: Tactical Dimensions', 
                 fontsize=14, weight='bold', y=0.98)
    
    return fig, (ax_dist, ax_corr, ax_stats, ax_top_corr)


def plot_clustering_validation_compact(optimization_results, figsize=(12, 5)):
    """
    Create compact validation figure with elbow + silhouette side-by-side.
    
    Args:
        optimization_results: DataFrame from optimize_k()
        
    Returns:
        fig, axes
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Elbow plot
    ax1.plot(optimization_results['k'], optimization_results['inertia'], 
             'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('Number of Clusters (k)', fontsize=11)
    ax1.set_ylabel('Inertia', fontsize=11)
    ax1.set_title('Elbow Method', fontsize=12, weight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(optimization_results['k'])
    
    # Silhouette plot
    ax2.plot(optimization_results['k'], optimization_results['silhouette'], 
             'ro-', linewidth=2, markersize=8)
    ax2.set_xlabel('Number of Clusters (k)', fontsize=11)
    ax2.set_ylabel('Silhouette Score', fontsize=11)
    ax2.set_title('Silhouette Analysis', fontsize=12, weight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(optimization_results['k'])
    ax2.axhline(y=0.25, color='orange', linestyle='--', alpha=0.5, label='Acceptable threshold')
    ax2.legend()
    
    plt.tight_layout()
    return fig, (ax1, ax2)


# Replace plot_archetype_summary in visualization.py

def plot_archetype_summary(cluster_centers, dimensions, archetype_names, 
                            profiles_df=None, labels=None, figsize=(18, 10)):
    """
    Create comprehensive archetype summary:
    - Top row: Radar charts for each archetype
    - Bottom row: Heatmap (left) and PCA (right) side-by-side
    """
    k = len(cluster_centers)
    
    # Create figure with custom layout
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, k, height_ratios=[1.2, 1], hspace=0.3, wspace=0.3)
    
    # Colors
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#06A77D', '#D62828', '#6A4C93']
    
    # Top row: Radar charts
    for cluster_id in range(k):
        ax_radar = fig.add_subplot(gs[0, cluster_id], projection='polar')
        _create_radar_chart(cluster_centers.iloc[cluster_id], dimensions, 
                           profiles_df, ax_radar, colors[cluster_id], 
                           archetype_names[cluster_id], 
                           cluster_centers.iloc[cluster_id]['size'])
    
    # Bottom left: Heatmap (spans first k-1 columns)
    ax_heat = fig.add_subplot(gs[1, :k-1])
    _create_heatmap(cluster_centers, dimensions, archetype_names, ax_heat)
    
    # Bottom right: PCA (spans last column)
    if profiles_df is not None and labels is not None:
        ax_pca = fig.add_subplot(gs[1, k-1])
        _create_pca_with_clusters(profiles_df, dimensions, labels, archetype_names, ax_pca, colors)
    
    plt.suptitle('Tactical Archetype Profiles', fontsize=16, weight='bold', y=0.98)
    
    return fig


def _create_radar_chart(cluster_data, dimensions, profiles_df, ax, color, archetype_name, size):
    """Helper: Create single radar chart with FIXED normalization"""
    categories = [d.replace('_', '\n').title() for d in dimensions]
    
    # Normalize to 0-1 using GLOBAL min/max from all teams
    values = []
    for dim in dimensions:
        min_val = profiles_df[dim].min()
        max_val = profiles_df[dim].max()
        cluster_val = cluster_data[dim]
        normalized = (cluster_val - min_val) / (max_val - min_val)
        values.append(normalized)
    
    # Complete the circle
    values += values[:1]
    angles = [n / float(len(categories)) * 2 * pi for n in range(len(categories))]
    angles += angles[:1]
    
    # Plot
    ax.plot(angles, values, 'o-', linewidth=2, color=color)
    ax.fill(angles, values, alpha=0.25, color=color)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=7)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75])
    ax.set_yticklabels(['', '0.5', ''], size=7)
    ax.grid(True, alpha=0.4)
    ax.set_title(f'{archetype_name}\n(n={int(size)})', size=11, weight='bold', pad=15)


def _create_heatmap(cluster_centers, dimensions, archetype_names, ax):
    """Helper: Create heatmap of cluster centers"""
    # Prepare data
    heat_data = cluster_centers[dimensions].T
    heat_data.columns = [archetype_names[i] for i in range(len(cluster_centers))]
    
    # Better color scale - use actual values
    sns.heatmap(heat_data, ax=ax, cmap='RdYlGn', center=heat_data.values.mean(), 
                annot=True, fmt='.2f', cbar_kws={'label': 'Value'}, 
                linewidths=0.5, annot_kws={'size': 8})
    ax.set_ylabel('')
    ax.set_xlabel('')
    ax.set_title('Archetype Comparison Heatmap', fontsize=11, weight='bold', pad=10)
    
    # Rotate y-labels for readability
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)


def _create_pca_with_clusters(profiles_df, dimensions, labels, archetype_names, ax, colors):
    """Helper: Create PCA scatter with cluster colors"""
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    
    scaler = StandardScaler()
    scaled = scaler.fit_transform(profiles_df[dimensions])
    pca = PCA(n_components=2)
    coords = pca.fit_transform(scaled)
    
    k = len(np.unique(labels))
    for cluster_id in range(k):
        mask = labels == cluster_id
        ax.scatter(coords[mask, 0], coords[mask, 1], 
                  c=colors[cluster_id], label=archetype_names[cluster_id],
                  s=80, alpha=0.7, edgecolors='black', linewidth=0.5)
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=10)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=10)
    ax.set_title('PCA: Tactical Space', fontsize=11, weight='bold')
    ax.legend(loc='best', fontsize=8)
    ax.grid(alpha=0.3)


def save_figure(fig, filename, dpi=300):
    """Save figure with consistent settings"""
    fig.savefig(f'../figures/{filename}', dpi=dpi, bbox_inches='tight')
    print(f"Saved: outputs/figures/{filename}")