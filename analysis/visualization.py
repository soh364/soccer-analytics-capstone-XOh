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


def plot_clustering_validation_compact(optimization_results, figsize=(7, 3.5)):
    """
    Create a reduced-size validation figure with elbow + silhouette side-by-side.
    """
    # We use a smaller constrained_layout to prevent label clipping at small sizes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)
    
    # Elbow plot
    ax1.plot(optimization_results['k'], optimization_results['inertia'], 
             'bo-', linewidth=1.5, markersize=5) # Reduced line/marker size for scale
    ax1.set_xlabel('k', fontsize=9)
    ax1.set_ylabel('Inertia', fontsize=9)
    ax1.set_title('Elbow Method', fontsize=10, weight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(optimization_results['k'])
    ax1.tick_params(labelsize=8) # Smaller ticks
    
    # Silhouette plot
    ax2.plot(optimization_results['k'], optimization_results['silhouette'], 
             'ro-', linewidth=1.5, markersize=5)
    ax2.set_xlabel('k', fontsize=9)
    ax2.set_ylabel('Score', fontsize=9)
    ax2.set_title('Silhouette Analysis', fontsize=10, weight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(optimization_results['k'])
    ax2.tick_params(labelsize=8)
    
    # Threshold line
    ax2.axhline(y=0.25, color='orange', linestyle='--', alpha=0.5, linewidth=1)
    
    return fig, (ax1, ax2)


def plot_archetype_radars(cluster_centers, dimensions, archetype_names, 
                          profiles_df=None, figsize=(15, 4)):
    """
    Clean radar charts only — one per archetype, side by side.
    """
    k = len(archetype_names)
    colors = ['#2d6a4f', '#e76f51', '#457b9d']
    
    fig, axes = plt.subplots(1, k, figsize=figsize, subplot_kw=dict(projection='polar'))
    fig.subplots_adjust(wspace=-0.75)
    if k == 1:
        axes = [axes]
    
    for cluster_id in range(k):
        ax = axes[cluster_id]
        center = cluster_centers.iloc[cluster_id]
        
        # Normalize values to 0-1 using global min/max from profiles
        values = []
        for dim in dimensions:
            if profiles_df is not None:
                val = (center[dim] - profiles_df[dim].min()) / (profiles_df[dim].max() - profiles_df[dim].min())
            else:
                val = center[dim]
            values.append(val)
        values.append(values[0])  # close the polygon
        
        # Angles
        angles = np.linspace(0, 2 * np.pi, len(dimensions), endpoint=False).tolist()
        angles.append(angles[0])
        
        # Plot
        ax.plot(angles, values, 'o-', linewidth=2, color=colors[cluster_id], markersize=4)
        ax.fill(angles, values, alpha=0.2, color=colors[cluster_id])
        
        # Labels
        labels_clean = [d.replace('_', '\n').title() for d in dimensions]
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels_clean, fontsize=7)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.25, 0.5, 0.75])
        ax.set_yticklabels(['0.25', '0.5', '0.75'], fontsize=8, color='#999')
        
        # Count teams in this archetype
        name = archetype_names[cluster_id]
        size = int(center.get('size', 0)) if 'size' in center.index else ''
        title = f"{name}\n(n={size})" if size else name
        ax.set_title(title, fontsize=11, weight='bold', pad=20, color=colors[cluster_id])
    
    plt.tight_layout()
    return fig


def plot_tactical_pca(profiles_df, dimensions, labels, archetype_names,
                      highlight_teams=None, figsize=(6, 6)):
    """
    PCA tactical map — the anchor visual. Separate from radars.
    
    Args:
        highlight_teams: Optional list of team names to annotate on the plot
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    
    colors = ['#2d6a4f', '#e76f51', '#457b9d']
    
    # PCA
    scaler = StandardScaler()
    scaled = scaler.fit_transform(profiles_df[dimensions])
    pca = PCA(n_components=2)
    coords = pca.fit_transform(scaled)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot each archetype
    for cluster_id in range(len(archetype_names)):
        mask = labels == cluster_id
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=colors[cluster_id], alpha=0.5, s=60, edgecolors='white',
                   linewidth=0.5, label=archetype_names[cluster_id], zorder=2)
        
        # Centroid
        cx, cy = coords[mask, 0].mean(), coords[mask, 1].mean()
        ax.scatter(cx, cy, c=colors[cluster_id], s=150, edgecolors='black',
                   linewidth=1, marker='D', zorder=3)
    
    # Annotate specific teams
    if highlight_teams and 'team' in profiles_df.columns:
        for team in highlight_teams:
            idx = profiles_df.index[profiles_df['team'] == team]
            if len(idx) > 0:
                i = idx[0]
                x, y = coords[i, 0], coords[i, 1]
                ax.annotate(team, (x, y), fontsize=8, 
                            xytext=(6, 6), textcoords='offset points',
                            arrowprops=dict(arrowstyle='-', color='#999', lw=0.8),
                            color='#1a1a2e', zorder=4)
    
    # Axis labels with variance explained
    ax.set_title("PCA: Tactical Space", fontsize=12, weight='bold')
    ax.legend(fontsize=8, loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.15)
    ax.set_axisbelow(True)

    loadings = pca.components_
    top_pc1 = [dimensions[i].replace('_', ' ').title() for i in np.argsort(np.abs(loadings[0]))[-2:]]
    top_pc2 = [dimensions[i].replace('_', ' ').title() for i in np.argsort(np.abs(loadings[1]))[-2:]]
    
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%}) — {', '.join(top_pc1)}", fontsize=9)
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%}) — {', '.join(top_pc2)}", fontsize=9)
    
    plt.tight_layout()
    return fig, pca


def _create_radar_chart(cluster_data, dimensions, profiles_df, ax, color, archetype_name, size):
    """Helper: Create single radar chart with FIXED normalization"""
    categories = [d.replace('_', '\n').title() for d in dimensions]
    
    # Normalize to 0-1 
    values = []
    for dim in dimensions:
            if profiles_df is not None:
                p5 = profiles_df[dim].quantile(0.05)
                p95 = profiles_df[dim].quantile(0.95)
                raw = np.clip((center[dim] - p5) / (p95 - p5), 0, 1)
                val = 0.5 + raw * 0.5
            else:
                val = center[dim]
            values.append(val)
    
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

def plot_league_distribution(profiles_df, archetype_names, figsize=(6, 4)):
    """
    Chart 4: Stacked bar chart of archetype proportions by league.
    
    Args:
        profiles_df: DataFrame with 'team', 'cluster', 'league' columns
        archetype_names: dict mapping cluster_id to name
        figsize: figure size tuple
    
    Returns:
        fig, ax
    """
    colors = ['#2d6a4f', '#e76f51', '#457b9d']
    league_order = ['La Liga', 'Premier League', 'Bundesliga', 'Serie A', 'Ligue 1']
    
    # Build proportions
    league_arch = (profiles_df[profiles_df['league'] != 'Unknown']
                   .groupby(['league', 'cluster']).size()
                   .unstack(fill_value=0))
    league_pct = league_arch.div(league_arch.sum(axis=1), axis=0) * 100
    league_pct = league_pct.reindex(league_order)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    bottom = np.zeros(len(league_order))
    for cluster_id in sorted(archetype_names.keys()):
        values = league_pct[cluster_id].values
        ax.bar(league_order, values, bottom=bottom, color=colors[cluster_id],
               label=archetype_names[cluster_id], edgecolor='white', linewidth=0.5)
        
        for i, (v, b) in enumerate(zip(values, bottom)):
            if v > 8:
                ax.text(i, b + v/2, f'{v:.0f}%', ha='center', va='center',
                        fontsize=9, fontweight='bold', color='white')
        
        bottom += values
    
    ax.set_ylabel('Share of Teams (%)', fontsize=10)
    ax.set_title('Archetype Distribution by League', fontsize=13, weight='bold')
    ax.legend(fontsize=9, loc='upper right', framealpha=0.9, edgecolor='#ddd')
    ax.set_ylim(0, 105)
    ax.tick_params(axis='both', labelsize=9)
    ax.grid(axis='y', alpha=0.12)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    return fig, ax

def plot_compression_overlay(baseline_df, recent_club_df, tournament_df,
                             archetype_names, figsize=(8, 6)):
    """
    Chart 6: Three-layer PCA overlay showing compression.
    Baseline (faint) → Same-era clubs (medium) → Tournament (bold)
    
    Args:
        baseline_df: 2015/16 club profiles with 'cluster', 'PC1', 'PC2'
        recent_club_df: Same-era club profiles with 'assigned_archetype', 'PC1', 'PC2'
        tournament_df: Tournament profiles with 'assigned_archetype', 'PC1', 'PC2'
        archetype_names: dict mapping cluster_id to name
    """
    from matplotlib.lines import Line2D
    
    colors = {0: '#2d6a4f', 1: '#e76f51', 2: '#457b9d'}
    fig, ax = plt.subplots(figsize=figsize)
    
    # Layer 1: Baseline (faintest)
    for cid in range(len(archetype_names)):
        mask = baseline_df['cluster'] == cid
        ax.scatter(baseline_df.loc[mask, 'PC1'], baseline_df.loc[mask, 'PC2'],
                   c=colors[cid], alpha=0.2, s=40, edgecolors='none', zorder=1)
    
    # Layer 2: Same-era clubs (medium)
    for cid in range(len(archetype_names)):
        mask = recent_club_df['assigned_archetype'] == cid
        if mask.sum() > 0:
            ax.scatter(recent_club_df.loc[mask, 'PC1'], recent_club_df.loc[mask, 'PC2'],
                       c=colors[cid], alpha=0.6, s=60, edgecolors='white',
                       linewidth=0.5, marker='^', zorder=2)
    
    # Layer 3: Tournament teams (boldest)
    for cid in range(len(archetype_names)):
        mask = tournament_df['assigned_archetype'] == cid
        if mask.sum() > 0:
            ax.scatter(tournament_df.loc[mask, 'PC1'], tournament_df.loc[mask, 'PC2'],
                       c=colors[cid], alpha=0.8, s=80, edgecolors='black',
                       linewidth=0.8, marker='s', zorder=3)
    
    # Legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
               alpha=0.2, markersize=8, label=f'2015/16 Baseline (n={len(baseline_df)})'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='gray',
               markeredgecolor='white', markersize=8, 
               label=f'Same-era clubs (n={len(recent_club_df)})'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='gray',
               markeredgecolor='black', markersize=8, 
               label=f'Tournament teams (n={len(tournament_df)})'),
    ]
    
    ax.legend(handles=legend_elements, fontsize=9, loc='upper left',
              framealpha=0.9, edgecolor='#ddd')
    ax.set_xlabel('PC1', fontsize=10)
    ax.set_ylabel('PC2', fontsize=10)
    ax.set_title('Three Eras on One Map: Baseline → Same-Era Clubs → Tournaments',
                 fontsize=13, weight='bold')
    ax.tick_params(axis='both', labelsize=7)
    ax.grid(True, alpha=0.12)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    return fig, ax

def print_cmi_table(cmi_results):
    """
    Display CMI results as a styled HTML table with colored bars.
    
    Args:
        cmi_results: DataFrame with 'dimension', 'cmi', 'compression_pct' columns
    """
    from IPython.display import display, HTML
    
    cmi_sorted = cmi_results.sort_values('cmi')
    
    rows_html = ""
    for _, row in cmi_sorted.iterrows():
        dim_name = row['dimension'].replace('_', ' ').title()
        cmi_val = row['cmi']
        comp_pct = row['compression_pct']
        
        if comp_pct > 15:
            color, tag = '#d62828', 'HIGH'
        elif comp_pct > 5:
            color, tag = '#e76f51', 'MOD'
        else:
            color, tag = '#2d6a4f', 'LOW'
        
        bar_width = min(int(abs(comp_pct) * 3), 200)
        bar = f'<div style="background:{color};width:{bar_width}px;height:14px;border-radius:3px;display:inline-block;"></div>'
        
        rows_html += f"""<tr>
            <td style="font-weight:600;color:#1a1a2e;">{dim_name}</td>
            <td style="font-family:'SF Mono',monospace;font-weight:700;color:#2d6a4f;">{cmi_val:.3f}</td>
            <td style="font-family:'SF Mono',monospace;color:{color};font-weight:700;">{comp_pct:+.1f}%</td>
            <td>{bar}</td>
            <td style="font-size:11px;font-weight:700;color:{color};">{tag}</td>
        </tr>"""
    
    html = f"""
    <style>
        .cmi {{ border-collapse:collapse; font-family:-apple-system,sans-serif; font-size:13px; }}
        .cmi th {{ background:#1a1a2e; color:white; padding:8px 14px; text-align:left; font-weight:600; }}
        .cmi td {{ padding:6px 14px; border-bottom:1px solid #e0e0e0; }}
        .cmi tr:hover {{ background:#f5f5f5; }}
    </style>
    <h4 style="font-family:-apple-system,sans-serif;color:#1a1a2e;margin-bottom:8px;">
        Compression by Dimension (Same-Era: 2022/23 Clubs → 2022-24 Tournaments)
    </h4>
    <table class="cmi">
        <tr><th>Dimension</th><th>CMI</th><th>Compression</th><th></th><th>Level</th></tr>
        {rows_html}
    </table>
    """
    display(HTML(html))

def plot_side_by_side(plot_func_left, plot_func_right, args_left, args_right,
                      figsize=(16, 6), suptitle=None):
    """
    Combine any two plot functions side by side.
    
    Each plot function must accept an 'ax' keyword argument.
    
    Usage:
        plot_side_by_side(
            plot_success_by_archetype, plot_progression_by_round,
            {'merged_df': df, 'archetype_names': names},
            {'merged_df': df, 'archetype_names': names},
            suptitle='Tournament Performance'
        )
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    plot_func_left(ax=ax1, **args_left)
    plot_func_right(ax=ax2, **args_right)
    
    if suptitle:
        fig.suptitle(suptitle, fontsize=14, weight='bold', y=1.02)
    
    plt.tight_layout()
    return fig, (ax1, ax2)