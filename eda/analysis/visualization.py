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
    from pathlib import Path
    figures_dir = Path('figures')
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures_dir / filename, dpi=dpi, bbox_inches='tight')
    print(f"Saved: figures/{filename}")

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
    return fig

"""
2026 World Cup Readiness — Visualizations
==========================================
1. Quadrant Plot: Readiness Score vs Archetype Tournament Success Rate
2. Upset Potential Index: which teams punch above their weight
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import seaborn as sns
import pandas as pd
import numpy as np

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':      'monospace',
    'axes.facecolor':   '#ffffff',
    'figure.facecolor': '#ffffff',
    'text.color':       '#1a1a2e',
    'axes.labelcolor':  '#1a1a2e',
    'xtick.color':      '#555555',
    'ytick.color':      '#555555',
    'axes.edgecolor':   '#cccccc',
    'grid.color':       '#e8e8e8',
    'grid.linewidth':   0.8,
})

# Confederation colours
CONF_COLORS = {
    'UEFA':     '#4dabf7',
    'CONMEBOL': '#69db7c',
    'CAF':      '#ffd43b',
    'CONCACAF': '#f783ac',
    'AFC':      '#da77f2',
}

# Archetype tournament success rates (from archetype_success.csv)
ARCHETYPE_SUCCESS = {
    'Elite Dominators':       68.75,   # r16_plus_pct
    'Conservative Pressers':  40.00,
    'Pragmatic Builders':     45.45,
    'Survival Mode':          23.08,
    'Unknown':                35.00,
}


# Quadrant Plot 

def plot_quadrant(df: pd.DataFrame, save_path: str = 'figures/wc2026_quadrant.png'):
    """
    X: Readiness Score  — overall squad quality + context factors
    Y: Star Power       — top-3 player ceiling
    Story:
      Top-right    → ELITE:        high floor + high ceiling (France, Germany)
      Top-left     → DARK HORSES:  weaker squad but match-winners (England, Croatia)
      Bottom-right → GRINDERS:     strong system, no superstar (Argentina, Spain)
      Bottom-left  → ALSO-RANS:    low floor, low ceiling
    """
    try:
        from adjustText import adjust_text
        HAS_ADJUST = True
    except ImportError:
        HAS_ADJUST = False

    df = df.copy()

    x     = df['Readiness_Score']
    y     = df['Star_Power']
    x_mid = x.median()
    y_mid = y.median()
    x_min, x_max = x.min() - 3,  x.max() + 5
    y_min, y_max = y.min() - 0.3, y.max() + 0.6

    fig, ax = plt.subplots(figsize=(15, 10))
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#fafafa')

    # Quadrant shading
    ax.axhspan(y_mid, y_max, xmin=0,   xmax=0.5, alpha=0.07, color='#f59f00', zorder=0)  # dark horses
    ax.axhspan(y_mid, y_max, xmin=0.5, xmax=1.0, alpha=0.07, color='#2f9e44', zorder=0)  # elite
    ax.axhspan(y_min, y_mid, xmin=0.5, xmax=1.0, alpha=0.07, color='#1971c2', zorder=0)  # grinders
    ax.axhspan(y_min, y_mid, xmin=0,   xmax=0.5, alpha=0.03, color='#868e96', zorder=0)  # also-rans

    # Dividers
    ax.axvline(x_mid, color='#ced4da', linewidth=1.0, linestyle='--', zorder=1)
    ax.axhline(y_mid, color='#ced4da', linewidth=1.0, linestyle='--', zorder=1)

    # Quadrant labels
    qkw = dict(fontsize=9, style='italic', fontfamily='monospace', alpha=0.45, fontweight='bold')
    ax.text(x_min + 0.5, y_max - 0.08, 'DARK HORSES',  color='#f59f00', **qkw)
    ax.text(x_mid + 0.3, y_max - 0.08, 'ELITE',        color='#2f9e44', **qkw)
    ax.text(x_mid + 0.3, y_min + 0.05, 'GRINDERS',     color='#1971c2', **qkw)
    ax.text(x_min + 0.5, y_min + 0.05, 'ALSO-RANS',    color='#868e96', **qkw)

    # Scatter + labels
    texts = []
    for _, row in df.iterrows():
        color = CONF_COLORS.get(row['Confederation'], '#868e96')
        size  = 160 + row['Players_Found'] * 20
        ax.scatter(
            row['Readiness_Score'], row['Star_Power'],
            s=size, color=color, alpha=0.88,
            edgecolors='white', linewidth=1.8, zorder=3,
        )
        t = ax.text(
            row['Readiness_Score'], row['Star_Power'],
            row['National_Team'],
            fontsize=8.5, fontfamily='monospace',
            color='#212529', fontweight='bold',
            ha='center', va='bottom', zorder=4,
        )
        texts.append(t)

    if HAS_ADJUST:
        adjust_text(
            texts, ax=ax,
            arrowprops=dict(arrowstyle='-', color='#adb5bd', lw=0.7),
            expand_points=(1.5, 1.8),
        )

    # Legend
    handles = [
        mpatches.Patch(facecolor=c, edgecolor='white', label=conf)
        for conf, c in CONF_COLORS.items()
        if conf in df['Confederation'].values
    ]
    leg = ax.legend(
        handles=handles, loc='upper left',
        fontsize=8.5, framealpha=0.9,
        facecolor='#f8f9fa', edgecolor='#dee2e6',
        title='Confederation', title_fontsize=8,
    )
    leg.get_title().set_fontfamily('monospace')

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel('Readiness Score  (squad quality + stability + context)  →',
                  fontsize=11, labelpad=12, color='#343a40')
    ax.set_ylabel('Star Power  (top-3 player ceiling above baseline)  →',
                  fontsize=11, labelpad=12, color='#343a40')
    ax.set_title(
        '2026 WORLD CUP — SQUAD FLOOR vs STAR CEILING',
        fontsize=14, fontweight='bold', color='#1a1a2e',
        pad=20, fontfamily='monospace',
    )
    ax.text(
        0.01, -0.06,
        '● Bubble size = players matched in database   '
        '|   Star Power = avg top-3 quality above dataset baseline',
        transform=ax.transAxes, fontsize=7.5, color='#868e96', style='italic',
    )

    ax.grid(True, alpha=0.35, linewidth=0.6)
    ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches='tight', facecolor='#ffffff')
    plt.show()
    print(f'Saved: {save_path}')
    if not HAS_ADJUST:
        print("Tip: pip install adjustText for better label placement")


# Upset Potential Index 

def compute_upset_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Upset Potential = (Star Power + Cohesion) / Readiness Score
    High = team with dangerous ceiling but inconsistent floor.
    Low  = consistent, hard to upset on a good day.
    """
    df = df.copy()
    df['Upset_Index'] = ((df['Star_Power'] + df['Cohesion']) / df['Readiness_Score'] * 100).round(2)
    return df.sort_values('Upset_Index', ascending=False).reset_index(drop=True)


def plot_upset_index(df: pd.DataFrame, save_path: str = 'figures/wc2026_upset.png'):
    df_upset = compute_upset_index(df)

    fig, ax = plt.subplots(figsize=(12, 9))
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#ffffff')

    colors = [CONF_COLORS.get(c, '#868e96') for c in df_upset['Confederation']]

    bars = ax.barh(
        df_upset['National_Team'],
        df_upset['Upset_Index'],
        color=colors, alpha=0.85, edgecolor='white', linewidth=0.8, height=0.65,
    )

    # Value labels
    for bar, val in zip(bars, df_upset['Upset_Index']):
        ax.text(bar.get_width() + 0.15, bar.get_y() + bar.get_height() / 2,
                f'{val:.1f}', va='center', fontsize=8.5,
                color='#495057', fontfamily='monospace', fontweight='bold')

    # Median line
    median_ui = df_upset['Upset_Index'].median()
    ax.axvline(median_ui, color='#adb5bd', linewidth=1.2, linestyle='--', zorder=0)
    ax.text(median_ui + 0.1, len(df_upset) - 0.5, 'median',
            fontsize=7.5, color='#adb5bd', fontfamily='monospace', style='italic')

    # Legend
    handles = [
        mpatches.Patch(facecolor=c, edgecolor='white', label=conf)
        for conf, c in CONF_COLORS.items()
    ]
    leg = ax.legend(
        handles=handles, loc='lower right',
        fontsize=8.5, framealpha=0.9,
        facecolor='#f8f9fa', edgecolor='#dee2e6',
        title='Confederation', title_fontsize=8,
    )
    leg.get_title().set_fontfamily('monospace')

    ax.invert_yaxis()
    ax.set_xlabel('Upset Potential Index  →', fontsize=11, labelpad=12, color='#343a40')
    ax.set_title(
        '2026 WORLD CUP — UPSET POTENTIAL INDEX\n(Star Power + Cohesion relative to overall Readiness Score)',
        fontsize=13, fontweight='bold', color='#1a1a2e',
        pad=16, fontfamily='monospace',
    )
    ax.set_xlim(0, df_upset['Upset_Index'].max() + 2)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='x', alpha=0.35, linewidth=0.6)
    ax.tick_params(axis='y', labelsize=9.5, colors='#343a40')
    ax.text(
        0.01, -0.06,
        'Index = (Star Power + Cohesion) / Readiness Score × 100   |   High = dangerous ceiling, inconsistent floor',
        transform=ax.transAxes, fontsize=7.5, color='#868e96', style='italic',
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches='tight', facecolor='#ffffff')
    plt.show()
    print(f'Saved: {save_path}')