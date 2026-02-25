"""
2026 World Cup — Matchup Analysis
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pandas as pd
import numpy as np
import ipywidgets as widgets
from IPython.display import display

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
})

DIMENSIONS = {
    'Squad_Quality':  'Squad Quality',
    'Star_Power':     'Star Power',
    'Stability':      'Manager Stability',
    'Cohesion':       'Club Cohesion',
    'Recovery':       'Recovery Edge',
}

WEIGHTS = {
    'Squad_Quality':  0.40,
    'Star_Power':     0.25,
    'Stability':      0.20,
    'Cohesion':       0.10,
    'Recovery':       0.05,
}

CONF_COLORS = {
    'UEFA':     '#4dabf7',
    'CONMEBOL': '#69db7c',
    'CAF':      '#ffd43b',
    'CONCACAF': '#f783ac',
    'AFC':      '#da77f2',
}

# Helpers

def _normalise_dim(df: pd.DataFrame, col: str) -> pd.Series:
    mn, mx = df[col].min(), df[col].max()
    if mx == mn:
        return pd.Series([50.0] * len(df), index=df.index)
    return (df[col] - mn) / (mx - mn) * 100


def _build_norm(df: pd.DataFrame) -> pd.DataFrame:
    norm = df[['National_Team']].copy()
    for col in DIMENSIONS:
        norm[col] = _normalise_dim(df, col)
    return norm.set_index('National_Team')


def _net_advantage(row_team: str, col_team: str, norm: pd.DataFrame) -> float:
    if row_team == col_team:
        return np.nan
    diff = norm.loc[row_team] - norm.loc[col_team]
    weighted = sum(diff[col] * WEIGHTS[col] for col in DIMENSIONS)
    return round(weighted, 2)


# Heatmap

def plot_matchup_heatmap(
    df: pd.DataFrame,
    save_path: str = 'figures/wc2026_matchup_heatmap.png',
):
    norm  = _build_norm(df)
    teams = norm.index.tolist()

    matrix = pd.DataFrame(index=teams, columns=teams, dtype=float)
    for r in teams:
        for c in teams:
            matrix.loc[r, c] = _net_advantage(r, c, norm)

    fig, ax = plt.subplots(figsize=(14, 12))
    fig.patch.set_facecolor('#ffffff')

    sns.heatmap(
        matrix.astype(float),
        ax=ax,
        cmap='RdBu_r',
        center=0,
        annot=True,
        fmt='.1f',
        annot_kws={'size': 7, 'fontfamily': 'monospace'},
        linewidths=0.4,
        linecolor='#f0f0f0',
        cbar_kws={'label': 'Net Advantage (row − column)', 'shrink': 0.7},
        mask=pd.isna(matrix.astype(float)),
    )

    ax.set_title(
        '2026 WORLD CUP — HEAD-TO-HEAD ADVANTAGE MATRIX\n'
        'Positive (red) = row team favoured   |   Negative (blue) = column team favoured',
        fontsize=12, fontweight='bold', color='#1a1a2e', pad=16, fontfamily='monospace',
    )
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.tick_params(axis='x', rotation=45, labelsize=8)
    ax.tick_params(axis='y', rotation=0,  labelsize=8)
    ax.text(
        0.01, -0.11,
        'Weighted advantage: Squad Quality 40% | Star Power 25% | Stability 20% | Cohesion 10% | Recovery 5%',
        transform=ax.transAxes, fontsize=7.5, color='#868e96', style='italic',
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches='tight', facecolor='#ffffff')
    plt.show()
    print(f'Saved: {save_path}')


def plot_power_rankings(df: pd.DataFrame, save_path: str = 'figures/wc2026_rankings.png'):
    df = df.copy().reset_index(drop=True)

    # Tag each team based on quadrant
    x_mid = df['Readiness_Score'].median()
    y_mid = df['Star_Power'].median()

    def _tag(row):
        hi_r = row['Readiness_Score'] >= x_mid
        hi_s = row['Star_Power'] >= y_mid
        if hi_r and hi_s:     return 'ELITE',       '#2f9e44'
        if not hi_r and hi_s: return 'DARK HORSE',  '#f59f00'
        if hi_r and not hi_s: return 'GRINDER',     '#1971c2'
        return                       'ALSO-RAN',     '#868e96'

    df['Tag'], df['Tag_Color'] = zip(*df.apply(_tag, axis=1))

    # Normalise components for mini bars
    components = ['Squad_Quality', 'Star_Power', 'Stability', 'Cohesion']
    for c in components:
        mn, mx = df[c].min(), df[c].max()
        df[f'{c}_norm'] = (df[c] - mn) / (mx - mn) if mx > mn else 0.5

    n     = len(df)
    fig   = plt.figure(figsize=(16, n * 0.52 + 2))
    fig.patch.set_facecolor('#ffffff')
    ax    = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor('#ffffff')
    ax.set_xlim(0, 16)
    ax.set_ylim(0, n + 1.5)
    ax.axis('off')

    # Title
    ax.text(8, n + 1.1, '2026 WORLD CUP — POWER RANKINGS',
            ha='center', va='center', fontsize=18, fontweight='bold',
            fontfamily='monospace', color='#1a1a2e')
    ax.text(8, n + 0.6, 'Ranked by Readiness Score  |  Components normalised 0–100',
            ha='center', va='center', fontsize=9,
            fontfamily='monospace', color='#868e96', style='italic')

    # Column headers
    headers = [('RANK', 0.5), ('TEAM', 2.2), ('SQUAD', 5.2),
               ('STARS', 6.8), ('STABILITY', 8.6), ('COHESION', 10.3),
               ('SCORE', 12.8), ('PROFILE', 14.3)]
    for label, x in headers:
        ax.text(x, n + 0.1, label, ha='left', va='center',
                fontsize=7.5, fontfamily='monospace',
                color='#868e96', fontweight='bold')

    ax.axhline(n, color='#dee2e6', linewidth=0.8, xmin=0.02, xmax=0.98)

    for i, row in df.iterrows():
        y      = n - i - 0.5
        rank   = i + 1
        conf_c = CONF_COLORS.get(row['Confederation'], '#868e96')

        # Alternating row background
        if i % 2 == 0:
            ax.add_patch(plt.Rectangle((0.2, y - 0.42), 15.6, 0.84,
                         facecolor='#f8f9fa', edgecolor='none', zorder=0))

        # Confederation accent bar
        ax.add_patch(plt.Rectangle((0.2, y - 0.38), 0.18, 0.76,
                     facecolor=conf_c, edgecolor='none', zorder=1))

        # Rank
        ax.text(0.55, y, f'#{rank}', ha='left', va='center',
                fontsize=10, fontfamily='monospace',
                color='#1a1a2e', fontweight='bold')

        # Team name
        ax.text(2.2, y, row['National_Team'], ha='left', va='center',
                fontsize=10, fontfamily='monospace', color='#1a1a2e', fontweight='bold')

        # Mini bars for each component
        bar_x     = [5.2, 6.8, 8.4, 10.1]
        bar_width = 1.2
        bar_h     = 0.28

        for bx, comp in zip(bar_x, components):
            val  = row[f'{comp}_norm']
            fill = val * bar_width
            # Background track
            ax.add_patch(plt.Rectangle((bx, y - bar_h/2), bar_width, bar_h,
                         facecolor='#e9ecef', edgecolor='none', zorder=1))
            # Fill
            bar_color = conf_c if fill > 0 else '#e9ecef'
            ax.add_patch(plt.Rectangle((bx, y - bar_h/2), fill, bar_h,
                         facecolor=bar_color, edgecolor='none', alpha=0.85, zorder=2))

        # Readiness score
        ax.text(12.8, y, f'{row["Readiness_Score"]:.1f}', ha='left', va='center',
                fontsize=12, fontfamily='monospace',
                color='#1a1a2e', fontweight='bold')

        # Tag pill
        tag, tag_c = row['Tag'], row['Tag_Color']
        pill_x = 14.0
        ax.add_patch(mpatches.FancyBboxPatch(
            (pill_x, y - 0.22), 1.75, 0.44,
            boxstyle='round,pad=0.05',
            facecolor=tag_c, edgecolor='none', alpha=0.15, zorder=1,
        ))
        ax.text(pill_x + 0.875, y, tag, ha='center', va='center',
                fontsize=7, fontfamily='monospace',
                color=tag_c, fontweight='bold', zorder=2)

        # Divider
        ax.axhline(y - 0.42, color='#f1f3f5', linewidth=0.5,
                   xmin=0.02, xmax=0.98)

    # Footer
    ax.text(0.3, -0.3,
            'Squad Quality 40% · Star Power 25% · Manager Stability 20% · Club Cohesion 10% · Recovery 5%'
            '   |   Bubble size reflects data coverage',
            ha='left', va='center', fontsize=7,
            fontfamily='monospace', color='#adb5bd', style='italic')

    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='#ffffff')
    plt.show()
    print(f'Saved: {save_path}')