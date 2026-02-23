import polars as pl
import matplotlib.pyplot as plt
import numpy as np

def plot_metric_distributions(metrics: dict, figsize=(14, 8)):
    """Violin plots for all 8 tactical dimensions."""
    
    # Metric column mapping
    COL_MAP = {
        'ppda':        'ppda',
        'field_tilt':  'field_tilt_pct',
        'possession_pct': 'possession_pct',
        'epr':         'epr',
        'line_height': 'defensive_line_height',
        'xg':          'total_xg',
        'progression': None,  # derived
        'buildup':     'avg_xg_per_buildup_possession',
    }
    
    LABELS = {
        'ppda':           'PPDA\n(lower = more press)',
        'field_tilt':     'Field Tilt %',
        'possession_pct': 'Possession %',
        'epr':            'EPR\n(lower = more efficient)',
        'line_height':    'Defensive Line Height',
        'xg':             'Total xG',
        'progression':    'Progressive Carry %',
        'buildup':        'xG per Buildup Possession',
    }
    
    data = []
    labels = []
    
    for key, col in COL_MAP.items():
        if key == 'progression':
            df = metrics['progression']
            vals = (df['progressive_carries'] / 
                   (df['progressive_carries'] + df['progressive_passes']) * 100
                   ).drop_nulls().to_list()
        elif key in metrics and col:
            vals = metrics[key][col].drop_nulls().to_list()
        else:
            continue
        data.append(vals)
        labels.append(LABELS[key])
    
    fig, axes = plt.subplots(2, 4, figsize=figsize)
    fig.patch.set_facecolor('#ffffff')
    axes = axes.flatten()
    
    for i, (vals, label) in enumerate(zip(data, labels)):
        ax = axes[i]
        ax.set_facecolor('#fafafa')
        vp = ax.violinplot(vals, showmedians=True, showextrema=True)
        
        for pc in vp['bodies']:
            pc.set_facecolor('#4dabf7')
            pc.set_alpha(0.7)
        vp['cmedians'].set_color('#1d3557')
        vp['cmedians'].set_linewidth(2)
        
        ax.set_title(label, fontsize=9, fontweight='bold')
        ax.set_xticks([])
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=0.3)
        
        # Annotate median
        median = np.median(vals)
        ax.text(1.15, median, f'{median:.2f}',
                va='center', fontsize=8, color='#1d3557',
                fontweight='bold')
    
    fig.suptitle('Distribution of 8 Tactical Dimensions Across All Team-Matches',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('figures/2_1_metric_distributions.png', dpi=180,
                bbox_inches='tight', facecolor='#ffffff')
    plt.show()

def plot_correlation_matrix(metrics: dict, figsize=(10, 8)):
    """Heatmap of inter-metric correlations — validates non-redundancy."""
    import seaborn as sns
    
    # Build a flat df joined on match_id + team
    base = metrics['ppda'].rename({'ppda': 'PPDA'})
    
    joins = [
        ('field_tilt',     'field_tilt_pct',                  'Field Tilt %'),
        ('possession_pct', 'possession_pct',                  'Possession %'),
        ('epr',            'epr',                             'EPR'),
        ('line_height',    'defensive_line_height',           'Line Height'),
        ('xg',             'total_xg',                        'npxG'),
        ('buildup',        'avg_xg_per_buildup_possession',   'xG Buildup'),
    ]
    
    for key, col, alias in joins:
        if key in metrics:
            base = base.join(
                metrics[key].select(['match_id', 'team', pl.col(col).alias(alias)]),
                on=['match_id', 'team'], how='left'
            )
    
    # Add progressive carry %
    prog = metrics['progression'].with_columns(
        (pl.col('progressive_carries') /
         (pl.col('progressive_carries') + pl.col('progressive_passes')) * 100
        ).alias('Carry %')
    ).select(['match_id', 'team', 'Carry %'])
    base = base.join(prog, on=['match_id', 'team'], how='left')
    
    corr = base.drop(['match_id', 'team']).to_pandas().corr()
    
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor('#ffffff')
    
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, ax=ax, mask=mask,
        cmap='RdBu_r', center=0, vmin=-1, vmax=1,
        annot=True, fmt='.2f',
        annot_kws={'size': 9, 'fontfamily': 'monospace'},
        linewidths=0.5, linecolor='#f0f0f0',
        cbar_kws={'shrink': 0.8, 'label': 'Pearson r'},
        square=True,
    )
    
    ax.set_title('II.2 — Inter-Metric Correlation Matrix\nValidating non-redundancy of the 8-dimension framework',
                 fontsize=11, fontweight='bold', pad=14)
    ax.tick_params(axis='x', rotation=45, labelsize=9)
    ax.tick_params(axis='y', rotation=0, labelsize=9)
    
    plt.tight_layout()
    plt.savefig('figures/2_2_correlation_matrix.png', dpi=180,
                bbox_inches='tight', facecolor='#ffffff')
    plt.show()

def plot_metric_extremes(metrics: dict, figsize=(14, 10)):
    """Top 5 and bottom 5 teams per metric — validates metrics against known football reality."""
    
    # Aggregate metrics to team level (mean across matches)
    MIN_MATCHES = 5  

    def team_mean(key, col, alias):
        return (metrics[key]
                .group_by('team')
                .agg([
                    pl.col(col).mean().alias(alias),
                    pl.len().alias('n_matches')
                ])
                .filter(pl.col('n_matches') >= MIN_MATCHES))
    
    team_metrics = {
        'PPDA':       team_mean('ppda',         'ppda',                        'PPDA'),
        'Field Tilt': team_mean('field_tilt',   'field_tilt_pct',              'Field Tilt'),
        'Possession': team_mean('possession_pct','possession_pct',             'Possession'),
        'npxG':       team_mean('xg',           'total_xg',                    'npxG'),
        'Line Height':team_mean('line_height',  'defensive_line_height',       'Line Height'),
        'xG Buildup': team_mean('buildup',      'avg_xg_per_buildup_possession','xG Buildup'),
    }
    
    # For PPDA: lower = better press, so invert ranking label
    invert = {'PPDA', 'EPR'}
    
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    fig.patch.set_facecolor('#ffffff')
    axes = axes.flatten()
    
    for i, (metric_name, df) in enumerate(team_metrics.items()):
        ax = axes[i]
        ax.set_facecolor('#fafafa')
        col = df.columns[-1]
        
        ascending = metric_name in invert
        sorted_df = df.sort(col, descending=not ascending).head(8).to_pandas()
        
        colors = ['#2f9e44' if j < 3 else '#adb5bd' 
                  for j in range(len(sorted_df))]
        
        bars = ax.barh(sorted_df['team'], sorted_df[col],
                      color=colors, height=0.65,
                      edgecolor='white', alpha=0.88)
        
        for bar, val in zip(bars, sorted_df[col]):
            ax.text(bar.get_width() + sorted_df[col].max() * 0.01,
                   bar.get_y() + bar.get_height() / 2,
                   f'{val:.2f}', va='center', fontsize=8,
                   color='#343a40')
        
        ax.invert_yaxis()
        ax.set_title(metric_name, fontsize=10, fontweight='bold',
                    fontfamily='monospace')
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(axis='y', labelsize=8)
        ax.grid(axis='x', alpha=0.3)
    
    fig.suptitle('II.3 — Top Teams per Tactical Dimension\nSanity check: do the right teams appear?',
                 fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('figures/2_3_metric_extremes.png', dpi=180,
                bbox_inches='tight', facecolor='#ffffff')
    plt.show()

def plot_tactical_scatter(metrics: dict, figsize=(12, 8)):
    """
    PPDA vs Possession scatter — bubble size = npxG, 
    coloured by tournament result depth.
    """
    # You'll need a result depth map — add your own match outcomes here
    # 1=Group, 2=R16, 3=QF, 4=SF, 5=Final, 6=Winner
    RESULT_DEPTH = {
        # Finalists
        'Argentina': 6, 
        'France': 5, 
        
        # Semi-Finalists
        'Croatia': 4, 
        'Morocco': 4,
        
        # Quarter-Finalists
        'Netherlands': 3, 
        'Brazil': 3, 
        'England': 3, 
        'Portugal': 3,
        
        # Round of 16
        'Australia': 2, 
        'Senegal': 2, 
        'USA': 2, 
        'Poland': 2, 
        'South Korea': 2, 
        'Japan': 2, 
        'Spain': 2, 
        'Switzerland': 2,
        
        # Group Stage (Partial list of notable teams)
        'Germany': 1, 
        'Belgium': 1, 
        'Uruguay': 1, 
        'Mexico': 1, 
        'Ecuador': 1, 
        'Cameroon': 1, 
        'Tunisia': 1, 
        'Saudi Arabia': 1,
        'Iran': 1,
        'Costa Rica': 1,
        'Denmark': 1,
        'Serbia': 1,
        'Wales': 1,
        'Canada': 1,
        'Ghana': 1,
        'Qatar': 1
    }

    # Build joined df
    base = (metrics['ppda']
            .group_by('team').agg(pl.col('ppda').mean())
            .join(
                metrics['possession_pct'].group_by('team').agg(pl.col('possession_pct').mean()),
                on='team'
            )
            .join(
                metrics['xg'].group_by('team').agg(pl.col('total_xg').mean().alias('npxg')),
                on='team'
            ))

    df = base.to_pandas()
    df['depth'] = df['team'].map(RESULT_DEPTH).fillna(1)

    DEPTH_LABELS = {1: 'Group Exit', 2: 'Round of 16', 3: 'Quarter-Final',
                    4: 'Semi-Final', 5: 'Final', 6: 'Winner'}
    DEPTH_COLORS = {
        1: '#dee2e6',  # Group Exit — light grey
        2: '#74c0fc',  # R16 — light blue
        3: '#f4a261',  # QF — orange
        4: '#e63946',  # SF — red
        5: '#9775fa',  # Final — purple
        6: '#2d6a4f',  # Winner — dark green
    }

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#fafafa')

    for depth, group in df.groupby('depth'):
        ax.scatter(
            group['ppda'], group['possession_pct'],
            s=group['npxg'] * 120,
            c=DEPTH_COLORS.get(depth, '#adb5bd'),
            alpha=0.8, edgecolors='white', linewidths=0.8,
            label=DEPTH_LABELS.get(depth, 'Unknown'), zorder=3
        )
        LABEL_ALWAYS = {
            'Spain', 'Argentina', 'France', 'England', 'Portugal',
            'Germany', 'Morocco', 'Croatia', 'Netherlands', 'Brazil'
        }

        for _, row in df.iterrows():
            # Always label deep teams, only label group exits if no nearby neighbour
            if row['team'] in LABEL_ALWAYS or row['depth'] >= 3:
                ax.annotate(
                    row['team'],
                    (row['ppda'], row['possession_pct']),
                    xytext=(5, 4), textcoords='offset points',
                    fontsize=7, fontfamily='monospace', color='#343a40'
                )

    from matplotlib.patches import Ellipse

    winner_zone = Ellipse(
        xy=(8, 57), width=8, height=15,
        angle=15, fill=False,
        edgecolor='#2d6a4f', linestyle='--',
        linewidth=1.5, alpha=0.6, zorder=4
    )
    ax.add_patch(winner_zone)
    ax.text(4.5, 63.5, 'Winner Zone', fontsize=8,
            color='#2d6a4f', 
            fontstyle='italic')

    # Median lines
    ax.axvline(df['ppda'].median(), color='#868e96', linestyle=':', alpha=0.6, lw=1.2)
    ax.axhline(df['possession_pct'].median(), color='#868e96', linestyle=':', alpha=0.6, lw=1.2)

    # Quadrant labels
    xm, ym = df['ppda'].median(), df['possession_pct'].median()
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    quad_kw = dict(fontsize=8, color='#868e96',
                   fontstyle='italic', alpha=0.7)
    ax.text(xmin + 0.5, ymax - 1,  'High Press\nHigh Possession', **quad_kw)
    ax.text(xmax - 6,   ymax - 1,  'Low Press\nHigh Possession', **quad_kw, ha='right')
    ax.text(xmin + 0.5, ymin + 0.5,'High Press\nLow Possession', **quad_kw)
    ax.text(xmax - 6,   ymin + 0.5,'Low Press\nLow Possession', **quad_kw, ha='right')

    ax.set_xlabel('PPDA (lower = more pressing)', fontsize=10, color='#343a40')
    ax.set_ylabel('Possession %', fontsize=10, color='#343a40')
    ax.set_title(
        'Tactical Identity vs Tournament Depth (2022 FIFA World Cup)\n'
        'PPDA × Possession × npxG (bubble size) × Tournament Result',
        fontsize=11, fontweight='bold',
        loc='left', pad=14
    )
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(fontsize=8, frameon=False, title='Tournament Depth',
              title_fontsize=8, loc='upper right')
    ax.grid(alpha=0.2, zorder=0)

    # Bubble size legend
    for size, label in [(1.0, '1.0 npxG'), (2.0, '2.0 npxG'), (3.0, '3.0 npxG')]:
        ax.scatter([], [], s=size*120, c='#adb5bd', alpha=0.6,
                   edgecolors='white', label=label)

    plt.tight_layout()
    plt.savefig('figures/2_4_tactical_scatter.png', dpi=180,
                bbox_inches='tight', facecolor='#ffffff')
    plt.show()


import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def extract_match_results(statsbomb_dir: Path) -> pl.DataFrame:
    """
    Extract match outcomes from matches.parquet.
    Returns a long-format df with match_id, team, outcome (W/D/L).
    """
    matches = pl.read_parquet(statsbomb_dir / "matches.parquet")

    # Compute outcomes for home and away team
    home = matches.select([
        pl.col('match_id'),
        pl.col('home_team').alias('team'),
        pl.col('home_score'),
        pl.col('away_score'),
    ]).with_columns(
        pl.when(pl.col('home_score') > pl.col('away_score')).then(pl.lit('W'))
          .when(pl.col('home_score') < pl.col('away_score')).then(pl.lit('L'))
          .otherwise(pl.lit('D'))
          .alias('outcome')
    ).select(['match_id', 'team', 'outcome'])

    away = matches.select([
        pl.col('match_id'),
        pl.col('away_team').alias('team'),
        pl.col('home_score'),
        pl.col('away_score'),
    ]).with_columns(
        pl.when(pl.col('away_score') > pl.col('home_score')).then(pl.lit('W'))
          .when(pl.col('away_score') < pl.col('home_score')).then(pl.lit('L'))
          .otherwise(pl.lit('D'))
          .alias('outcome')
    ).select(['match_id', 'team', 'outcome'])

    return pl.concat([home, away])


def plot_outcome_by_quadrant(
    metrics: dict,
    statsbomb_dir: Path,
    figsize=(14, 5)
):
    """
    Win rate by tactical quadrant — extracts match results automatically.
    Left panel: stacked W/D/L bar per quadrant.
    Right panel: clean win rate ranking.
    """

    # ── Extract results ───────────────────────────────────────────────────────
    results_df = extract_match_results(statsbomb_dir)

    # ── Build per-match tactical base ─────────────────────────────────────────
    ppda = metrics['ppda']
    poss = metrics['possession_pct']
    base = ppda.join(poss, on=['match_id', 'team'])

    ppda_med = base['ppda'].median()
    poss_med = base['possession_pct'].median()

    base = base.with_columns(
        pl.when(
            (pl.col('ppda') <= ppda_med) & (pl.col('possession_pct') >= poss_med)
        ).then(pl.lit('High Press\nHigh Possession'))
        .when(
            (pl.col('ppda') > ppda_med) & (pl.col('possession_pct') >= poss_med)
        ).then(pl.lit('Low Press\nHigh Possession'))
        .when(
            (pl.col('ppda') <= ppda_med) & (pl.col('possession_pct') < poss_med)
        ).then(pl.lit('High Press\nLow Possession'))
        .otherwise(pl.lit('Low Press\nLow Possession'))
        .alias('quadrant')
    )

    # ── Join outcomes ─────────────────────────────────────────────────────────
    base = base.join(results_df, on=['match_id', 'team'], how='left')
    df = base.to_pandas()

    df['win']  = (df['outcome'] == 'W').astype(int)
    df['draw'] = (df['outcome'] == 'D').astype(int)
    df['loss'] = (df['outcome'] == 'L').astype(int)

    summary = (
        df.groupby('quadrant')
          .agg(
              win_rate=('win',  'mean'),
              draw_rate=('draw', 'mean'),
              loss_rate=('loss', 'mean'),
              n=('win', 'count')
          )
          .reset_index()
          .sort_values('win_rate', ascending=False)
    )

    # ── Colors ────────────────────────────────────────────────────────────────
    QUAD_COLORS = {
        'High Press\nHigh Possession': '#2d6a4f',
        'High Press\nLow Possession':  '#4dabf7',
        'Low Press\nHigh Possession':  '#f4a261',
        'Low Press\nLow Possession':   '#dee2e6',
    }

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=figsize,
        gridspec_kw={'width_ratios': [1.5, 1]}
    )
    fig.patch.set_facecolor('#ffffff')
    for ax in (ax1, ax2):
        ax.set_facecolor('#fafafa')

    # ── Left: stacked W/D/L ───────────────────────────────────────────────────
    quads  = summary['quadrant'].tolist()
    wins   = summary['win_rate'].tolist()
    draws  = summary['draw_rate'].tolist()
    losses = summary['loss_rate'].tolist()
    ns     = summary['n'].tolist()
    colors = [QUAD_COLORS.get(q, '#adb5bd') for q in quads]
    x      = np.arange(len(quads))

    # Win bars
    ax1.bar(x, wins, color=colors, alpha=0.92,
            edgecolor='white', linewidth=0.8, label='Win')
    # Draw bars
    ax1.bar(x, draws, bottom=wins,
            color=colors, alpha=0.45,
            edgecolor='white', linewidth=0.8, label='Draw')
    # Loss bars
    ax1.bar(x, losses,
            bottom=[w + d for w, d in zip(wins, draws)],
            color=colors, alpha=0.18,
            edgecolor='white', linewidth=0.8, label='Loss')

    # Labels inside bars
    for i, (w, d, l, n) in enumerate(zip(wins, draws, losses, ns)):
        if w > 0.05:
            ax1.text(i, w / 2, f'{w:.0%}',
                     ha='center', va='center', fontsize=9.5,
                     fontweight='bold', fontfamily='monospace', color='white')
        if d > 0.05:
            ax1.text(i, w + d / 2, f'{d:.0%}',
                     ha='center', va='center', fontsize=8,
                     fontfamily='monospace', color='#495057')
        if l > 0.05:
            ax1.text(i, w + d + l / 2, f'{l:.0%}',
                     ha='center', va='center', fontsize=8,
                     fontfamily='monospace', color='#868e96')
        ax1.text(i, -0.05, f'n={n}',
                 ha='center', fontsize=7.5,
                 fontfamily='monospace', color='#868e96')

    ax1.set_xticks(x)
    ax1.set_xticklabels(quads, fontsize=8.5, fontfamily='monospace')
    ax1.set_ylabel('Match Outcome Rate', fontsize=9, color='#343a40')
    ax1.set_ylim(-0.1, 1.08)
    ax1.set_title('Win / Draw / Loss Rate by Tactical Quadrant',
                  fontsize=10, fontweight='bold',
                  fontfamily='monospace', loc='left')
    ax1.spines[['top', 'right']].set_visible(False)
    ax1.legend(fontsize=8, frameon=False, loc='upper right')
    ax1.grid(axis='y', alpha=0.25, linestyle=':')

    # ── Right: win rate horizontal bar ────────────────────────────────────────
    summary_asc = summary.sort_values('win_rate', ascending=True)
    bar_colors_r = [QUAD_COLORS.get(q, '#adb5bd')
                    for q in summary_asc['quadrant']]

    bars = ax2.barh(
        summary_asc['quadrant'], summary_asc['win_rate'],
        color=bar_colors_r, height=0.45,
        alpha=0.88, edgecolor='white'
    )

    for bar, val in zip(bars, summary_asc['win_rate']):
        ax2.text(
            val + 0.005, bar.get_y() + bar.get_height() / 2,
            f'{val:.0%}', va='center', fontsize=9,
            fontfamily='monospace', fontweight='bold', color='#343a40'
        )

    ax2.set_xlim(0, max(summary['win_rate']) * 1.3)
    ax2.set_title('Win Rate Ranking',
                  fontsize=10, fontweight='bold',
                  fontfamily='monospace', loc='left')
    ax2.spines[['top', 'right']].set_visible(False)
    ax2.tick_params(axis='y', labelsize=8)
    ax2.grid(axis='x', alpha=0.25, linestyle=':')

    fig.suptitle(
        'II.5 — Does Tactical Identity Predict Match Outcomes?\n'
        'WC 2022 + Euro 2024  |  Quadrants split on dataset medians',
        fontsize=11, fontweight='bold',
        fontfamily='monospace', y=1.03
    )

    plt.tight_layout()
    plt.savefig('figures/2_5_outcome_by_quadrant.png',
                dpi=180, bbox_inches='tight', facecolor='#ffffff')
    plt.show()

import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from pathlib import Path
import math


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: build flat joined df from metrics dict
# ─────────────────────────────────────────────────────────────────────────────

def _build_flat(metrics: dict) -> pl.DataFrame:
    """Join all 8 metrics into a single match-level df."""
    prog = metrics['progression'].with_columns(
        (pl.col('progressive_carries') /
         (pl.col('progressive_carries') + pl.col('progressive_passes')) * 100
        ).alias('carry_pct')
    ).select(['match_id', 'team', 'carry_pct'])

    base = (
        metrics['ppda']
        .join(metrics['field_tilt'],    on=['match_id', 'team'], how='left')
        .join(metrics['possession_pct'],on=['match_id', 'team'], how='left')
        .join(metrics['epr'].with_columns(pl.col('epr').clip(0, 200)),
              on=['match_id', 'team'], how='left')
        .join(metrics['line_height'],   on=['match_id', 'team'], how='left')
        .join(metrics['xg'],            on=['match_id', 'team'], how='left')
        .join(metrics['buildup'],       on=['match_id', 'team'], how='left')
        .join(prog,                     on=['match_id', 'team'], how='left')
    )
    return base


DIM_COLS = ['ppda', 'field_tilt_pct', 'possession_pct', 'epr',
            'defensive_line_height', 'total_xg',
            'avg_xg_per_buildup_possession', 'carry_pct']

DIM_LABELS = ['PPDA', 'Field Tilt %', 'Possession %', 'EPR',
              'Line Height', 'npxG', 'xG Buildup', 'Carry %']

# For PPDA and EPR, lower = "more" of the concept, so invert for radar display
INVERT = {'ppda', 'epr'}


# ─────────────────────────────────────────────────────────────────────────────
# 1. RADAR — WC 2022 vs Euro 2024
# ─────────────────────────────────────────────────────────────────────────────

def plot_competition_radar(metrics: dict, statsbomb_dir: Path, figsize=(10, 8)):
    """
    Radar chart comparing average tactical profile:
    WC 2022 vs Euro 2024.
    """
    matches = pl.read_parquet(statsbomb_dir / "matches.parquet")

    comp_map = matches.select([
        pl.col('match_id'),
        pl.col('competition_name'),
    ])

    flat = _build_flat(metrics).join(comp_map, on='match_id', how='left')

    wc     = flat.filter(pl.col('competition_name').str.contains('World Cup'))
    euro   = flat.filter(pl.col('competition_name').str.contains('Euro'))
    copa   = flat.filter(pl.col('competition_name').str.contains('Copa America'))
    afcon  = flat.filter(pl.col('competition_name').str.contains('African Cup'))

    def norm_means(df):
        """Return 0-1 normalised mean per dimension (inverted where needed)."""
        means = {}
        for col in DIM_COLS:
            if col not in df.columns:
                means[col] = 0.5
                continue
            col_data = flat[col].drop_nulls()
            mn, mx = col_data.min(), col_data.max()
            val = df[col].mean()
            normed = (val - mn) / (mx - mn) if mx != mn else 0.5
            if col in INVERT:
                normed = 1 - normed
            means[col] = normed
        return means

    wc_means   = norm_means(wc)
    euro_means = norm_means(euro)
    copa_means  = norm_means(copa)
    afcon_means = norm_means(afcon)

    copa_vals  = [copa_means[c]  for c in DIM_COLS] + [copa_means[DIM_COLS[0]]]
    afcon_vals = [afcon_means[c] for c in DIM_COLS] + [afcon_means[DIM_COLS[0]]]

    # Radar setup
    N      = len(DIM_COLS)
    angles = [n / float(N) * 2 * math.pi for n in range(N)]
    angles += angles[:1]

    wc_vals   = [wc_means[c]   for c in DIM_COLS] + [wc_means[DIM_COLS[0]]]
    euro_vals = [euro_means[c] for c in DIM_COLS] + [euro_means[DIM_COLS[0]]]

    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#fafafa')

    # Grid
    ax.set_theta_offset(math.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(DIM_LABELS, fontsize=9.5,
                       fontfamily='monospace', color='#343a40')
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75])
    ax.set_yticklabels(['0.25', '0.50', '0.75'], fontsize=7, color='#adb5bd')
    ax.spines['polar'].set_color('#dee2e6')
    ax.grid(color='#dee2e6', linewidth=0.8)

    # WC 2022
    ax.plot(angles, wc_vals, color='#1d3557', linewidth=2.2,
            linestyle='-', label='WC 2022')
    ax.fill(angles, wc_vals, color='#1d3557', alpha=0.12)

    # Euro 2024
    ax.plot(angles, euro_vals, color='#e63946', linewidth=2.2,
            linestyle='--', label='Euro 2024')
    ax.fill(angles, euro_vals, color='#e63946', alpha=0.10)


    ax.plot(angles, copa_vals,  color='#f4a261', linewidth=2.2,
            linestyle='-.', label='Copa América')
    ax.fill(angles, copa_vals,  color='#f4a261', alpha=0.08)

    ax.plot(angles, afcon_vals, color='#c77dff', linewidth=2.2,
            linestyle=':', label='AFCON')
    ax.fill(angles, afcon_vals, color='#c77dff', alpha=0.08)

    ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.1),
              fontsize=9, frameon=False)

    ax.set_title(
        'II.6 — Average Tactical Profile by Tournament\n'
        'WC 2022 vs Euro 2024  |  Normalised 0–1 (higher = more of concept)',
        fontsize=11, fontweight='bold', fontfamily='monospace',
        pad=20, loc='center'
    )

    plt.tight_layout()
    plt.savefig('figures/2_6_competition_radar.png',
                dpi=180, bbox_inches='tight', facecolor='#ffffff')
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# 2. MATCH VOLATILITY — team consistency across matches
# ─────────────────────────────────────────────────────────────────────────────

def plot_match_volatility(metrics: dict, min_matches: int = 3, figsize=(14, 6)):
    """
    Dot plot of team-level std deviation for PPDA and Possession %.
    High std = tactically inconsistent. Low std = disciplined system.
    """
    flat = _build_flat(metrics).to_pandas()

    # Filter to teams with enough matches
    counts = flat.groupby('team')['match_id'].count()
    valid_teams = counts[counts >= min_matches].index
    flat = flat[flat['team'].isin(valid_teams)]

    vol = (flat.groupby('team')
               .agg(ppda_std=('ppda', 'std'),
                    poss_std=('possession_pct', 'std'),
                    n=('match_id', 'count'))
               .reset_index()
               .dropna())

    # Highlight teams we care about for 2026
    HIGHLIGHT = {
        'Spain', 'France', 'Germany', 'England', 'Argentina',
        'Brazil', 'Portugal', 'Netherlands', 'Morocco', 'Croatia'
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    fig.patch.set_facecolor('#ffffff')

    for ax, col, label, title in [
        (ax1, 'ppda_std', 'Std Dev of PPDA',
         'Pressing Consistency\n(lower = more disciplined)'),
        (ax2, 'poss_std', 'Std Dev of Possession %',
         'Possession Consistency\n(lower = more disciplined)'),
    ]:
        ax.set_facecolor('#fafafa')
        sorted_vol = vol.sort_values(col, ascending=True)

        colors = ['#2d6a4f' if t in HIGHLIGHT else '#adb5bd'
                  for t in sorted_vol['team']]

        ax.barh(sorted_vol['team'], sorted_vol[col],
                color=colors, height=0.65,
                edgecolor='white', alpha=0.85)

        # Label highlighted teams' values
        for _, row in sorted_vol.iterrows():
            if row['team'] in HIGHLIGHT:
                ax.text(row[col] + sorted_vol[col].max() * 0.01,
                        row['team'], f"{row[col]:.1f}",
                        va='center', fontsize=7.5,
                        fontfamily='monospace', color='#2d6a4f',
                        fontweight='bold')

        ax.set_xlabel(label, fontsize=9, color='#343a40')
        ax.set_title(title, fontsize=10, fontweight='bold',
                     fontfamily='monospace', loc='left')
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(axis='y', labelsize=7)
        ax.grid(axis='x', alpha=0.25, linestyle=':')

    # Legend
    handles = [
        mpatches.Patch(color='#2d6a4f', label='2026 WC team'),
        mpatches.Patch(color='#adb5bd', label='Other'),
    ]
    fig.legend(handles=handles, fontsize=8, frameon=False,
               loc='lower right', bbox_to_anchor=(0.98, 0.02))

    fig.suptitle(
        'II.7 — Tactical Consistency: Match-to-Match Volatility by Team\n'
        'Standard deviation across all matches  |  min 3 matches',
        fontsize=11, fontweight='bold', fontfamily='monospace', y=1.02
    )

    plt.tight_layout()
    plt.savefig('figures/2_7_match_volatility.png',
                dpi=180, bbox_inches='tight', facecolor='#ffffff')
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# 3. WINNER PROFILE — deep runs vs early exits across all 8 dimensions
# ─────────────────────────────────────────────────────────────────────────────

def plot_winner_profile(
    metrics: dict,
    depth_map: dict,
    figsize=(14, 6)
):
    """
    Grouped bar chart: mean of each tactical dimension for
    'Deep Run' (SF+) vs 'Early Exit' (Group + R16).

    depth_map: dict mapping team name -> tournament depth int
               (1=Group, 2=R16, 3=QF, 4=SF, 5=Final, 6=Winner)
    """
    flat = _build_flat(metrics).to_pandas()
    flat['depth'] = flat['team'].map(depth_map)
    flat = flat.dropna(subset=['depth'])

    flat['group'] = flat['depth'].apply(
        lambda d: 'Deep Run (SF+)' if d >= 4 else 'Early Exit (≤R16)'
    )

    # Normalise each dimension 0-1 for comparability
    for col in DIM_COLS:
        if col not in flat.columns:
            continue
        mn, mx = flat[col].min(), flat[col].max()
        if mx != mn:
            flat[f'{col}_norm'] = (flat[col] - mn) / (mx - mn)
            if col in INVERT:
                flat[f'{col}_norm'] = 1 - flat[f'{col}_norm']
        else:
            flat[f'{col}_norm'] = 0.5

    norm_cols = [f'{c}_norm' for c in DIM_COLS if f'{c}_norm' in flat.columns]

    summary = flat.groupby('group')[norm_cols].mean().reset_index()

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#fafafa')

    x     = np.arange(len(norm_cols))
    width = 0.35

    deep_row  = summary[summary['group'] == 'Deep Run (SF+)'].iloc[0]
    early_row = summary[summary['group'] == 'Early Exit (≤R16)'].iloc[0]

    deep_vals  = [deep_row[c]  for c in norm_cols]
    early_vals = [early_row[c] for c in norm_cols]

    bars1 = ax.bar(x - width/2, deep_vals,  width,
                   color='#2d6a4f', alpha=0.88,
                   edgecolor='white', label='Deep Run (SF+)')
    bars2 = ax.bar(x + width/2, early_vals, width,
                   color='#adb5bd', alpha=0.75,
                   edgecolor='white', label='Early Exit (≤R16)')

    # Delta annotations above each pair
    for i, (d, e) in enumerate(zip(deep_vals, early_vals)):
        delta = d - e
        color = '#2d6a4f' if delta > 0 else '#e63946'
        sign  = '+' if delta > 0 else ''
        ax.text(i, max(d, e) + 0.02, f'{sign}{delta:.2f}',
                ha='center', fontsize=8,
                fontfamily='monospace', color=color, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(
        [l + ('\n↑ better' if c not in INVERT else '\n↓ better')
         for l, c in zip(DIM_LABELS, DIM_COLS)],
        fontsize=8.5, fontfamily='monospace'
    )
    ax.set_ylabel('Normalised Score (0–1)', fontsize=9, color='#343a40')
    ax.set_ylim(0, 1.15)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(fontsize=9, frameon=False)
    ax.grid(axis='y', alpha=0.25, linestyle=':')

    ax.set_title(
        'II.8 — Tactical DNA of Tournament Success\n'
        'Mean normalised score: Deep Runs (SF+) vs Early Exits  |  delta labelled above',
        fontsize=11, fontweight='bold', fontfamily='monospace',
        loc='left', pad=14
    )

    plt.tight_layout()
    plt.savefig('figures/2_8_winner_profile.png',
                dpi=180, bbox_inches='tight', facecolor='#ffffff')
    plt.show()

import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path


# 2026 World Cup qualified teams — update as needed
WC_2026_TEAMS = {
    'Argentina', 'France', 'England', 'Spain', 'Germany', 'Portugal',
    'Brazil', 'Netherlands', 'Belgium', 'Croatia', 'Morocco', 'Uruguay',
    'United States', 'Mexico', 'Canada', 'Japan', 'South Korea',
    'Senegal', 'Colombia', 'Ecuador', 'Switzerland', 'Denmark',
    'Austria', 'Turkey', 'Poland', 'Australia',
}

# Confederation colors for dot coloring
CONF_COLORS = {
    'Argentina': '#69db7c',   # CONMEBOL
    'Brazil':    '#69db7c',
    'Uruguay':   '#69db7c',
    'Colombia':  '#69db7c',
    'Ecuador':   '#69db7c',
    'France':    '#4dabf7',   # UEFA
    'England':   '#4dabf7',
    'Spain':     '#4dabf7',
    'Germany':   '#4dabf7',
    'Portugal':  '#4dabf7',
    'Netherlands':'#4dabf7',
    'Belgium':   '#4dabf7',
    'Croatia':   '#4dabf7',
    'Switzerland':'#4dabf7',
    'Denmark':   '#4dabf7',
    'Austria':   '#4dabf7',
    'Turkey':    '#4dabf7',
    'Poland':    '#4dabf7',
    'Morocco':   '#ffd43b',   # CAF
    'Senegal':   '#ffd43b',
    'United States': '#f783ac', # CONCACAF
    'Mexico':    '#f783ac',
    'Canada':    '#f783ac',
    'Japan':     '#da77f2',   # AFC
    'South Korea':'#da77f2',
    'Australia': '#da77f2',
}

CONF_LABELS = {
    '#69db7c': 'CONMEBOL',
    '#4dabf7': 'UEFA',
    '#ffd43b': 'CAF',
    '#f783ac': 'CONCACAF',
    '#da77f2': 'AFC',
}


def plot_tactical_consistency_scatter(
    metrics: dict,
    min_matches: int = 3,
    figsize=(11, 9)
):
    """
    Scatter: PPDA std dev (x) vs Possession std dev (y)
    for 2026 World Cup teams only.
    Four quadrants: Tactically Rigid / Pressing-Adaptive /
                    Possession-Adaptive / Chaotic
    """

    # ── Build flat match-level df ─────────────────────────────────────────────
    flat = (
        metrics['ppda']
        .join(metrics['possession_pct'], on=['match_id', 'team'], how='left')
        .to_pandas()
    )

    # Filter to 2026 teams with enough matches
    flat = flat[flat['team'].isin(WC_2026_TEAMS)]
    counts = flat.groupby('team')['match_id'].count()
    valid  = counts[counts >= min_matches].index
    flat   = flat[flat['team'].isin(valid)]

    vol = (
        flat.groupby('team')
            .agg(ppda_std=('ppda', 'std'),
                 poss_std=('possession_pct', 'std'),
                 n=('match_id', 'count'))
            .reset_index()
            .dropna()
    )

    ppda_med = vol['ppda_std'].median()
    poss_med = vol['poss_std'].median()

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#fafafa')

    # Quadrant shading
    xmin, xmax = vol['ppda_std'].min() - 0.5, vol['ppda_std'].max() + 1.5
    ymin, ymax = vol['poss_std'].min() - 0.5, vol['poss_std'].max() + 1.5

    quad_kw = dict(alpha=0.04, zorder=0)
    ax.fill_betweenx([ymin, poss_med], xmin, ppda_med,
                     color='#2d6a4f', **quad_kw)   # rigid
    ax.fill_betweenx([poss_med, ymax], xmin, ppda_med,
                     color='#4dabf7', **quad_kw)   # possession-adaptive
    ax.fill_betweenx([ymin, poss_med], ppda_med, xmax,
                     color='#f4a261', **quad_kw)   # pressing-adaptive
    ax.fill_betweenx([poss_med, ymax], ppda_med, xmax,
                     color='#e63946', **quad_kw)   # chaotic

    # Median lines
    ax.axvline(ppda_med, color='#868e96', linestyle=':', linewidth=1.2, alpha=0.7)
    ax.axhline(poss_med, color='#868e96', linestyle=':', linewidth=1.2, alpha=0.7)

    # Quadrant labels
    label_kw = dict(fontsize=8.5, fontfamily='monospace',
                    fontstyle='italic', alpha=0.55)
    ax.text(xmin + 0.1, ymin + 0.2,  'TACTICALLY RIGID',       color='#2d6a4f', **label_kw)
    ax.text(xmin + 0.1, ymax - 0.5,  'POSSESSION-ADAPTIVE',    color='#4dabf7', **label_kw)
    ax.text(ppda_med + 0.1, ymin + 0.2, 'PRESSING-ADAPTIVE',   color='#f4a261', **label_kw)
    ax.text(ppda_med + 0.1, ymax - 0.5, 'CHAOTIC',             color='#e63946', **label_kw)

    # Scatter points
    for _, row in vol.iterrows():
        team   = row['team']
        color  = CONF_COLORS.get(team, '#adb5bd')
        ax.scatter(row['ppda_std'], row['poss_std'],
                   color=color, s=90, zorder=4,
                   edgecolors='white', linewidths=0.8, alpha=0.92)
        ax.annotate(
            team,
            (row['ppda_std'], row['poss_std']),
            xytext=(5, 4), textcoords='offset points',
            fontsize=7.5, fontfamily='monospace', color='#343a40'
        )

    ax.set_xlabel('Std Dev of PPDA  (lower = consistent pressing)',
                  fontsize=10, color='#343a40')
    ax.set_ylabel('Std Dev of Possession %  (lower = consistent possession)',
                  fontsize=10, color='#343a40')
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(alpha=0.2, linestyle=':', zorder=0)

    # Confederation legend
    handles = [
        mpatches.Patch(color=c, label=l, alpha=0.85)
        for c, l in CONF_LABELS.items()
    ]
    ax.legend(handles=handles, fontsize=8, frameon=False,
              title='Confederation', title_fontsize=8,
              loc='upper right')

    ax.set_title(
        'II.7 — Tactical Consistency: 2026 World Cup Teams\n'
        'Match-to-match volatility in pressing and possession identity',
        fontsize=11, fontweight='bold', fontfamily='monospace',
        loc='left', pad=14
    )

    plt.tight_layout()
    plt.savefig('figures/2_7_tactical_consistency.png',
                dpi=180, bbox_inches='tight', facecolor='#ffffff')
    plt.show()