"""
Supplementary EDA Functions for 2026 World Cup Prediction Framework
Extends eda_starter_template.py with prediction-focused analysis

Used by: EDA/EDA.ipynb
Focus: Validate dataset readiness for Team Archetype, Player Quality, System Fit analysis
"""

from __future__ import annotations
from pathlib import Path
from typing import Any
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from IPython.display import clear_output
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
from IPython.display import display
import pandas as pd
import matplotlib.gridspec as gridspec
import textwrap
from analysis.visualization import save_figure

# Set plotting style
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 10

# Constants
WIDTH = 80


def generate_summary_stats(statsbomb_dir: Path) -> dict[str, Any]:
    """Generate summary statistics."""
    matches_lf = pl.scan_parquet(statsbomb_dir / "matches.parquet")
    events_lf = pl.scan_parquet(statsbomb_dir / "events.parquet")
    lineups_lf = pl.scan_parquet(statsbomb_dir / "lineups.parquet")
    
    total_matches = matches_lf.select(pl.len()).collect()[0, 0]
    total_events = events_lf.select(pl.len()).collect()[0, 0]
    unique_players = lineups_lf.select(pl.col("player_id").n_unique()).collect()[0, 0]
    
    has_loc = events_lf.filter(pl.col("location_x").is_not_null()).select(pl.len()).collect()[0, 0]
    
    return {
        "total_matches": total_matches,
        "total_events": total_events,
        "unique_players": unique_players,
        "location_coverage_pct": has_loc / total_events * 100
    }

def overall_summary_table(statsbomb_dir: Path) -> None:
    """
    Display dataset summary as a clean, fixed-width ASCII table.
    """
    # 1. Get stats (this might print "scanning" messages)
    summary_stats = plot_dataset_overview_summary(statsbomb_dir)
    
    # 2. Wipe any noise from the helper function so ONLY the table shows
    clear_output(wait=True)
    
    header_text = "DATASET SUMMARY: STATSBOMB CORE ENGINE"
    cols = ["Dataset", "Records", "Coverage", "Details"]
    
    rows = [
        ["Matches", f"{summary_stats['matches']['total']:,}", f"{summary_stats['matches']['competitions']} competitions", f"~{summary_stats['events']['total'] // summary_stats['matches']['total']:,} events/match"],
        ["Events", f"{summary_stats['events']['total']:,}", f"{summary_stats['events']['types']} types", f"{summary_stats['events']['has_location']/summary_stats['events']['total']*100:.1f}% w/ loc"],
        ["Lineups", f"{summary_stats['lineups']['total']:,}", f"{summary_stats['lineups']['players']:,} players", f"{summary_stats['lineups']['positions']} positions"],
        ["360 Tracking", f"{summary_stats['three60']['total']:,}", f"{summary_stats['three60']['matches']} matches", f"{summary_stats['three60']['matches']/summary_stats['matches']['total']*100:.1f}% coverage"]
    ]

    print("=" * 90)
    print(f"{header_text:^90}")
    print("=" * 90)
    print(f"{cols[0]:<15} {cols[1]:<15} {cols[2]:<25} {cols[3]:<25}")
    print("-" * 90)
    
    for r in rows:
        print(f"{r[0]:<15} {r[1]:<15} {r[2]:<25} {r[3]:<25}")
    
    print("=" * 90)


def plot_dataset_overview_summary(statsbomb_dir: Path, figsize=(16, 6)) -> dict[str, Any]:
    """
    Create summary table showing all 4 datasets at a glance.
    Returns key statistics for display.
    """
    matches_lf = pl.scan_parquet(statsbomb_dir / "matches.parquet")
    events_lf = pl.scan_parquet(statsbomb_dir / "events.parquet")
    lineups_lf = pl.scan_parquet(statsbomb_dir / "lineups.parquet")
    three60_lf = pl.scan_parquet(statsbomb_dir / "three_sixty.parquet")
    
    # Gather statistics
    stats = {
        "matches": {
            "total": matches_lf.select(pl.len()).collect()[0, 0],
            "competitions": matches_lf.select(pl.col("competition_name").n_unique()).collect()[0, 0],
            "date_range": matches_lf.select([
                pl.col("match_date").min().alias("min"),
                pl.col("match_date").max().alias("max")
            ]).collect()
        },
        "events": {
            "total": events_lf.select(pl.len()).collect()[0, 0],
            "types": events_lf.select(pl.col("type").n_unique()).collect()[0, 0],
            "has_location": events_lf.filter(
                pl.col("location_x").is_not_null()
            ).select(pl.len()).collect()[0, 0]
        },
        "lineups": {
            "total": lineups_lf.select(pl.len()).collect()[0, 0],
            "players": lineups_lf.select(pl.col("player_id").n_unique()).collect()[0, 0],
            "positions": lineups_lf.select(pl.col("position_name").n_unique()).collect()[0, 0]
        },
        "three60": {
            "total": three60_lf.select(pl.len()).collect()[0, 0],
            "events": three60_lf.select(pl.col("event_uuid").n_unique()).collect()[0, 0],
            "matches": three60_lf.select(pl.col("match_id").n_unique()).collect()[0, 0]
        }
    }
    
    return stats

# Sophisticated palette — perceptually distinct, matched to known competition names
COMP_PALETTE = {
    'La Liga':                  '#e63946',
    'Ligue 1':                  '#f4a261',
    'Serie A':                  '#2a9d8f',
    'Premier League':           '#1d3557',
    'Bundesliga':               '#e9c46a',
    'FA Women':                 '#c77dff',
    'FIFA World Cup':           '#2d6a4f',
    'Women\'s World Cup':       '#457b9d',
    'Indian Super':             '#b5838d',
    'UEFA Euro':                '#6d6875',
}

FALLBACK_COLORS = ['#f4e285', '#a8dadc', '#ff6b6b', '#84a98c', '#cdb4db']


def gradient_barh(ax, y, width, height, cmap_name='Blues'):
    """Draw horizontal gradient bars — light to dark left to right."""
    cmap = plt.get_cmap(cmap_name)
    for this_y, this_width in zip(y, width):
        grad = np.atleast_2d(np.linspace(0.3, 0.9, 256))
        ax.imshow(
            grad,
            extent=[0, this_width, this_y - height / 2, this_y + height / 2],
            aspect='auto', cmap=cmap, zorder=3,
        )

def plot_matches_market_overview(statsbomb_dir: Path, figsize=(20, 7)) -> None:
    # ── Data ─────────────────────────────────────────────────────────────────
    lf = pl.scan_parquet(statsbomb_dir / "matches.parquet")
    df_matches = (
        lf.with_columns(
            pl.col("match_date").cast(pl.String)
            .str.strptime(pl.Date, "%Y-%m-%d")
            .dt.year().alias("year")
        )
        .filter(pl.col("year").is_between(2010, 2025))
        .collect()
    )

    # Left chart — Top 8 for the Ranking
    top_8_df = (
        df_matches.group_by("competition_name")
        .agg(pl.len().alias("match_count"))
        .sort("match_count", descending=True)
        .head(8)
    )

    # Right chart — Let's use Top 12 to ensure UEFA Euro/World Cups aren't hidden
    top_stack_df = (
        df_matches.group_by("competition_name")
        .agg(pl.len().alias("match_count"))
        .sort("match_count", descending=True)
        .head(12) 
    )
    
    top_stack_names = top_stack_df["competition_name"].to_list()
    all_years = list(range(2010, 2026))

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=figsize)
    # Squeezed first ratio to 0.4 to shrink the X-axis footprint
    gs = gridspec.GridSpec(1, 2, width_ratios=[0.4, 1.6])
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    # ── Left: Shrunken Bars & Large Names ────────────────────────────────────
    names  = top_8_df["competition_name"].to_list()
    counts = top_8_df["match_count"].to_list()
    y_pos  = np.arange(len(names))

    bar_colors = [plt.get_cmap('Blues')(0.3 + 0.6 * (1 - i / len(counts))) for i in range(len(counts))]
    ax1.barh(y_pos, counts, height=0.7, color=bar_colors, zorder=3)

    for y, c in zip(y_pos, counts):
        ax1.text(c + max(counts) * 0.05, y, str(c), va='center', fontsize=11, fontweight='bold')

    # FIX: Increased font size to 14 and kept wrapped
    wrapped_names = [textwrap.fill(name, 12) for name in names] 
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(wrapped_names, fontsize=10, fontweight='400', ha='right')

    ax1.invert_yaxis()
    ax1.set_title('Top 8 Competition by Volume', fontsize=12, fontweight='bold', pad=12)
    
    # FIX: Shrink the x-axis footprint by making the limit much larger than the bars
    ax1.set_xlim(0, max(counts) * 1.8) 
    ax1.spines[['top', 'right', 'left', 'bottom']].set_visible(False)
    ax1.set_xticks([]) 
    ax1.tick_params(axis='y', length=0)

    # ── Right: Stacked Bars (Ensuring Euro is visible) ──────────────────────
    bottoms = np.zeros(len(all_years))

    for i, name in enumerate(top_stack_names):
        color = next(
            (v for k, v in COMP_PALETTE.items() if k.lower() in name.lower()),
            FALLBACK_COLORS[i % len(FALLBACK_COLORS)],
        )
        yearly_counts = [
            df_matches.filter((pl.col("year") == y) & (pl.col("competition_name") == name)).height
            for y in all_years
        ]
        ax2.bar(all_years, yearly_counts, bottom=bottoms, label=name[:30], color=color,
                width=0.8, edgecolor='white', linewidth=0.5)
        bottoms += np.array(yearly_counts)

    # Others Category
    other_counts = [
        df_matches.filter((pl.col("year") == y) & (~pl.col("competition_name").is_in(top_stack_names))).height
        for y in all_years
    ]
    ax2.bar(all_years, other_counts, bottom=bottoms, label='Others', color='#dee2e6', width=0.8)

    # Totals on top
    for year_idx, total in enumerate(bottoms + np.array(other_counts)):
        if total > 0:
            ax2.text(all_years[year_idx], total + 1, str(int(total)),
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax2.set_title('Annual Competition Mix (2010–2025)', fontsize=12, fontweight='bold', pad=12)
    ax2.set_xticks(all_years)
    ax2.set_xticklabels(all_years, rotation=45, fontsize=10)
    ax2.legend(loc='upper right', bbox_to_anchor=(0.99, 0.99), fontsize=9, frameon=True, facecolor='white', framealpha=0.8)
    ax2.spines[['top', 'right']].set_visible(False)

    # ── FINAL SPACING ADJUSTMENT (The Gap Killer) ──
    # left=0.22 gives room for the big labels, wspace=0.05 snaps the charts together
    plt.subplots_adjust(left=0.22, right=0.97, wspace=0.05, top=0.85, bottom=0.15)
    
    plt.suptitle('Match Data: Competition Coverage & Temporal Scope', fontsize=14, fontweight='bold', y=0.96)
    
    # REMOVED tight_layout to preserve subplots_adjust
    save_figure(fig, 'matches_overview.png', dpi=180)
    plt.show()


def plot_player_versatility_gradient(statsbomb_dir: Path, player_name: str):
    # 1. The "Nerdy" Tiered Palette
    POSITION_COLOURS = {
        # --- THE DEFENSIVE BLOCK (Deep Forest Greens) ---
        'Center Back': '#132a13', 'Left Center Back': '#31572c', 'Right Center Back': '#4f772d',
        'Left Back': '#90a955', 'Right Back': '#ecf39e', 
        'Left Wing Back': '#95d5b2', 'Right Wing Back': '#b7e4c7', 
        'Goalkeeper': '#6c757d',

        # --- THE CONTROLLERS / '6' (Deep Midnight Purples) ---
        # Darker and more "Purple-Navy" to separate from the blues
        'Center Defensive Midfield': '#240046', 
        'Left Defensive Midfield': '#3c096c', 
        'Right Defensive Midfield': '#5a189a',

        # --- THE ENGINES / '8' (Vibrant Cyan/Electric Blues) ---
        # Shifted away from purple tones toward "Bright Water" tones
        'Center Midfield': '#0077b6', 
        'Left Center Midfield': '#00b4d8', 
        'Right Center Midfield': '#48cae4',
        'Left Midfield': '#90e0ef', 
        'Right Midfield': '#ade8f4',

        # --- THE CREATORS / '10' (Electric Magentas) ---
        'Center Attacking Midfield': '#ff006e', 
        'Left Attacking Midfield': '#ff70a6', 
        'Right Attacking Midfield': '#ff97b7',

        # --- THE FINISHERS (Hot Oranges/Reds) ---
        'Center Forward': '#6a040f', 'Left Center Forward': '#9d0208', 'Right Center Forward': '#d00000',
        'Secondary Striker': '#dc2f02', 'Left Wing': '#e85d04', 'Right Wing': '#f48c06',
        
        'nan': '#adb5bd'
    }

    # 2. Data Loading (read_parquet for DataFrame)
    lineups = pl.read_parquet(statsbomb_dir / "lineups.parquet")
    matches = pl.read_parquet(statsbomb_dir / "matches.parquet")

    df = lineups.join(matches, on="match_id").filter(
        pl.col("player_name").str.to_lowercase().str.contains(player_name.lower())
    )

    if df.is_empty():
        return print(f"No data for {player_name}")

    def parse_time(s):
        try:
            p = s.split(':')
            return float(p[0]) + float(p[1])/60.0
        except: return 90.0

    # 3. Processing
    res = (
        df.with_columns([
            pl.col("from_time").map_elements(parse_time, return_dtype=pl.Float64).alias("start"),
            pl.col("to_time").map_elements(parse_time, return_dtype=pl.Float64).alias("end"),
            pl.col("position_name").cast(pl.Utf8).str.strip_chars().fill_null("nan")
        ])
        .with_columns((pl.col("end") - pl.col("start")).alias("mins"))
        .group_by(["season_name", "position_name"])
        .agg(pl.col("mins").sum())
        .pivot(index="season_name", on="position_name", values="mins")
        .fill_null(0)
        .sort("season_name")
    )

    # 4. Plotting
    seasons = res["season_name"].to_list()
    # Only include positions that have > 0 minutes total
    positions = [pos for pos in res.columns if pos != "season_name" and res[pos].sum() > 0]
    
    # Sort positions based on their order in POSITION_COLOURS (Defense to Attack)
    # This makes the legend look logically ordered
    order = list(POSITION_COLOURS.keys())
    positions.sort(key=lambda x: order.index(x) if x in order else 999)

    fig, ax = plt.subplots(figsize=(10, 7))
    bottom = np.zeros(len(seasons))
    
    for pos in positions:
        vals = np.array(res[pos].to_list())
        color = POSITION_COLOURS.get(pos, '#333333') 
        ax.bar(seasons, vals, bottom=bottom, label=pos, color=color, edgecolor='white', alpha=0.9, width=0.7)
        bottom += vals

    # 5. Presentation Styling
    ax.set_ylim(0, max(bottom) * 1.1) 
    ax.set_title(f"Tactical Profile Evolution: {player_name}", fontsize=14, fontweight='bold', loc='center', pad=25)
    ax.set_ylabel("Minutes Played", fontsize=12)
    
    plt.xticks(rotation=45, ha='right')
    
    # CLEAN LEGEND: Only shows roles played, organized by tier
    ax.legend(title="Tactical Roles (Played)", bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False, fontsize=9)
    
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    save_figure(fig, 'profile_evol_example.png', dpi=180)
    plt.show()

def plot_360_coverage_analysis(statsbomb_dir: Path, figsize=(10, 4.5)) -> None:
    """Efficiently loads and plots 360° tracking data coverage."""
    
    # --- EFFICIENT DATA LOADING ---
    # scan_parquet lets Polars plan the query before loading anything
    matches_lf = pl.scan_parquet(statsbomb_dir / "matches.parquet")
    three60_lf = pl.scan_parquet(statsbomb_dir / "three_sixty.parquet")
    
    # Only load match_id to count unique 360 matches (avoids loading massive tracking coords)
    three60_match_ids = (
        three60_lf.select("match_id")
        .unique()
        .collect()
        .get_column("match_id")
        .to_list()
    )
    
    # Filter matches and select only necessary columns for the final aggregation
    matches_summary = (
        matches_lf.select(["match_id", "competition_name"])
        .collect()
    )
    
    total_matches = len(matches_summary)
    three60_count = len(three60_match_ids)
    
    # Aggregation for the bottom chart
    matches_with_360 = (
        matches_summary.filter(pl.col("match_id").is_in(three60_match_ids))
        .group_by("competition_name")
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
        .head(6)
    )
    
    # --- STYLING & COLORS ---
    TOP_RED = '#e63946'    # High-impact red for summary
    BOTTOM_GREEN = '#2a9d8f' # Professional green for breakdown
    TEXT_DARK = '#264653'
    TRACK_GREY = '#edf2f4' # Clean background track
    
    fig = plt.figure(figsize=figsize, facecolor='white')
    gs = fig.add_gridspec(2, 1, height_ratios=[0.8, 4])
    
# --- TOP: ANCHORED COVERAGE STATUS ---
    ax0 = fig.add_subplot(gs[0])
    
    # Adding a label 'TOTAL SAMPLE'
    ax0.barh(["TOTAL SAMPLE"], [total_matches], color=TRACK_GREY, height=0.5)
    ax0.barh(["TOTAL SAMPLE"], [three60_count], color=TOP_RED, height=0.5)
    
    # Title
    ax0.set_title(f"360° Data Coverage", 
                 loc='center', fontsize=14, fontweight='bold', color=TEXT_DARK, pad=10)
    
    # Clean up axes
    ax0.spines[['top', 'right', 'bottom', 'left']].set_visible(False)
    ax0.set_xticks([])
    
    # Correct way to set label properties (color and bold)
    ax0.tick_params(axis='y', length=0, labelsize=9, labelcolor=TOP_RED)
    plt.setp(ax0.get_yticklabels(), fontweight='bold') # This bolds the Y-axis label
    
    # Percentage label - placed at the end of the red bar
    pct = (three60_count / total_matches) * 100
    ax0.text(three60_count + (total_matches * 0.02), 0, f"{three60_count} Matches ({pct:.1f}%)", 
             va='center', ha='left', color=TOP_RED, fontweight='bold', fontsize=9)
             
    # --- BOTTOM: COMPETITION BREAKDOWN ---
    ax1 = fig.add_subplot(gs[1])
    comps = matches_with_360['competition_name'].to_list()
    counts = matches_with_360['count'].to_list()
    
    bars = ax1.barh(comps, counts, color=BOTTOM_GREEN, alpha=0.9, height=0.7)
    
    ax1.spines[['top', 'right', 'bottom', 'left']].set_visible(False)
    ax1.tick_params(axis='both', which='both', length=0)
    ax1.set_yticklabels(comps, fontsize=9.5, color=TEXT_DARK)
    ax1.set_xticks([])
    ax1.invert_yaxis()
    
    for bar in bars:
        width = bar.get_width()
        ax1.text(width + 2, bar.get_y() + bar.get_height()/2, f'{int(width)}', 
                 va='center', fontsize=9.5, color=TEXT_DARK, fontweight='bold')

    plt.tight_layout()
    save_figure(fig, '360_coverage.png', dpi=180)
    plt.show()


def plot_360_player_heatmap(statsbomb_dir: Path, figsize=(12, 7)) -> None:
    """
    Create a heatmap showing player positions from 360° tracking data.
    Uses one match as an example to show the richness of tracking data.
    """
    
    # Load 360 data
    three60_lf = pl.scan_parquet(statsbomb_dir / "three_sixty.parquet")
    
    # Get a match with good 360 coverage
    match_counts = three60_lf.group_by("match_id").agg(
        pl.len().alias("count")
    ).sort("count", descending=True).collect()
    
    # Use the match with most tracking data
    sample_match_id = match_counts['match_id'][0]
    
    # Get data for this match
    match_data = three60_lf.filter(
        pl.col("match_id") == sample_match_id
    ).collect()
    
    print(f"Visualizing match ID: {sample_match_id}")
    print(f"Tracking points: {len(match_data):,}")
    
    # Extract player positions (visible players around the event)
    all_x = []
    all_y = []
    
    for row in match_data.iter_rows(named=True):
        x = row.get('location_x')
        y = row.get('location_y')
        if x is not None and y is not None:
            all_x.append(x)
            all_y.append(y)
    
    # Create figure with pitch
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # --- Left: Heatmap ---
    # Create 2D histogram (heatmap)
    heatmap, xedges, yedges = np.histogram2d(
        all_x, all_y, 
        bins=[40, 30],
        range=[[0, 120], [0, 80]]
    )
    
    # Plot heatmap
    extent = [0, 120, 0, 80]
    
    # Custom colormap
    colors = ['#f7fbff', '#6baed6', '#2171b5', '#e74c3c']
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list('custom', colors, N=n_bins)
    
    im = ax1.imshow(
        heatmap.T, 
        extent=extent, 
        origin='lower',
        cmap=cmap,
        aspect='auto',
        interpolation='gaussian'
    )
    
    # Draw pitch lines
    draw_pitch(ax1, patches)
    
    ax1.set_xlim(0, 120)
    ax1.set_ylim(0, 80)
    ax1.set_xlabel('Pitch Length (yards)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Pitch Width (yards)', fontsize=11, fontweight='bold')
    ax1.set_title('360° Tracking Data: Player Position Heatmap', 
                 fontsize=14, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    cbar.set_label('Activity Density', rotation=270, labelpad=20, fontweight='bold')
    
    # --- Right: Scatter plot ---
    ax2.scatter(all_x, all_y, alpha=0.3, s=1, c='steelblue')
    
    # Draw pitch lines
    draw_pitch(ax2, patches)
    
    ax2.set_xlim(0, 120)
    ax2.set_ylim(0, 80)
    ax2.set_xlabel('Pitch Length (yards)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Pitch Width (yards)', fontsize=11, fontweight='bold')
    ax2.set_title('360° Tracking Data: All Position Points', 
                 fontsize=14, fontweight='bold')
    ax2.text(60, -10, f'{len(all_x):,} tracking points', 
            ha='center', fontsize=10, style='italic')
    
    plt.tight_layout()
    save_figure(fig, '360_heatmap.png', dpi=180)
    plt.show()


def draw_pitch(ax, patches):
    """Draw football pitch lines on axis"""
    # Pitch outline
    ax.add_patch(patches.Rectangle((0, 0), 120, 80, 
                                   linewidth=2, edgecolor='white', 
                                   facecolor='none'))
    
    # Halfway line
    ax.plot([60, 60], [0, 80], color='white', linewidth=2)
    
    # Center circle
    center_circle = patches.Circle((60, 40), 10, 
                                   linewidth=2, edgecolor='white', 
                                   facecolor='none')
    ax.add_patch(center_circle)
    
    # Penalty boxes
    ax.add_patch(patches.Rectangle((0, 18), 18, 44, 
                                   linewidth=2, edgecolor='white', 
                                   facecolor='none'))
    ax.add_patch(patches.Rectangle((102, 18), 18, 44, 
                                   linewidth=2, edgecolor='white', 
                                   facecolor='none'))
    
    # Goal boxes
    ax.add_patch(patches.Rectangle((0, 30), 6, 20, 
                                   linewidth=2, edgecolor='white', 
                                   facecolor='none'))
    ax.add_patch(patches.Rectangle((114, 30), 6, 20, 
                                   linewidth=2, edgecolor='white', 
                                   facecolor='none'))
    
    # Set green pitch background
    ax.set_facecolor('#2d5f3a')


def plot_event_type_distribution(statsbomb_dir: Path, figsize=(8, 4)) -> None:
    
    # 1. High-speed Lazy Load (only reads 'type' column)
    events_lf = pl.scan_parquet(statsbomb_dir / "events.parquet").select("type")
    
    total_count = events_lf.select(pl.len()).collect().item()
    dist = events_lf.group_by("type").agg(pl.len().alias("n")).sort("n", descending=True).head(10).collect()

    # 2. Setup Gradient Colormap based on #2a9d8f
    # Creates a smooth transition from dark teal to your color
    teal_map = LinearSegmentedColormap.from_list("custom_teal", ["#1a635a", "#2a9d8f", "#7ec4bb"])
    colors = teal_map([i/10 for i in range(10)])

    # 3. Plotting
    fig, ax = plt.subplots(figsize=figsize, facecolor='white')
    
    labels = dist["type"].to_list()
    counts = dist["n"].to_list()
    pcts = [(c / total_count) * 100 for c in counts]
    
    # edgecolor='none' removes the borders around the bars
    ax.barh(range(len(labels)), pcts, color=colors, edgecolor='none', alpha=0.9, height=0.75)
    
    # 4. Styling (Minimalist)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=10, color='#333333')
    
    # Unbolded axis titles as requested
    ax.set_xlabel('Percentage of total dataset (%)', fontsize=10, fontweight='normal', color='#555555')
    ax.set_title(f'Event Distribution | Total: {total_count:,} events', fontsize=14, fontweight='bold', loc='center', pad=20)
    
    ax.invert_yaxis()
    ax.grid(axis='x', linestyle=':', alpha=0.4)
    
    # Remove top, right, and bottom spines for a floating look
    for spine in ['top', 'right', 'bottom']:
        ax.spines[spine].set_visible(False)
    ax.spines['left'].set_color('#dddddd')

    # 5. Data Labels (Percentage + Count)
    for i, (p, c) in enumerate(zip(pcts, counts)):
        ax.text(
            p + 0.4, i, 
            f'{p:.1f}% ({c:,})', 
            va='center', 
            fontsize=9, 
            color='#2a9d8f', # Matches your theme color
            fontweight='medium'
        )

    plt.tight_layout()
    save_figure(fig, 'event_type_dis.png', dpi=180)
    plt.show()
   

def plot_tactical_evolution_basics(statsbomb_dir: Path, figsize=(14, 8)) -> None:
    """
    Optimized tactical evolution chart with every year labeled.
    """

    # ── Optimized Data Loading ───────────────────────────────────────────────
    events_lf = pl.scan_parquet(statsbomb_dir / "events.parquet").select(["match_id", "type"])
    matches_lf = pl.scan_parquet(statsbomb_dir / "matches.parquet").select(["match_id", "match_date"])

    year_map = matches_lf.with_columns(
        pl.col("match_date").str.to_date().dt.year().alias("year")
    ).select(["match_id", "year"])

    stats = (
        events_lf.join(year_map, on="match_id")
        .filter(pl.col("year") >= 2010)
        .group_by("year")
        .agg([
            pl.col("match_id").n_unique().alias("matches"),
            (pl.col("type") == "Pressure").sum().alias("pressures"),
            (pl.col("type") == "Carry").sum().alias("carries"),
            (pl.col("type") == "Shot").sum().alias("shots"),
            (pl.col("type") == "Pass").sum().alias("passes"),
        ])
        .sort("year")
        .collect()
    )

    plot_df = stats.with_columns([
        (pl.col("pressures") / pl.col("matches")).alias("Intensity (Pressures)"),
        (pl.col("carries")   / pl.col("matches")).alias("Verticality (Carries)"),
        (pl.col("shots")     / pl.col("matches")).alias("Tactical Discipline (Shots)"),
        (pl.col("passes")    / pl.col("matches")).alias("Style (Passes)"),
    ]).to_pandas()

    metric_cols = ["Intensity (Pressures)", "Verticality (Carries)",
                   "Tactical Discipline (Shots)", "Style (Passes)"]
    for col in metric_cols:
        mn, mx = plot_df[col].min(), plot_df[col].max()
        plot_df[col] = (plot_df[col] - mn) / (mx - mn) if mx != mn else 0

    # ── Figure ───────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=figsize, sharex=True,
        gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.1},
    )
    fig.patch.set_facecolor('#ffffff')

    styles = [
        ('Intensity (Pressures)',       '#1d3557', 'o', '-',  1.8),
        ('Verticality (Carries)',        '#f4a261', 's', '-',  1.8),
        ('Tactical Discipline (Shots)', '#2a9d8f', '^', '-',  1.8),
        ('Style (Passes)',              '#e63946', 'x', '--', 1.5),
    ]

    for col, color, marker, ls, lw in styles:
        ax1.plot(plot_df["year"], plot_df[col], label=col, color=color, 
                 marker=marker, linestyle=ls, linewidth=lw, markersize=4)

    # Shaded regions
    ax1.axvspan(2015, 2016.5, color='#e63946', alpha=0.06)
    ax1.axvspan(2021, 2025.2, color='#2a9d8f', alpha=0.06)
    
    # Lowered Labels (Y=0.85 instead of 1.05)
    ax1.text(2015.7, 0.85, "Legacy Spike", color='#e63946', ha='center', fontsize=9, fontweight='bold')
    ax1.text(2023, 0.85, "Modern Window", color='#2a9d8f', ha='center', fontsize=9, fontweight='bold')

    ax1.set_ylabel('Normalised Score (0–1)', fontsize=10)
    
    # Move legend to bottom left
    ax1.legend(fontsize=8, loc='lower left', ncol=2, frameon=True, facecolor='white', framealpha=0.8)
    
    ax1.spines[['top', 'right']].set_visible(False)
    ax1.grid(axis='y', linestyle=':', alpha=0.4)
    ax1.set_title('Tactical Evolution of the Game (2010–2025)', fontsize=14, fontweight='bold', pad=20)

    # ── Bottom: Bar Chart ────────────────────────────────────────────────────
    bar_colors = ['#e63946' if 2015 <= y <= 2016 else '#2a9d8f' if y >= 2021 else '#adb5bd' for y in plot_df["year"]]
    ax2.bar(plot_df["year"], plot_df["matches"], color=bar_colors, width=0.7, alpha=0.8)
    
    ax2.set_ylabel('Matches', fontsize=9)
    ax2.set_xlabel('Year', fontsize=10)
    
    # SHOW EVERY YEAR ON X-AXIS
    all_years = plot_df["year"].unique()
    ax2.set_xticks(all_years)
    ax2.set_xticklabels(all_years, rotation=45, fontsize=9)
    
    ax2.spines[['top', 'right']].set_visible(False)

    for i, val in enumerate(plot_df["matches"]):
        if val > 5: ax2.text(plot_df["year"][i], val + 2, str(int(val)), ha='center', fontsize=7, color='#444444')

    plt.tight_layout()
    save_figure(fig, 'tactical_evolution.png', dpi=180)
    plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_tactical_lifespan_from_path(lineup_csv_path):
    # Load the data inside the function
    df = pd.read_csv(lineup_csv_path) 
    
    def time_to_float(time_str):
        if pd.isna(time_str) or time_str == 'NaN':
            return 95.0
        try:
            m, s = map(int, str(time_str).split(':'))
            return m + s/60.0
        except:
            return 95.0

    plot_df = df.copy() # This will now work because 'df' is a DataFrame
    plot_df['minutes_played'] = plot_df['to_time'].apply(time_to_float)

    # 2. Calculate average duration per position
    # We sort it to see the "High Turnover" positions at the top
    pos_rank = plot_df.groupby('position_name')['minutes_played'].mean().sort_values().reset_index()

    # 3. Visualization
    plt.figure(figsize=(11, 7))
    sns.set_style("whitegrid")
    
    # Using a diverging palette to highlight the difference between 'Anchors' and 'Engines'
    ax = sns.barplot(
        data=pos_rank, 
        x='minutes_played', 
        y='position_name', 
        palette="mako"
    )

    # Tactical Markers
    plt.axvline(x=65, color='#e67e22', linestyle='--', alpha=0.7, label='Intensity Refresh Window (Subs)')
    plt.axvline(x=90, color='#c0392b', linestyle='-', alpha=0.9, label='Full Match Duration')

    # Aesthetics
    plt.title('Tactical Lifespan: Average Minutes by Starting Position', fontsize=15, fontweight='bold', pad=20)
    plt.xlabel('Average Minutes on Pitch', fontsize=12)
    plt.ylabel('')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xlim(0, 105)
    
    plt.tight_layout()
    plt.show()

# To run this, simply pass your dataframe:
# plot_tactical_lifespan(your_lineup_df)