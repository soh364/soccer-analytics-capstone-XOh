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

from IPython.display import display
import pandas as pd

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
    Display dataset summary as a clean, professional table.
    Simple styling, no fancy colors.
    """
    # Get summary stats
    summary_stats = plot_dataset_overview_summary(statsbomb_dir)
    
    # Create DataFrame
    data = {
        'Dataset': ['Matches', 'Events', 'Lineups', '360° Tracking'],
        'Records': [
            f"{summary_stats['matches']['total']:,}",
            f"{summary_stats['events']['total']:,}",
            f"{summary_stats['lineups']['total']:,}",
            f"{summary_stats['three60']['total']:,}"
        ],
        'Coverage': [
            f"{summary_stats['matches']['competitions']} competitions",
            f"{summary_stats['events']['types']} event types",
            f"{summary_stats['lineups']['players']:,} players",
            f"{summary_stats['three60']['matches']} matches"
        ],
        'Details': [
            f"~{summary_stats['events']['total'] // summary_stats['matches']['total']:,} events/match",
            f"{summary_stats['events']['has_location']/summary_stats['events']['total']*100:.2f}% with location",
            f"{summary_stats['lineups']['positions']} position types",
            f"{summary_stats['three60']['matches']/summary_stats['matches']['total']*100:.1f}% coverage"
        ]
    }
    
    df = pd.DataFrame(data)
    
    # Simple, clean styling
    styled = df.style.set_table_styles([
        # Header
        {'selector': 'thead th', 'props': [
            ('background-color', '#f8f9fa'),
            ('color', '#212529'),
            ('font-weight', '600'),
            ('text-align', 'left'),
            ('padding', '12px 15px'),
            ('border-bottom', '2px solid #dee2e6'),
            ('font-size', '13px')
        ]},
        # Body cells
        {'selector': 'tbody td', 'props': [
            ('padding', '12px 15px'),
            ('border-bottom', '1px solid #e9ecef'),
            ('font-size', '13px'),
            ('color', '#212529')
        ]},
        # First column (dataset names)
        {'selector': 'tbody td:first-child', 'props': [
            ('font-weight', '500')
        ]},
        # Alternating rows
        {'selector': 'tbody tr:nth-child(even)', 'props': [
            ('background-color', '#f8f9fa')
        ]},
        # Table
        {'selector': '', 'props': [
            ('border-collapse', 'collapse'),
            ('width', '100%'),
            ('margin', '10px 0'),
            ('border', '1px solid #dee2e6')
        ]}
    ]).hide(axis='index')
    
    display(styled)



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


def plot_competition_distribution(statsbomb_dir: Path, figsize=(14, 8)) -> None:
    """Competition distribution pie and bar charts."""
    lf = pl.scan_parquet(statsbomb_dir / "matches.parquet")
    
    comp_dist = lf.group_by("competition_name").agg(
        pl.len().alias("count")
    ).sort("count", descending=True).collect()
    
    top_10 = comp_dist.head(10)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Left: Pie chart (Top 5)
    top_5 = comp_dist.head(5)
    colors = plt.cm.Set3(np.linspace(0, 1, 5))
    
    ax1.pie(top_5['count'].to_list(), labels=top_5['competition_name'].to_list(),
           autopct='%1.1f%%', startangle=90, colors=colors)
    ax1.set_title('Top 5 Competitions', fontsize=13, fontweight='bold')
    
    # Right: Bar chart (Top 10)
    competitions = top_10['competition_name'].to_list()
    counts = top_10['count'].to_list()
    
    bars = ax2.barh(range(len(competitions)), counts, color='steelblue', edgecolor='black')
    ax2.set_yticks(range(len(competitions)))
    ax2.set_yticklabels(competitions, fontsize=10)
    ax2.set_xlabel('Number of Matches', fontsize=11)
    ax2.set_title('Top 10 Competitions', fontsize=13, fontweight='bold')
    ax2.invert_yaxis()
    ax2.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, count in enumerate(counts):
        ax2.text(count + 15, i, f'{count}', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.show()


def plot_temporal_coverage_stacked(statsbomb_dir: Path, figsize=(16, 7)) -> None:
    """Temporal distribution with club vs tournament stacked bars."""
    lf = pl.scan_parquet(statsbomb_dir / "matches.parquet")
    
    with_year = lf.with_columns(
        pl.col("match_date").str.strptime(pl.Date, "%Y-%m-%d").dt.year().alias("year")
    ).filter(
        pl.col("year") >= 2015
    ).with_columns(
        pl.when(
            (pl.col("competition_name").str.contains("World Cup")) |
            (pl.col("competition_name").str.contains("Euro")) |
            (pl.col("competition_name").str.contains("Copa"))
        ).then(pl.lit("Tournament"))
        .otherwise(pl.lit("Club"))
        .alias("category")
    )
    
    breakdown = with_year.group_by(["year", "category"]).agg(
        pl.len().alias("count")
    ).sort("year").collect()
    
    years = sorted(breakdown['year'].unique().to_list())
    club_counts = []
    tournament_counts = []
    
    for year in years:
        year_data = breakdown.filter(pl.col("year") == year)
        club = year_data.filter(pl.col("category") == "Club")
        tourn = year_data.filter(pl.col("category") == "Tournament")
        
        club_counts.append(club['count'].sum() if len(club) > 0 else 0)
        tournament_counts.append(tourn['count'].sum() if len(tourn) > 0 else 0)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    x = np.arange(len(years))
    width = 0.6
    
    bars1 = ax.bar(x, club_counts, width, label='Club', color='lightcoral', edgecolor='black')
    bars2 = ax.bar(x, tournament_counts, width, bottom=club_counts, label='Tournament',
                   color='steelblue', edgecolor='black')
    
    ax.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Matches', fontsize=12, fontweight='bold')
    ax.set_title('Temporal Distribution (2015-2025)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(years, fontsize=11)
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    
    # Add total labels
    for i, (club, tourn) in enumerate(zip(club_counts, tournament_counts)):
        total = club + tourn
        if total > 0:
            ax.text(i, total + 20, f'{total}', ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.show()


def plot_event_type_distribution(statsbomb_dir: Path, figsize=(12, 8)) -> None:
    """Event type distribution bar chart."""
    events_lf = pl.scan_parquet(statsbomb_dir / "events.parquet")
    total_events = events_lf.select(pl.len()).collect()[0, 0]
    
    event_dist = events_lf.group_by("type").agg(
        pl.len().alias("count")
    ).sort("count", descending=True).head(15).collect()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    event_types = event_dist['type'].to_list()
    counts = event_dist['count'].to_list()
    percentages = [(c / total_events) * 100 for c in counts]
    
    bars = ax.barh(range(len(event_types)), percentages, color='steelblue', edgecolor='black')
    
    ax.set_yticks(range(len(event_types)))
    ax.set_yticklabels(event_types, fontsize=10)
    ax.set_xlabel('Percentage of Total Events (%)', fontsize=11)
    ax.set_title('Event Type Distribution (Top 15)', fontsize=13, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    
    # Add labels
    for i, (pct, count) in enumerate(zip(percentages, counts)):
        ax.text(pct + 0.3, i, f'{pct:.1f}% ({count:,})', va='center', fontsize=8)
    
    plt.tight_layout()
    plt.show()


def plot_pass_completion_analysis(statsbomb_dir: Path, figsize=(14, 6)) -> None:
    """Pass completion and shot quality distributions."""
    events_lf = pl.scan_parquet(statsbomb_dir / "events.parquet")
    
    # Pass completion
    passes = events_lf.filter(pl.col("type") == "Pass")
    pass_stats = passes.select([
        pl.len().alias("total"),
        pl.col("pass_outcome").is_null().sum().alias("completed")
    ]).collect()
    
    completion_rate = (pass_stats['completed'][0] / pass_stats['total'][0]) * 100
    incomplete_rate = 100 - completion_rate
    
    # Shot xG
    shots = events_lf.filter(pl.col("type") == "Shot").filter(
        pl.col("shot_statsbomb_xg").is_not_null()
    )
    shot_xg = shots.select("shot_statsbomb_xg").collect()['shot_statsbomb_xg'].to_list()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Left: Pass completion pie
    ax1.pie([completion_rate, incomplete_rate], 
           labels=['Completed', 'Incomplete'],
           autopct='%1.1f%%', startangle=90,
           colors=['lightgreen', 'lightcoral'])
    ax1.set_title('Pass Completion Rate', fontsize=13, fontweight='bold')
    
    # Right: xG histogram
    ax2.hist(shot_xg, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    ax2.axvline(np.median(shot_xg), color='red', linestyle='--', linewidth=2,
               label=f'Median: {np.median(shot_xg):.3f}')
    ax2.set_xlabel('Expected Goals (xG)', fontsize=11)
    ax2.set_ylabel('Number of Shots', fontsize=11)
    ax2.set_title('Shot Quality Distribution', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()


def plot_player_participation(statsbomb_dir: Path, figsize=(14, 7)) -> None:
    """Player participation by position and match distribution."""
    lineups_lf = pl.scan_parquet(statsbomb_dir / "lineups.parquet")
    
    # Position distribution
    pos_dist = lineups_lf.filter(
        pl.col("position_name").is_not_null()
    ).group_by("position_name").agg(
        pl.len().alias("count")
    ).sort("count", descending=True).head(12).collect()
    
    # Matches per player
    matches_per_player = lineups_lf.group_by("player_id").agg(
        pl.len().alias("matches")
    ).collect()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Left: Position distribution
    positions = pos_dist['position_name'].to_list()
    counts = pos_dist['count'].to_list()
    
    bars = ax1.barh(range(len(positions)), counts, color='steelblue', edgecolor='black')
    ax1.set_yticks(range(len(positions)))
    ax1.set_yticklabels(positions, fontsize=10)
    ax1.set_xlabel('Number of Records', fontsize=11)
    ax1.set_title('Position Distribution (Top 12)', fontsize=13, fontweight='bold')
    ax1.invert_yaxis()
    ax1.grid(axis='x', alpha=0.3)
    
    # Right: Matches per player
    matches_list = matches_per_player['matches'].to_list()
    ax2.hist(matches_list, bins=50, color='steelblue', edgecolor='black', 
            alpha=0.7, range=(0, 200))
    ax2.axvline(np.median(matches_list), color='red', linestyle='--', linewidth=2,
               label=f'Median: {np.median(matches_list):.0f}')
    ax2.set_xlabel('Matches per Player', fontsize=11)
    ax2.set_ylabel('Number of Players (log scale)', fontsize=11)
    ax2.set_title('Player Participation Distribution', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3, axis='y')
    ax2.set_yscale('log')
    
    plt.tight_layout()
    plt.show()


def plot_360_coverage_analysis(statsbomb_dir: Path, figsize=(14, 6)) -> None:
    """360° tracking data coverage analysis."""
    matches_lf = pl.scan_parquet(statsbomb_dir / "matches.parquet")
    three60_lf = pl.scan_parquet(statsbomb_dir / "three_sixty.parquet")
    
    total_matches = matches_lf.select(pl.len()).collect()[0, 0]
    three60_matches = three60_lf.select(pl.col("match_id").n_unique()).collect()[0, 0]
    
    coverage_pct = (three60_matches / total_matches) * 100
    no_coverage_pct = 100 - coverage_pct
    
    # Get competitions with 360
    three60_match_ids = three60_lf.select("match_id").unique().collect()["match_id"].to_list()
    
    matches_with_360 = matches_lf.filter(
        pl.col("match_id").is_in(three60_match_ids)
    ).group_by("competition_name").agg(
        pl.len().alias("count")
    ).sort("count", descending=True).collect()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Left: Coverage pie
    ax1.pie([coverage_pct, no_coverage_pct],
           labels=[f'With 360° ({three60_matches})', 
                  f'Without 360° ({total_matches - three60_matches})'],
           autopct='%1.1f%%', startangle=90,
           colors=['lightgreen', 'lightcoral'])
    ax1.set_title('360° Data Coverage', fontsize=13, fontweight='bold')
    
    # Right: Competitions
    comps = matches_with_360['competition_name'].to_list()[:8]
    comp_counts = matches_with_360['count'].to_list()[:8]
    
    bars = ax2.barh(range(len(comps)), comp_counts, color='steelblue', edgecolor='black')
    ax2.set_yticks(range(len(comps)))
    ax2.set_yticklabels(comps, fontsize=10)
    ax2.set_xlabel('Matches with 360° Data', fontsize=11)
    ax2.set_title('Competitions with 360° Coverage', fontsize=13, fontweight='bold')
    ax2.invert_yaxis()
    ax2.grid(axis='x', alpha=0.3)
    
    # Add labels
    for i, count in enumerate(comp_counts):
        ax2.text(count + 1, i, f'{count}', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.show()

def plot_reference_breakdown(statsbomb_dir: Path, figsize=(10, 6)) -> None:
    """Reference dataset entity type breakdown."""
    lf = pl.scan_parquet(statsbomb_dir / "reference.parquet")
    
    entity_dist = lf.group_by("table_name").agg(
        pl.len().alias("count")
    ).sort("count", descending=True).collect()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    entities = entity_dist['table_name'].to_list()
    counts = entity_dist['count'].to_list()
    
    bars = ax.bar(range(len(entities)), counts, color='steelblue', edgecolor='black')
    
    ax.set_xticks(range(len(entities)))
    ax.set_xticklabels(entities, fontsize=11)
    ax.set_ylabel('Number of Records', fontsize=11)
    ax.set_title('Reference Data Entity Types', fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, (bar, count) in enumerate(zip(bars, counts)):
        ax.text(i, count + 100, f'{count}', ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.show()
    
    
### just to display things better

