"""
Supplementary EDA Functions for Soccer Analytics Capstone
Extends eda_starter_template.py with project-specific analysis

Used by: EDA/EDA.ipynb
"""

from __future__ import annotations
from pathlib import Path
from typing import Any
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns

# Set plotting style
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 100

# Constants
WIDTH = 80

# ============================================================================
# WOMEN'S FOOTBALL ANALYSIS (Project Focus)
# ============================================================================

def analyze_womens_focus(statsbomb_dir: Path) -> dict[str, Any]:
    """
    Detailed women's football analysis.
    Returns stats for FA WSL, Euro, World Cup.
    """
    print("\n" + "=" * WIDTH)
    print("  WOMEN'S FOOTBALL - PROJECT FOCUS (504 matches)")
    print("=" * WIDTH)
    
    lf = pl.scan_parquet(statsbomb_dir / "matches.parquet")
    womens = lf.filter(pl.col("competition_name").str.contains("Women"))
    
    # Overall stats
    total = womens.select(pl.len()).collect()[0, 0]
    date_range = womens.select([
        pl.col("match_date").min().alias("min"),
        pl.col("match_date").max().alias("max")
    ]).collect()
    
    goals = womens.select(
        (pl.col("home_score") + pl.col("away_score")).mean().alias("avg_goals")
    ).collect()
    
    print(f"\nTotal Women's Matches: {total:,}")
    print(f"Date Range: {date_range['min'][0]} to {date_range['max'][0]}")
    print(f"Avg Goals/Match: {goals['avg_goals'][0]:.2f}")
    
    # Focus competitions
    print("\n" + "-" * WIDTH)
    print("FOCUS COMPETITIONS FOR ANALYSIS:")
    print("-" * WIDTH)
    
    fa_wsl = womens.filter(pl.col("competition_name") == "FA Women's Super League")
    euro = womens.filter(pl.col("competition_name").str.contains("Euro"))
    wc = womens.filter(pl.col("competition_name").str.contains("World Cup"))
    
    fa_wsl_count = fa_wsl.select(pl.len()).collect()[0, 0]
    euro_count = euro.select(pl.len()).collect()[0, 0]
    wc_count = wc.select(pl.len()).collect()[0, 0]
    
    print(f"\nFA Women's Super League:  {fa_wsl_count:3d} matches (Club Baseline)")
    print(f"UEFA Women's Euro:        {euro_count:3d} matches (Tournament)")
    print(f"FIFA Women's World Cup:   {wc_count:3d} matches (Tournament)")
    print("―" * WIDTH)
    print(f"Total for Analysis:       {fa_wsl_count + euro_count + wc_count:3d} matches")
    
    # Year distribution
    print("\n" + "-" * WIDTH)
    print("TEMPORAL DISTRIBUTION:")
    print("-" * WIDTH)
    
    years = womens.with_columns(
        pl.col("match_date").str.strptime(pl.Date, "%Y-%m-%d").dt.year().alias("year")
    ).group_by("year").agg(pl.len().alias("count")).sort("year").collect()
    
    print("\nWomen's Matches by Year:")
    for row in years.iter_rows(named=True):
        print(f"  {row['year']}: {row['count']:3d} matches")
    
    return {
        "total": total,
        "fa_wsl": fa_wsl_count,
        "euro": euro_count,
        "wc": wc_count,
        "focus_total": fa_wsl_count + euro_count + wc_count,
        "avg_goals": float(goals['avg_goals'][0])
    }

def analyze_mens_football(statsbomb_dir: Path) -> dict[str, Any]:
    """
    Analyze men's football dataset (comparative context).
    Shows why we chose women's football for our analysis.
    """
    print("\n" + "=" * WIDTH)
    print("  MEN'S FOOTBALL ANALYSIS (Comparative Context)")
    print("=" * WIDTH)
    
    lf = pl.scan_parquet(statsbomb_dir / "matches.parquet")
    mens = lf.filter(~pl.col("competition_name").str.contains("Women"))
    
    # Overall stats
    total = mens.select(pl.len()).collect()[0, 0]
    date_range = mens.select([
        pl.col("match_date").min().alias("min"),
        pl.col("match_date").max().alias("max")
    ]).collect()
    
    goals = mens.select(
        (pl.col("home_score") + pl.col("away_score")).mean().alias("avg_goals")
    ).collect()
    
    print(f"\nTotal Men's Matches: {total:,}")
    print(f"Date Range: {date_range['min'][0]} to {date_range['max'][0]}")
    print(f"Avg Goals/Match: {goals['avg_goals'][0]:.2f}")
    
    # Top competitions
    print("\n" + "-" * WIDTH)
    print("TOP COMPETITIONS:")
    print("-" * WIDTH)
    
    top_comps = mens.group_by("competition_name").agg(
        pl.len().alias("count")
    ).sort("count", descending=True).head(10).collect()
    
    print("\nTop 10 Men's Competitions:")
    for row in top_comps.iter_rows(named=True):
        pct = row['count'] / total * 100
        print(f"  {row['competition_name']:40s}: {row['count']:4d} ({pct:5.2f}%)")
    
    # Temporal concentration in men's data
    print("\n" + "-" * WIDTH)
    print("TEMPORAL DISTRIBUTION:")
    print("-" * WIDTH)
    
    with_year = mens.with_columns(
        pl.col("match_date").str.strptime(pl.Date, "%Y-%m-%d").dt.year().alias("year")
    )
    
    # Count 2015-2016 concentration
    period_2015_2016 = with_year.filter(
        pl.col("year").is_in([2015, 2016])
    ).select(pl.len()).collect()[0, 0]
    
    print(f"\n2015-2016 Period: {period_2015_2016:,} matches ({period_2015_2016/total*100:.1f}%)")
    print(" Men's data heavily concentrated in 2015-2016")
    
    # Year distribution (recent years)
    recent_years = with_year.filter(pl.col("year") >= 2015).group_by("year").agg(
        pl.len().alias("count")
    ).sort("year").collect()
    
    print("\nMen's Matches by Year (2015+):")
    for row in recent_years.iter_rows(named=True):
        print(f"  {row['year']}: {row['count']:4d} matches")
    
    return {
        "total": total,
        "date_min": str(date_range['min'][0]),
        "date_max": str(date_range['max'][0]),
        "avg_goals": float(goals['avg_goals'][0]),
        "period_2015_2016": period_2015_2016,
        "pct_2015_2016": period_2015_2016 / total * 100
    }

def analyze_temporal_bias(statsbomb_dir: Path) -> dict[str, Any]:
    """
    Analyze 2015-2016 temporal concentration issue.
    """
    print("\n" + "=" * WIDTH)
    print("  TEMPORAL CONCENTRATION ANALYSIS (2015-2016 Bias)")
    print("=" * WIDTH)
    
    lf = pl.scan_parquet(statsbomb_dir / "matches.parquet")
    
    with_year = lf.with_columns(
        pl.col("match_date").str.strptime(pl.Date, "%Y-%m-%d").dt.year().alias("year")
    )
    
    total = lf.select(pl.len()).collect()[0, 0]
    
    # Count periods
    period_2015_2016 = with_year.filter(
        pl.col("year").is_in([2015, 2016])
    ).select(pl.len()).collect()[0, 0]
    
    pre_2015 = with_year.filter(pl.col("year") < 2015).select(pl.len()).collect()[0, 0]
    post_2016 = with_year.filter(pl.col("year") > 2016).select(pl.len()).collect()[0, 0]
    
    pct_2015_2016 = period_2015_2016 / total * 100
    
    print(f"\nFULL DATASET TEMPORAL BREAKDOWN:")
    print(f"  2015-2016 Period:  {period_2015_2016:,} matches ({pct_2015_2016:.1f}%)")
    print(f"  Pre-2015:          {pre_2015:,} matches ({pre_2015/total*100:.1f}%)")
    print(f"  Post-2016:         {post_2016:,} matches ({post_2016/total*100:.1f}%)")


def analyze_data_quality(statsbomb_dir: Path) -> dict[str, Any]:
    """
    Comprehensive data quality assessment for metric calculation.
    """
    print("\n" + "=" * WIDTH)
    print("  DATA QUALITY ASSESSMENT (For 12-Metric Framework)")
    print("=" * WIDTH)
    
    events_lf = pl.scan_parquet(statsbomb_dir / "events.parquet")
    
    total_events = events_lf.select(pl.len()).collect()[0, 0]
    
    # Location coverage (critical for spatial metrics)
    print("\nSPATIAL DATA COVERAGE:")
    has_location = events_lf.filter(
        pl.col("location_x").is_not_null() & pl.col("location_y").is_not_null()
    ).select(pl.len()).collect()[0, 0]
    
    location_pct = has_location / total_events * 100
    print(f"  Events with location: {has_location:,} / {total_events:,} ({location_pct:.2f}%)")
    
    # Pressure context (for PPDA and defensive metrics)
    print("\nCONTEXT FLAGS:")
    under_pressure = events_lf.filter(pl.col("under_pressure") == True).select(pl.len()).collect()[0, 0]
    print(f"  Under pressure: {under_pressure:,} ({under_pressure/total_events*100:.2f}%)")
    
    # Duplicate check
    print("\nDATA INTEGRITY:")
    unique_ids = events_lf.select(pl.col("id").n_unique()).collect()[0, 0]
    duplicates = total_events - unique_ids
    print(f"  Duplicate event IDs: {duplicates}")
    
    # Shot quality (xG availability)
    print("\nSHOT QUALITY METRICS:")
    shots = events_lf.filter(pl.col("type") == "Shot")
    shot_count = shots.select(pl.len()).collect()[0, 0]
    
    xg_stats = shots.select([
        pl.col("shot_statsbomb_xg").mean().alias("mean_xg"),
        pl.col("shot_statsbomb_xg").median().alias("median_xg")
    ]).collect()
    
    print(f"  Total shots: {shot_count:,}")
    print(f"  Mean xG: {xg_stats['mean_xg'][0]:.3f}")
    print(f"  Median xG: {xg_stats['median_xg'][0]:.3f}")
    
    # Pass completion (for possession metrics)
    print("\nPASS QUALITY METRICS:")
    passes = events_lf.filter(pl.col("type") == "Pass")
    pass_stats = passes.select([
        pl.len().alias("total"),
        pl.col("pass_outcome").is_null().sum().alias("completed")
    ]).collect()
    
    completion_rate = pass_stats['completed'][0] / pass_stats['total'][0] * 100
    print(f"  Total passes: {pass_stats['total'][0]:,}")
    print(f"  Completion rate: {completion_rate:.1f}%")

    return {
        "total_events": total_events,
        "location_coverage": location_pct,
        "duplicates": duplicates,
        "under_pressure_pct": under_pressure/total_events*100,
        "shot_count": shot_count,
        "median_xg": float(xg_stats['median_xg'][0]),
        "pass_completion": completion_rate
    }


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_competition_comparison(statsbomb_dir: Path, figsize=(16, 6)) -> None:
    """Side-by-side comparison of men's vs women's competitions."""
    lf = pl.scan_parquet(statsbomb_dir / "matches.parquet")
    
    womens = lf.filter(pl.col("competition_name").str.contains("Women"))
    womens_comps = womens.group_by("competition_name").agg(
        pl.len().alias("count")
    ).sort("count", descending=True).collect()
    
    mens = lf.filter(~pl.col("competition_name").str.contains("Women"))
    mens_comps = mens.group_by("competition_name").agg(
        pl.len().alias("count")
    ).sort("count", descending=True).head(10).collect()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    ax1.barh(range(len(womens_comps)), womens_comps['count'].to_list(), color='#e74c3c')
    ax1.set_yticks(range(len(womens_comps)))
    ax1.set_yticklabels(womens_comps['competition_name'].to_list())
    ax1.set_xlabel('Matches', fontsize=11)
    ax1.set_title("Women's Football (Our Focus)", fontweight='bold', fontsize=12)
    ax1.invert_yaxis()
    ax1.grid(axis='x', alpha=0.3)
    
    ax2.barh(range(len(mens_comps)), mens_comps['count'].to_list(), color='#3498db')
    ax2.set_yticks(range(len(mens_comps)))
    ax2.set_yticklabels(mens_comps['competition_name'].to_list())
    ax2.set_xlabel('Matches', fontsize=11)
    ax2.set_title("Men's Football (Top 10)", fontweight='bold', fontsize=12)
    ax2.invert_yaxis()
    ax2.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_temporal_concentration(statsbomb_dir: Path, figsize=(16, 6)) -> None:
    """Visualize 2015-2016 concentration vs women's balanced distribution."""
    lf = pl.scan_parquet(statsbomb_dir / "matches.parquet")
    
    with_year = lf.with_columns(
        pl.col("match_date").str.strptime(pl.Date, "%Y-%m-%d").dt.year().alias("year")
    )
    
    all_years = with_year.group_by("year").agg(
        pl.len().alias("count")
    ).sort("year").filter(pl.col("year") >= 2015).collect()
    
    womens_years = with_year.filter(
        pl.col("competition_name").str.contains("Women")
    ).group_by("year").agg(
        pl.len().alias("count")
    ).sort("year").collect()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    years_list = all_years['year'].to_list()
    counts_list = all_years['count'].to_list()
    colors = ['#e74c3c' if y in [2015, 2016] else '#95a5a6' for y in years_list]
    
    ax1.bar(years_list, counts_list, color=colors, edgecolor='black', linewidth=0.7)
    ax1.axvline(x=2015.5, color='red', linestyle='--', alpha=0.6, linewidth=2, label='2015-2016 Period')
    ax1.set_xlabel('Year', fontsize=11)
    ax1.set_ylabel('Matches', fontsize=11)
    ax1.set_title('Full Dataset: 53.7% in 2015-2016 (BIAS)', fontweight='bold', fontsize=12, color='#e74c3c')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    womens_years_list = womens_years['year'].to_list()
    womens_counts_list = womens_years['count'].to_list()
    
    ax2.bar(womens_years_list, womens_counts_list, color='#3498db', edgecolor='black', linewidth=0.7)
    ax2.set_xlabel('Year', fontsize=11)
    ax2.set_ylabel('Matches', fontsize=11)
    ax2.set_title("Women's Data: Balanced 2018-2025", fontweight='bold', fontsize=12, color='#27ae60')
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def generate_summary_stats(statsbomb_dir: Path) -> dict[str, Any]:
    """
    Generate all summary statistics for executive notebook.
    """
    matches_lf = pl.scan_parquet(statsbomb_dir / "matches.parquet")
    total_matches = matches_lf.select(pl.len()).collect()[0, 0]
    
    womens = matches_lf.filter(pl.col("competition_name").str.contains("Women"))
    womens_total = womens.select(pl.len()).collect()[0, 0]
    
    events_lf = pl.scan_parquet(statsbomb_dir / "events.parquet")
    total_events = events_lf.select(pl.len()).collect()[0, 0]
    
    has_loc = events_lf.filter(pl.col("location_x").is_not_null()).select(pl.len()).collect()[0, 0]
    
    with_year = matches_lf.with_columns(
        pl.col("match_date").str.strptime(pl.Date, "%Y-%m-%d").dt.year().alias("year")
    )
    period_2015_2016 = with_year.filter(pl.col("year").is_in([2015, 2016])).select(pl.len()).collect()[0, 0]
    
    return {
        "total_matches": total_matches,
        "womens_matches": womens_total,
        "womens_pct": womens_total / total_matches * 100,
        "total_events": total_events,
        "location_coverage_pct": has_loc / total_events * 100,
        "temporal_concentration_pct": period_2015_2016 / total_matches * 100
    }
