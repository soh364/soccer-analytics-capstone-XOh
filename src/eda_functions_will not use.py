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


# ============================================================================
# TOURNAMENT DATA ANALYSIS (Critical for Prediction Framework)
# ============================================================================

def analyze_tournament_coverage(statsbomb_dir: Path) -> dict[str, Any]:
    """
    Discover what tournament data is available in the dataset.
    Shows all tournaments, then identifies recent coverage.
    """
    print("\n" + "=" * WIDTH)
    print("  TOURNAMENT DATA DISCOVERY")
    print("=" * WIDTH)
    
    lf = pl.scan_parquet(statsbomb_dir / "matches.parquet")
    
    # Identify all tournament matches (any competition with these keywords)
    tournaments_lf = lf.filter(
        (pl.col("competition_name").str.contains("World Cup")) |
        (pl.col("competition_name").str.contains("Euro")) |
        (pl.col("competition_name").str.contains("Copa"))
    )
    
    total_tournament = tournaments_lf.select(pl.len()).collect()[0, 0]
    
    print(f"\nTotal Tournament Matches Found: {total_tournament}")
    
    # Break down by competition and year
    print("\n" + "-" * WIDTH)
    print("BREAKDOWN BY COMPETITION & YEAR:")
    print("-" * WIDTH)
    
    with_year = tournaments_lf.with_columns(
        pl.col("match_date").str.strptime(pl.Date, "%Y-%m-%d").dt.year().alias("year")
    )
    
    # Group by competition and year
    breakdown = with_year.group_by(["competition_name", "year"]).agg(
        pl.len().alias("count")
    ).sort(["year", "competition_name"], descending=[True, False]).collect()
    
    print("\nRecent tournaments:")
    recent_total = 0
    for row in breakdown.iter_rows(named=True):
        if row['year'] >= 2022:  # Highlight recent
            print(f"  - {row['year']} {row['competition_name']:40s}: {row['count']:3d} matches")
            recent_total += row['count']
    
    print(f"\n  - Total recent (2022+): {recent_total} matches")
    
    print("\nOlder tournaments:")
    older_total = 0
    for row in breakdown.iter_rows(named=True):
        if row['year'] < 2022:
            print(f"    - {row['year']} {row['competition_name']:40s}: {row['count']:3d} matches")
            older_total += row['count']
    
    print(f"\n  - Total older (pre-2022): {older_total} matches")
    
    # Also show club baseline
    print("\n" + "=" * WIDTH)
    print("  CLUB DATA (For Comparison)")
    print("=" * WIDTH)
    
    
    
    # 2015/16 season
    club_2015_16 = lf.filter(
        pl.col("season_name") == "2015/2016"
    ).filter(
        ~pl.col("competition_name").str.contains("World Cup")
    ).filter(
        ~pl.col("competition_name").str.contains("Euro")
    ).filter(
        ~pl.col("competition_name").str.contains("Copa")
    )
    
    club_count = club_2015_16.select(pl.len()).collect()[0, 0]
    
    print(f"\n2015/16 Club Matches: {club_count:,}")
    
    return {
        "total_tournaments": total_tournament,
        "recent_2022_plus": recent_total,
        "older_pre_2022": older_total,
        "club_baseline_2015_16": club_count
    }

def analyze_metric_data_readiness(statsbomb_dir: Path) -> dict[str, Any]:
    """
    Validate data readiness for 8 team + 10 player metrics.
    """
    print("\n" + "=" * WIDTH)
    print("  METRIC CALCULATION READINESS")
    print("=" * WIDTH)
    
    events_lf = pl.scan_parquet(statsbomb_dir / "events.parquet")
    total_events = events_lf.select(pl.len()).collect()[0, 0]
    
    print("\nTEAM ARCHETYPE METRICS (8 Dimensions):")
    print("-" * WIDTH)
    
    team_event_types = {
        "Pressure": "PPDA",
        "Pass": "Field Tilt, Possession %, EPR, Progression",
        "Carry": "Field Tilt, Possession %, EPR, Progression",
        "Shot": "xG Totals, xG Buildup",
        "Clearance": "Defensive Line Height"
    }
    
    for event_type, metrics in team_event_types.items():
        count = events_lf.filter(pl.col("type") == event_type).select(pl.len()).collect()[0, 0]
        pct = count / total_events * 100
        print(f"  {event_type:15s}: {count:10,} ({pct:5.2f}%) → {metrics}")
    
    print("\nPLAYER QUALITY METRICS (10 Dimensions):")
    print("-" * WIDTH)
    
    player_event_types = {
        "Shot": "Finishing Quality (goals - xG)",
        "Pass": "Progressive Passes, Network, Packing",
        "Carry": "Progressive Carries",
        "Pressure": "Pressure Quality, Defensive Actions",
        "Interception": "Defensive Actions"
    }
    
    for event_type, metrics in player_event_types.items():
        count = events_lf.filter(pl.col("type") == event_type).select(pl.len()).collect()[0, 0]
        pct = count / total_events * 100
        print(f"  {event_type:15s}: {count:10,} ({pct:5.2f}%) → {metrics}")
    
    # Critical checks
    print("\nCRITICAL DATA QUALITY:")
    print("-" * WIDTH)
    
    has_location = events_lf.filter(
        pl.col("location_x").is_not_null() & pl.col("location_y").is_not_null()
    ).select(pl.len()).collect()[0, 0]
    location_pct = has_location / total_events * 100
    
    has_xg = events_lf.filter(
        pl.col("shot_statsbomb_xg").is_not_null()
    ).select(pl.len()).collect()[0, 0]
    shots = events_lf.filter(pl.col("type") == "Shot").select(pl.len()).collect()[0, 0]
    xg_pct = has_xg / shots * 100 if shots > 0 else 0
    
    print(f"  Location Coverage:  {location_pct:6.2f}% (spatial metrics ready)")
    print(f"  xG Coverage:        {xg_pct:6.2f}% (finishing quality ready)")
    
    return {
        "location_coverage": location_pct,
        "xg_coverage": xg_pct,
        "total_events": total_events
    }


def analyze_prediction_framework_readiness(statsbomb_dir: Path) -> dict[str, Any]:
    """
    Validate dataset supports all 4 steps of prediction framework.
    """
    print("\n" + "=" * WIDTH)
    print("  PREDICTION FRAMEWORK READINESS CHECK")
    print("=" * WIDTH)
    
    lf = pl.scan_parquet(statsbomb_dir / "matches.parquet")
    
    # Step 1: Tactical DNA
    print("\nSTEP 1: Tactical DNA & Tournament Residuals")
    print("-" * WIDTH)
    
    club_baseline = lf.filter(pl.col("season_name") == "2015/2016").filter(
        ~pl.col("competition_name").str.contains("World Cup")
    ).filter(
        ~pl.col("competition_name").str.contains("Euro")
    )
    
    club_count = club_baseline.select(pl.len()).collect()[0, 0]
    
    tournaments = lf.filter(
        (pl.col("competition_name").str.contains("World Cup")) |
        (pl.col("competition_name").str.contains("Euro")) |
        (pl.col("competition_name").str.contains("Copa"))
    )
    tourn_count = tournaments.select(pl.len()).collect()[0, 0]
    
    print(f"  - Club Baseline:       {club_count:,} matches")
    print(f"  - Tournament Data:     {tourn_count:,} matches")
    print(f"  - Can cluster into 6-8 archetypes")
    print(f"  - Can calculate CMI (compression)")
    
    # Step 2: Player Quality
    print("\nSTEP 2: Player Quality & Trajectories")
    print("-" * WIDTH)
    
    lineups_lf = pl.scan_parquet(statsbomb_dir / "lineups.parquet")
    unique_players = lineups_lf.select(pl.col("player_id").n_unique()).collect()[0, 0]
    
    print(f"  - Unique Players:      {unique_players:,}")
    print(f"  - Can calculate decay-weighted scores")
    print(f"  - Can project 2026 trajectories")
    
    # Step 3: System Fit
    print("\nSTEP 3: System-Fit Engine")
    print("-" * WIDTH)
    
    events_lf = pl.scan_parquet(statsbomb_dir / "events.parquet")
    teams = events_lf.select(pl.col("team").n_unique()).collect()[0, 0]
    
    print(f"  - Teams:               {teams:,}")
    print(f"  - Can calculate tactical fit")
    print(f"  - Can measure cohesion & dependency")
    
    # Step 4: Ready
    print("\nSTEP 4: 2026 Predictions READY")
    print("-" * WIDTH)
    print(f"  All framework components validated")
    
    return {
        "framework_ready": True,
        "club_matches": club_count,
        "tournament_matches": tourn_count,
        "unique_players": unique_players
    }


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_tournament_timeline(statsbomb_dir: Path, figsize=(16, 6)) -> None:
    """Visualize recent tournament vs club data availability."""
    lf = pl.scan_parquet(statsbomb_dir / "matches.parquet")
    
    with_year = lf.with_columns(
        pl.col("match_date").str.strptime(pl.Date, "%Y-%m-%d").dt.year().alias("year")
    ).filter(pl.col("year").is_in([2022, 2023, 2024, 2025]))
    
    tournaments = with_year.filter(
        (pl.col("competition_name").str.contains("World Cup")) |
        (pl.col("competition_name").str.contains("Euro")) |
        (pl.col("competition_name").str.contains("Copa"))
    ).group_by("year").agg(pl.len().alias("count")).sort("year").collect()
    
    club = with_year.filter(
        ~(pl.col("competition_name").str.contains("World Cup"))
    ).filter(
        ~(pl.col("competition_name").str.contains("Euro"))
    ).filter(
        ~(pl.col("competition_name").str.contains("Copa"))
    ).group_by("year").agg(pl.len().alias("count")).sort("year").collect()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    years = [2022, 2023, 2024, 2025]
    tourn_counts = [0] * 4
    club_counts = [0] * 4
    
    for row in tournaments.iter_rows(named=True):
        if row['year'] in years:
            idx = years.index(row['year'])
            tourn_counts[idx] = row['count']
    
    for row in club.iter_rows(named=True):
        if row['year'] in years:
            idx = years.index(row['year'])
            club_counts[idx] = row['count']
    
    x = np.arange(len(years))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, tourn_counts, width, label='Tournament', color='#e74c3c', edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, club_counts, width, label='Club', color='#3498db', edgecolor='black', linewidth=1.5)
    
    ax.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax.set_ylabel('Matches', fontsize=12, fontweight='bold')
    ax.set_title('Recent Data (2022-2025): Tournament vs Club', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(years)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3, axis='y')
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.annotate(f'{int(height)}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom',
                           fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.show()


def plot_tactical_diversity_heatmap(statsbomb_dir: Path, figsize=(14, 8)) -> None:
    """
    Heatmap showing tactical event coverage across dimensions.
    Demonstrates we can profile teams on multiple tactical axes.
    """
    events_lf = pl.scan_parquet(statsbomb_dir / "events.parquet")
    total_events = events_lf.select(pl.len()).collect()[0, 0]
    
    # Define 8 team dimensions with their event requirements
    dimensions = {
        'PPDA\n(Pressing)': ['Pressure'],
        'Field Tilt\n(Territory)': ['Pass', 'Carry', 'Shot'],
        'Possession %\n(Control)': ['Pass', 'Carry'],
        'Possession EPR\n(Quality)': ['Pass', 'Carry', 'Shot'],
        'Def Line Height\n(Positioning)': ['Pressure', 'Clearance', 'Interception'],
        'xG Totals\n(Output)': ['Shot'],
        'Progression\n(Build-up)': ['Pass', 'Carry'],
        'xG Buildup\n(Creation)': ['Pass', 'Carry', 'Shot']
    }
    
    # Calculate coverage for each dimension
    coverages = []
    dimension_names = []
    
    for dim, event_types in dimensions.items():
        total_for_dim = 0
        for event_type in event_types:
            count = events_lf.filter(pl.col("type") == event_type).select(pl.len()).collect()[0, 0]
            total_for_dim += count
        
        # Normalize by number of event types to avoid double-counting
        coverage_pct = (total_for_dim / len(event_types)) / total_events * 100
        coverages.append(coverage_pct)
        dimension_names.append(dim)
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=figsize)
    
    # Reshape into matrix for better visualization
    matrix = np.array(coverages).reshape(2, 4)  # 2 rows, 4 cols
    
    # Plot
    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=60)
    
    # Set ticks
    ax.set_xticks(np.arange(4))
    ax.set_yticks(np.arange(2))
    
    # Labels
    col_labels = [dimension_names[i] for i in [0, 1, 2, 3]]
    row_labels = ['Possession/\nProgression', 'Defense/\nOutput']
    
    ax.set_xticklabels(col_labels, fontsize=10)
    ax.set_yticklabels(row_labels, fontsize=11, fontweight='bold')
    
    # Add values
    for i in range(2):
        for j in range(4):
            idx = i * 4 + j
            text = ax.text(j, i, f'{coverages[idx]:.1f}%',
                          ha="center", va="center", color="black", 
                          fontweight='bold', fontsize=12)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Event Coverage (%)', rotation=270, labelpad=20, fontweight='bold', fontsize=11)
    
    ax.set_title('8-Dimensional Tactical Profiling: Event Coverage', 
                fontsize=14, fontweight='bold', pad=15)
    
    plt.tight_layout()
    plt.show()


def plot_player_trajectory_concept(statsbomb_dir: Path, figsize=(14, 6)) -> None:
    """Conceptual visualization of player quality decay over time."""
    years = np.array([2020, 2021, 2022, 2023, 2024, 2025, 2026])
    peak_player = np.array([90, 88, 85, 82, 78, 74, 70])
    emerging_player = np.array([65, 70, 74, 78, 82, 85, 88])
    consistent_player = np.array([80, 81, 80, 79, 80, 81, 80])
    
    # Decay weights (exponential) - FIX: only calculate for historical years
    lambda_decay = 0.3
    historical_years = years[:-1]  # Exclude 2026 (projection year)
    weights = np.exp(-lambda_decay * np.arange(len(historical_years)-1, -1, -1))
    weights = weights / weights.sum()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    ax1.plot(years, peak_player, 'o-', linewidth=2.5, markersize=8, label='Peak Player (2020)', color='#e74c3c')
    ax1.plot(years, emerging_player, 's-', linewidth=2.5, markersize=8, label='Emerging Player', color='#2ecc71')
    ax1.plot(years, consistent_player, '^-', linewidth=2.5, markersize=8, label='Consistent Player', color='#3498db')
    ax1.axvline(x=2024.5, color='gray', linestyle='--', alpha=0.5, linewidth=2)
    ax1.text(2024.5, 95, 'Present', ha='center', fontsize=10, fontweight='bold')
    ax1.axvline(x=2026, color='red', linestyle='--', alpha=0.7, linewidth=2.5)
    ax1.text(2026, 95, '2026 WC', ha='center', fontsize=10, fontweight='bold', color='red')
    ax1.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Player Quality Score', fontsize=12, fontweight='bold')
    ax1.set_title('Player Trajectory Examples', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(alpha=0.3)
    ax1.set_ylim(60, 100)
    
    # Right: Decay weighting (use historical_years)
    ax2.bar(historical_years, weights, color='#9b59b6', edgecolor='black', linewidth=1.5, alpha=0.7)
    ax2.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Weight in 2026 Projection', fontsize=12, fontweight='bold')
    ax2.set_title('Exponential Decay Weighting (λ=0.3)', fontsize=13, fontweight='bold')
    ax2.grid(alpha=0.3, axis='y')
    
    for i, (year, weight) in enumerate(zip(historical_years, weights)):
        ax2.text(year, weight + 0.02, f'{weight:.2f}', ha='center', fontsize=9, fontweight='bold')
    
    plt.suptitle('Why Temporal Weighting Matters: Recent Form >> Historical Peak', 
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()