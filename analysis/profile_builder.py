"""
Build 12-dimensional tactical profiles from filtered metrics.
Uses Polars for efficient aggregation.
"""

import polars as pl
from typing import Dict

def build_team_profile(metrics: Dict[str, pl.DataFrame], verbose: bool = True) -> pl.DataFrame:
    """
    Build 12-dimensional tactical profiles for all teams.
    
    Args:
        metrics: Dictionary of filtered Polars DataFrames from TacticalDataLoader
        verbose: Print progress messages
    
    Returns:
        Polars DataFrame with columns: ['team', 'dimension_1', ..., 'dimension_12']
    """
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"BUILDING TEAM TACTICAL PROFILES")
        print(f"{'='*60}\n")
    
    # Get unique teams
    teams = metrics['qual']['team'].unique().to_list()
    
    if verbose:
        print(f"Teams to profile: {len(teams)}")
    
    # D1 & D2: From possession quality
    d1_d2 = (
        metrics['qual']
        .group_by('team')
        .agg([
            pl.col('possession_pct').mean().alias('possession_dominance'),
            pl.col('field_tilt_pct').mean().alias('territorial_control')
        ])
    )
    
    # D3: Possession efficiency (inverted EPR)
    d3 = (
        metrics['eff']
        .group_by('team')
        .agg([
            (1 / pl.col('epr').mean()).alias('possession_efficiency')
        ])
    )
    
    # D4: Progression intensity
    d4 = (
        metrics['prog_summary']
        .group_by('team')
        .agg([
            pl.col('progressive_actions').mean().alias('progression_intensity')
        ])
    )
    
    # D5: Progression method
    d5 = (
        metrics['prog_detail']
        .group_by('team')
        .agg([
            pl.col('progression_method_ratio').mean().alias('progression_method')
        ])
    )
    
    # D6: Build-up complexity
    d6 = (
        metrics['xg_buildup']
        .group_by('team')
        .agg([
            pl.col('avg_xg_per_buildup_possession').mean().alias('buildup_complexity')
        ])
    )
    
    # D7: Offensive threat
    d7 = (
        metrics['xg_total']
        .group_by('team')
        .agg([
            pl.col('xg').mean().alias('offensive_threat')
        ])
    )
    
    # D8: Tempo
    d8 = (
        metrics['seq']
        .group_by('team')
        .agg([
            pl.col('avg_passes_per_sequence').mean().alias('tempo')
        ])
    )
    
    # D9: Press intensity (inverted PPDA)
    d9 = (
        metrics['ppda']
        .group_by('team')
        .agg([
            (1 / pl.col('ppda').mean()).alias('press_intensity')
        ])
    )
    
    # D10: Defensive line height
    d10 = (
        metrics['def_line']
        .group_by('team')
        .agg([
            pl.col('defensive_line_height').mean().alias('defensive_line_height')
        ])
    )
    
    # D11: Press Effectiveness (ENHANCED with pressure success)
    # Calculate turnover component
    d11_turnovers = (
        metrics['turnover']
        .group_by('team')
        .agg([
            pl.col('final_third_turnovers').mean().alias('turnover_rate')
        ])
    )
    
    # Calculate pressure success component
    d11_pressure = (
        metrics['pressure']
        .group_by('team')
        .agg([
            (pl.col('pressure_regains').sum() * 100.0 / 
             pl.col('total_pressures').sum()).alias('pressure_success')
        ])
    )
    
    # Combine into press effectiveness
    d11 = (
        d11_turnovers
        .join(d11_pressure, on='team', how='left')
        .with_columns([
            (
                (pl.col('turnover_rate').clip(upper_bound=15) / 15.0) * 0.5 +
                (pl.col('pressure_success') / 100.0) * 0.5
            ).alias('press_effectiveness')
        ])
        .select(['team', 'press_effectiveness'])
    )
    
    # D12: Counter speed
    d12 = (
        metrics['counter']
        .group_by('team')
        .agg([
            pl.col('avg_speed_units_per_sec').mean().alias('counter_speed')
        ])
    )
    
    # Join all dimensions (Polars join is very fast)
    profile = (
        d1_d2
        .join(d3, on='team', how='left')
        .join(d4, on='team', how='left')
        .join(d5, on='team', how='left')
        .join(d6, on='team', how='left')
        .join(d7, on='team', how='left')
        .join(d8, on='team', how='left')
        .join(d9, on='team', how='left')
        .join(d10, on='team', how='left')
        .join(d11, on='team', how='left')
        .join(d12, on='team', how='left')
    )
    
    if verbose:
        print(f"\n✓ Profiles created: {len(profile)} teams")
        print(f"  Dimensions: {len(profile.columns) - 1}")
        print(f"\nMissing data summary:")
        print(profile.null_count())
    
    return profile