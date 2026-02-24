"""
Build 8-dimensional team profiles from tournament metrics.
"""

import polars as pl


def build_team_profile_8d(metrics, verbose=True):
    """
    Build 8-dimensional team tactical profiles from tournament metrics.
    
    Dimensions: pressing intensity, territorial dominance, ball control, 
    possession efficiency, defensive positioning, attacking threat, 
    progression style, and build-up quality.
    
    Args:
        metrics: Dict of polars DataFrames with tournament statistics
        verbose: Print progress summary
        
    Returns:
        polars DataFrame with 8 dimensions per team
    """
    
    if verbose:
        print("\n" + "="*70)
        print("BUILDING 8-DIMENSIONAL TEAM PROFILES")
        print("="*70)
    
    teams = metrics['ppda']['team'].unique().sort()
    
    if verbose:
        print(f"Teams to profile: {len(teams)}")
    
    # Dimension 1: Pressing Intensity (inverse PPDA)
    d1 = metrics['ppda'].group_by('team').agg([
        (1 / pl.col('ppda').mean()).alias('pressing_intensity')
    ])
    
    # Dimension 2: Territorial Dominance
    d2 = metrics['field_tilt'].group_by('team').agg([
        pl.col('field_tilt_pct').mean().alias('territorial_dominance')
    ])
    
    # Dimension 3: Ball Control
    d3 = metrics['possession_pct'].group_by('team').agg([
        pl.col('possession_pct').mean().alias('ball_control')
    ])
    
    # Dimension 4: Possession Efficiency
    d4 = metrics['epr'].group_by('team').agg([
        (1 / pl.col('epr').mean()).alias('possession_efficiency')
    ])
    
    # Dimension 5: Defensive Positioning
    d5 = metrics['line_height'].group_by('team').agg([
        pl.col('defensive_line_height').mean().alias('defensive_positioning')
    ])
    
    # Dimension 6: Attacking Threat
    d6 = metrics['xg'].group_by('team').agg([
        pl.col('total_xg').mean().alias('attacking_threat')
    ])
    
    # Dimension 7: Progression Style
    d7 = metrics['progression'].group_by('team').agg([
        (pl.col('progressive_passes').sum() / pl.col('progressive_carries').sum()).alias('progression_style')
    ])
    
    # Dimension 8: Build-up Quality
    d8 = metrics['buildup'].group_by('team').agg([
        pl.col('avg_xg_per_buildup_possession').mean().alias('buildup_quality')
    ])
    
    # Join all dimensions
    profile = (
        d1.join(d2, on='team', how='left')
         .join(d3, on='team', how='left')
         .join(d4, on='team', how='left')
         .join(d5, on='team', how='left')
         .join(d6, on='team', how='left')
         .join(d7, on='team', how='left')
         .join(d8, on='team', how='left')
    )
    
    if verbose:
        print(f"\nProfiles created: {len(profile)} teams, 8 dimensions")
        
        # Check for missing data
        null_counts = profile.null_count()
        total_nulls = null_counts.select(pl.all().sum()).to_numpy()[0].sum()
        
        if total_nulls > 0:
            print(f"Missing data: {total_nulls} values")
            print(null_counts)
    
    return profile