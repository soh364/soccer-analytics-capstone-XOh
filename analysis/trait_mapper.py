"""
Map 8 tactical dimensions to 4 trait buckets for system-fit analysis.
"""

import pandas as pd
import numpy as np


def map_dimensions_to_traits(cluster_centers, profiles_df, dimensions):
    """
    Map 8 dimensions to 4 trait buckets and calculate trait scores.
    
    Args:
        cluster_centers: DataFrame with cluster centers (8 dimensions)
        profiles_df: DataFrame with all team profiles (for normalization)
        dimensions: List of 8 dimension names
        
    Returns:
        DataFrame with trait scores (0-100) for each archetype
    """
    
    # Define trait mapping
    trait_mapping = {
        'Mobility_Intensity': {
            'dims': ['pressing_intensity', 'defensive_positioning'],
            'description': 'Pressing intensity + Defensive line height'
        },
        'Progression': {
            'dims': ['progression_style', 'territorial_dominance'],
            'description': 'Progression style + Territorial dominance'
        },
        'Control': {
            'dims': ['ball_control', 'possession_efficiency'],
            'description': 'Ball control + Possession efficiency'
        },
        'Final_Third_Output': {
            'dims': ['attacking_threat', 'buildup_quality'],
            'description': 'Attacking threat + Build-up quality'
        }
    }
    
    # Calculate trait scores for each archetype
    archetype_traits = []
    
    for idx, center in cluster_centers.iterrows():
        traits = {'cluster': center.get('cluster', idx)}
        
        for trait_name, config in trait_mapping.items():
            # Normalize each dimension to 0-1 scale
            normalized = []
            for dim in config['dims']:
                min_val = profiles_df[dim].min()
                max_val = profiles_df[dim].max()
                
                # Normalize center value
                norm_val = (center[dim] - min_val) / (max_val - min_val)
                normalized.append(norm_val)
            
            # Average and scale to 0-100
            traits[trait_name] = np.mean(normalized) * 100
        
        archetype_traits.append(traits)
    
    trait_df = pd.DataFrame(archetype_traits)
    
    return trait_df, trait_mapping


def print_trait_mapping_table(trait_mapping):
    """
    Print a nicely formatted trait mapping table.
    
    Args:
        trait_mapping: Dict with trait definitions
    """
    print("\n" + "="*70)
    print("TRAIT BUCKET DEFINITIONS")
    print("="*70)
    print(f"{'Trait':<25} {'Component Dimensions':<45}")
    print("-"*70)
    
    for trait_name, config in trait_mapping.items():
        # Format trait name
        trait_display = trait_name.replace('_', ' ')
        
        # Format dimensions
        dims_display = config['description']
        
        print(f"{trait_display:<25} {dims_display:<45}")
    
    print("="*70)


def print_archetype_traits_table(trait_df, archetype_names):
    """
    Print nicely formatted archetype trait profiles.
    
    Args:
        trait_df: DataFrame with trait scores
        archetype_names: Dict mapping cluster to name
    """
    # Add archetype names if not present
    if 'archetype_name' not in trait_df.columns:
        trait_df['archetype_name'] = trait_df['cluster'].map(archetype_names)
    
    # Reorder columns
    cols = ['archetype_name', 'Mobility_Intensity', 'Progression', 'Control', 'Final_Third_Output']
    display_df = trait_df[cols].copy()
    
    # Round values
    for col in cols[1:]:
        display_df[col] = display_df[col].round(1)
    
    print("\n" + "="*70)
    print("ARCHETYPE TRAIT PROFILES (0-100 scale)")
    print("="*70)
    
    # Print header
    print(f"{'Archetype':<25} {'Mobility':<12} {'Progress':<12} {'Control':<12} {'Output':<12}")
    print("-"*70)
    
    # Print each archetype
    for _, row in display_df.iterrows():
        name = row['archetype_name']
        mob = f"{row['Mobility_Intensity']:.1f}"
        prog = f"{row['Progression']:.1f}"
        ctrl = f"{row['Control']:.1f}"
        out = f"{row['Final_Third_Output']:.1f}"
        
        print(f"{name:<25} {mob:<12} {prog:<12} {ctrl:<12} {out:<12}")
    
    print("="*70)


def get_trait_mapping_description():
    """Return trait mapping for documentation."""
    return {
        'Mobility_Intensity': 'Pressing intensity + Defensive line height',
        'Progression': 'Progression style + Territorial dominance',
        'Control': 'Ball control + Possession efficiency',
        'Final_Third_Output': 'Attacking threat + Build-up quality'
    }