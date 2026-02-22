"""
Map tactical dimensions to trait buckets for archetype analysis.
Dimensions are aggregated into 4 traits: Mobility, Progression, Control, Output.
"""

import pandas as pd
import numpy as np

# Trait definitions
TRAIT_MAPPING = {
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


def map_dimensions_to_traits(cluster_centers, profiles_df, dimensions):
    """Map 8 dimensions to 4 trait buckets and calculate trait scores (0-100)."""
    archetype_traits = []
    
    for idx, center in cluster_centers.iterrows():
        traits = {'cluster': center.get('cluster', idx)}
        
        for trait_name, config in TRAIT_MAPPING.items():
            normalized = []
            for dim in config['dims']:
                min_val = profiles_df[dim].min()
                max_val = profiles_df[dim].max()
                norm_val = (center[dim] - min_val) / (max_val - min_val)
                normalized.append(norm_val)
            
            traits[trait_name] = np.mean(normalized) * 100
        
        archetype_traits.append(traits)
    
    trait_df = pd.DataFrame(archetype_traits)
    
    return trait_df, TRAIT_MAPPING


def print_trait_mapping_table(trait_mapping):
    """Print trait definitions."""
    print("\n" + "="*70)
    print("TRAIT BUCKET DEFINITIONS")
    print("="*70)
    print(f"{'Trait':<25} {'Component Dimensions':<45}")
    print("-"*70)
    
    for trait_name, config in trait_mapping.items():
        trait_display = trait_name.replace('_', ' ')
        dims_display = config['description']
        print(f"{trait_display:<25} {dims_display:<45}")
    
    print("="*70)


def print_archetype_traits_table(trait_df, archetype_names):
    """Print archetype trait profiles in a formatted table."""
    if 'archetype_name' not in trait_df.columns:
        trait_df = trait_df.copy()
        trait_df['archetype_name'] = trait_df['cluster'].map(archetype_names)
    
    cols = ['archetype_name', 'Mobility_Intensity', 'Progression', 'Control', 'Final_Third_Output']
    display_df = trait_df[cols].copy()
    
    for col in cols[1:]:
        display_df[col] = display_df[col].round(1)
    
    print("\n" + "="*70)
    print("ARCHETYPE TRAIT PROFILES (0-100 scale)")
    print("="*70)
    print(f"{'Archetype':<25} {'Mobility':<12} {'Progress':<12} {'Control':<12} {'Output':<12}")
    print("-"*70)
    
    for _, row in display_df.iterrows():
        name = row['archetype_name']
        mob = f"{row['Mobility_Intensity']:.1f}"
        prog = f"{row['Progression']:.1f}"
        ctrl = f"{row['Control']:.1f}"
        out = f"{row['Final_Third_Output']:.1f}"
        print(f"{name:<25} {mob:<12} {prog:<12} {ctrl:<12} {out:<12}")
    
    print("="*70)
