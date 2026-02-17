"""
Tournament compression analysis: CMI calculation and archetype success.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import euclidean


def assign_to_archetypes(tournament_profiles, baseline_centers, dimensions):
    """
    Assign tournament teams to nearest baseline archetype.
    
    Args:
        tournament_profiles: DataFrame with tournament team profiles
        baseline_centers: DataFrame with baseline cluster centers
        dimensions: List of dimension names
        
    Returns:
        tournament_profiles with 'assigned_archetype' and 'distance_to_center'
    """
    tournament_profiles = tournament_profiles.copy()
    
    assignments = []
    distances = []
    
    for idx, team_profile in tournament_profiles.iterrows():
        team_vector = team_profile[dimensions].values
        
        min_distance = float('inf')
        assigned_cluster = None
        
        for cluster_id, center in baseline_centers.iterrows():
            center_vector = center[dimensions].values
            dist = euclidean(team_vector, center_vector)
            
            if dist < min_distance:
                min_distance = dist
                assigned_cluster = cluster_id
        
        assignments.append(assigned_cluster)
        distances.append(min_distance)
    
    tournament_profiles['assigned_archetype'] = assignments
    tournament_profiles['distance_to_center'] = distances
    
    return tournament_profiles


def calculate_cmi(baseline_profiles, tournament_profiles, dimensions):
    """
    Calculate Complexity Maintenance Index (CMI) for each dimension.
    
    CMI = tournament_variance / baseline_variance
    Lower CMI = more compression
    
    Args:
        baseline_profiles: DataFrame with baseline team profiles
        tournament_profiles: DataFrame with tournament team profiles
        dimensions: List of dimension names
        
    Returns:
        DataFrame with CMI for each dimension
    """
    cmi_results = []
    
    for dim in dimensions:
        baseline_var = baseline_profiles[dim].var()
        baseline_std = baseline_profiles[dim].std()
        baseline_mean = baseline_profiles[dim].mean()
        
        tournament_var = tournament_profiles[dim].var()
        tournament_std = tournament_profiles[dim].std()
        tournament_mean = tournament_profiles[dim].mean()
        
        cmi = tournament_std / baseline_std if baseline_std > 0 else 1.0
        compression_pct = (1 - cmi) * 100
        
        cmi_results.append({
            'dimension': dim,
            'baseline_mean': baseline_mean,
            'baseline_std': baseline_std,
            'tournament_mean': tournament_mean,
            'tournament_std': tournament_std,
            'cmi': cmi,
            'compression_pct': compression_pct
        })
    
    cmi_df = pd.DataFrame(cmi_results)
    
    # Overall CMI
    overall_cmi = cmi_df['cmi'].mean()
    overall_compression = (1 - overall_cmi) * 100
    
    print(f"\nOverall CMI: {overall_cmi:.3f}")
    print(f"Overall Compression: {overall_compression:.1f}%")
    
    return cmi_df


def analyze_archetype_distribution_shift(baseline_profiles, tournament_profiles, archetype_names):
    """
    Compare archetype distribution between baseline and tournament.
    
    Args:
        baseline_profiles: With 'cluster' column
        tournament_profiles: With 'assigned_archetype' column
        archetype_names: Dict mapping cluster_id to name
        
    Returns:
        DataFrame comparing distributions
    """
    baseline_dist = baseline_profiles['cluster'].value_counts(normalize=True).sort_index() * 100
    tournament_dist = tournament_profiles['assigned_archetype'].value_counts(normalize=True).sort_index() * 100
    
    comparison = pd.DataFrame({
        'Archetype': [archetype_names[i] for i in baseline_dist.index],
        'Baseline_%': baseline_dist.values,
        'Tournament_%': tournament_dist.values
    })
    
    comparison['Shift'] = comparison['Tournament_%'] - comparison['Baseline_%']
    
    return comparison


def calculate_archetype_success(tournament_profiles, progression_data, archetype_col='assigned_archetype'):
    """
    Calculate success metrics by archetype.
    
    Args:
        tournament_profiles: With archetype assignments
        progression_data: DataFrame with team progression scores
        archetype_col: Column name for archetype assignment
        
    Returns:
        DataFrame with success metrics by archetype
    """
    # Merge profiles with progression
    merged = tournament_profiles.merge(progression_data, on='team', how='left')
    
    success_stats = merged.groupby(archetype_col).agg({
        'progression_score': ['mean', 'std', 'count'],
        'team': 'count'
    }).round(2)
    
    success_stats.columns = ['avg_progression', 'std_progression', 'count', 'n_teams']
    success_stats = success_stats.reset_index()
    
    return success_stats