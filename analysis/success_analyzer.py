"""
Analyze archetype success rates in tournaments.
"""

import pandas as pd
import numpy as np


def calculate_archetype_success(profiles_df, progression_df, archetype_names):
    """
    Merge progression scores and calculate success by archetype.
    
    Args:
        profiles_df: DataFrame with team profiles and cluster assignments
        progression_df: DataFrame with progression scores (from tournament_progression.py)
        archetype_names: Dict mapping cluster_id to archetype name
        
    Returns:
        Tuple of (team_success_df, archetype_success_df)
    """
    
    # Merge progression scores
    team_success = profiles_df.merge(
        progression_df,
        on='team',
        how='left'
    )
    
    # Fill missing with 0 (group stage)
    team_success['progression_score'] = team_success['progression_score'].fillna(0)
    
    # Add archetype names
    team_success['archetype_name'] = team_success['cluster'].map(archetype_names)
    
    # Calculate success by archetype
    archetype_success = team_success.groupby('cluster').agg({
        'progression_score': ['mean', 'std', 'count', 'min', 'max'],
        'team': 'count'
    }).round(3)
    
    archetype_success.columns = ['avg_progression', 'std_progression', 
                                  'n_teams', 'min_progression', 
                                  'max_progression', 'count_check']
    
    archetype_success = archetype_success.reset_index()
    archetype_success['archetype_name'] = archetype_success['cluster'].map(archetype_names)
    
    # Calculate success rates (% reaching each stage)
    success_rates = []
    for cluster_id, name in archetype_names.items():
        cluster_teams = team_success[team_success['cluster'] == cluster_id]
        
        rates = {
            'cluster': cluster_id,
            'archetype_name': name,
            'n_teams': len(cluster_teams),
            'group_stage_pct': (cluster_teams['progression_score'] == 0).sum() / len(cluster_teams) * 100,
            'r16_plus_pct': (cluster_teams['progression_score'] >= 1).sum() / len(cluster_teams) * 100,
            'qf_plus_pct': (cluster_teams['progression_score'] >= 2).sum() / len(cluster_teams) * 100,
            'sf_plus_pct': (cluster_teams['progression_score'] >= 3).sum() / len(cluster_teams) * 100,
            'final_plus_pct': (cluster_teams['progression_score'] >= 4).sum() / len(cluster_teams) * 100,
            'winner_pct': (cluster_teams['progression_score'] == 5).sum() / len(cluster_teams) * 100
        }
        success_rates.append(rates)
    
    success_rates_df = pd.DataFrame(success_rates)
    
    # Merge with main archetype success
    archetype_success = archetype_success.merge(
        success_rates_df[['cluster', 'r16_plus_pct', 'qf_plus_pct', 
                         'sf_plus_pct', 'final_plus_pct', 'winner_pct']],
        on='cluster',
        how='left'
    )
    
    return team_success, archetype_success


def print_success_table(archetype_success):
    """
    Print nicely formatted success rates table.
    
    Args:
        archetype_success: DataFrame with success metrics
    """
    print("\n" + "="*80)
    print("TOURNAMENT SUCCESS RATES BY ARCHETYPE (2022-24)")
    print("="*80)
    
    # Header
    print(f"{'Archetype':<25} {'n':<5} {'Avg':<6} {'SF+':<8} {'Final+':<8} {'Winner':<8}")
    print("-"*80)
    
    # Each archetype
    for _, row in archetype_success.iterrows():
        name = row['archetype_name']
        n = int(row['n_teams'])
        avg = f"{row['avg_progression']:.2f}"
        sf = f"{row['sf_plus_pct']:.1f}%"
        final = f"{row['final_plus_pct']:.1f}%"
        winner = f"{row['winner_pct']:.1f}%"
        
        print(f"{name:<25} {n:<5} {avg:<6} {sf:<8} {final:<8} {winner:<8}")
    
    print("="*80)


def print_key_findings(archetype_success):
    """
    Print key findings summary.
    
    Args:
        archetype_success: DataFrame with success metrics
    """
    best = archetype_success.loc[archetype_success['avg_progression'].idxmax()]
    worst = archetype_success.loc[archetype_success['avg_progression'].idxmin()]
    
    print("\n" + "="*70)
    print("KEY FINDINGS")
    print("="*70)
    
    print(f"\n🏆 BEST ARCHETYPE: {best['archetype_name']}")
    print(f"   Average Progression:  {best['avg_progression']:.2f} (Semi-final level)")
    print(f"   Reach Semi-finals:    {best['sf_plus_pct']:.1f}% of teams")
    print(f"   Reach Finals:         {best['final_plus_pct']:.1f}% of teams")
    print(f"   Win Tournament:       {best['winner_pct']:.1f}% of teams")
    print(f"   Sample: {int(best['n_teams'])} teams (Argentina, Spain, France, Germany, Brazil...)")
    
    print(f"\n📉 WORST ARCHETYPE: {worst['archetype_name']}")
    print(f"   Average Progression:  {worst['avg_progression']:.2f} (Group stage)")
    print(f"   Reach Semi-finals:    {worst['sf_plus_pct']:.1f}% of teams")
    print(f"   Sample: {int(worst['n_teams'])} teams (Qatar, Albania, Bolivia...)")
    
    # Calculate performance gap
    gap = best['avg_progression'] - worst['avg_progression']
    multiplier = best['avg_progression'] / worst['avg_progression'] if worst['avg_progression'] > 0 else float('inf')
    
    print(f"\n📊 PERFORMANCE GAP:")
    print(f"   Elite Dominators outperform Survival Mode by {gap:.2f} progression levels")
    if multiplier != float('inf'):
        print(f"   That's {multiplier:.1f}x better tournament performance")
    
    print("\n" + "="*70)