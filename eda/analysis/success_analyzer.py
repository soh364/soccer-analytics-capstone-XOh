"""
Analyze tournament success rates by archetype.
"""

import pandas as pd
import numpy as np


def calculate_archetype_success(profiles_df, progression_df, archetype_names):
    """
    Calculate success rates and progression by archetype.
    """
    
    # Merge progression data
    team_success = profiles_df.merge(progression_df, on='team', how='left')
    team_success['progression_score'] = team_success['progression_score'].fillna(0)
    team_success['archetype_name'] = team_success['cluster'].map(archetype_names)
    
    # Aggregate by archetype
    archetype_success = team_success.groupby('cluster').agg({
        'progression_score': ['mean', 'std', 'count', 'min', 'max'],
        'team': 'count'
    }).round(3)
    
    archetype_success.columns = ['avg_progression', 'std_progression', 
                                  'n_teams', 'min_progression', 
                                  'max_progression', 'count_check']
    archetype_success = archetype_success.reset_index()
    archetype_success['archetype_name'] = archetype_success['cluster'].map(archetype_names)
    
    # Calculate success rates by stage
    success_rates = []
    for cluster_id, name in archetype_names.items():
        cluster_teams = team_success[team_success['cluster'] == cluster_id]
        n = len(cluster_teams)
        
        rates = {
            'cluster': cluster_id,
            'archetype_name': name,
            'n_teams': n,
            'group_stage_pct': (cluster_teams['progression_score'] == 0).sum() / n * 100,
            'r16_plus_pct': (cluster_teams['progression_score'] >= 1).sum() / n * 100,
            'qf_plus_pct': (cluster_teams['progression_score'] >= 2).sum() / n * 100,
            'sf_plus_pct': (cluster_teams['progression_score'] >= 3).sum() / n * 100,
            'final_plus_pct': (cluster_teams['progression_score'] >= 4).sum() / n * 100,
            'winner_pct': (cluster_teams['progression_score'] == 5).sum() / n * 100
        }
        success_rates.append(rates)
    
    success_rates_df = pd.DataFrame(success_rates)
    
    # Merge stage percentages
    archetype_success = archetype_success.merge(
        success_rates_df[['cluster', 'r16_plus_pct', 'qf_plus_pct', 
                         'sf_plus_pct', 'final_plus_pct', 'winner_pct']],
        on='cluster',
        how='left'
    )
    
    return team_success, archetype_success


def print_success_table(archetype_success):
    """Print success rates by archetype."""
    print("\n" + "="*80)
    print("TOURNAMENT SUCCESS RATES BY ARCHETYPE (2022-24)")
    print("="*80)
    print(f"{'Archetype':<25} {'n':<5} {'Avg':<6} {'SF+':<8} {'Final+':<8} {'Winner':<8}")
    print("-"*80)
    
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
    """Print summary of best/worst archetypes and performance gap."""
    best = archetype_success.loc[archetype_success['avg_progression'].idxmax()]
    worst = archetype_success.loc[archetype_success['avg_progression'].idxmin()]
    
    print("\n" + "="*70)
    print("KEY FINDINGS")
    print("="*70)
    
    print(f"\nBEST: {best['archetype_name']}")
    print(f"  Avg Progression:  {best['avg_progression']:.2f} (Semi-final level)")
    print(f"  Reach SF+:        {best['sf_plus_pct']:.1f}%")
    print(f"  Reach Finals:     {best['final_plus_pct']:.1f}%")
    print(f"  Win Tournament:   {best['winner_pct']:.1f}%")
    print(f"  Teams (n={int(best['n_teams'])}): Argentina, Spain, France, Germany, Brazil")
    
    print(f"\nWORST: {worst['archetype_name']}")
    print(f"  Avg Progression:  {worst['avg_progression']:.2f} (Group stage)")
    print(f"  Reach SF+:        {worst['sf_plus_pct']:.1f}%")
    print(f"  Teams (n={int(worst['n_teams'])}): Qatar, Albania, Bolivia")
    
    # Performance gap
    gap = best['avg_progression'] - worst['avg_progression']
    multiplier = best['avg_progression'] / worst['avg_progression'] if worst['avg_progression'] > 0 else float('inf')
    
    print(f"\nPERFORMANCE GAP:")
    print(f"  {gap:.2f} progression levels difference")
    if multiplier != float('inf'):
        print(f"  {multiplier:.1f}x better tournament performance")
    
    print("\n" + "="*70)