"""Demo of all soccer analytics metrics - basic and advanced."""

from src.data import load_data
from src.metrics import (
    # Basic metrics
    calculate_ppda,
    calculate_field_tilt,
    calculate_progressive_passes,
    calculate_progressive_carries,
    aggregate_xg_by_team,
    aggregate_xg_by_player,
    # Advanced metrics
    calculate_xg_chain,
    calculate_xg_buildup,
    compare_xg_chain_vs_buildup,
    calculate_packing,
)

def run_basic_metrics():
    """Calculate basic metrics."""
    
    print("="*60)
    print("BASIC METRICS")
    print("="*60)
    
    loader = load_data(data_dir="./data/Statsbomb")
    events_path = str(loader.available_files['events'])
    
    # Dataset info
    summary = loader.get_data_summary()
    print(f"\nDataset: {summary.get('total_matches', 'N/A')} matches, "
          f"{summary.get('total_events', 'N/A'):,} events")
    
    # PPDA
    print("\n1. PPDA (Pressing Intensity)")
    ppda = calculate_ppda(events_path, loader.conn)
    print(f"   Top 5 lowest PPDA:")
    top = ppda.nsmallest(5, 'ppda')[['team', 'ppda']]
    print(top.to_string(index=False))
    
    # Progressive Passes
    print("\n2. Progressive Passes")
    prog_passes = calculate_progressive_passes(events_path, loader.conn)
    print(f"   Top 5 progressive passers:")
    top = prog_passes.nlargest(5, 'progressive_passes')[['player', 'team', 'progressive_passes']]
    print(top.to_string(index=False))
    
    # Progressive Carries
    print("\n3. Progressive Carries")
    prog_carries = calculate_progressive_carries(events_path, loader.conn)
    print(f"   Top 5 progressive carriers:")
    top = prog_carries.nlargest(5, 'progressive_carries')[['player', 'team', 'progressive_carries']]
    print(top.to_string(index=False))
    
    # xG
    print("\n4. xG by Player")
    player_xg = aggregate_xg_by_player(events_path, loader.conn, min_shots=10)
    print(f"   Top 5 by xG (min 10 shots):")
    top = player_xg.nlargest(5, 'xg')[['player', 'team', 'xg', 'goals']]
    print(top.to_string(index=False))
    
    loader.close()
    return ppda, prog_passes, prog_carries, player_xg


def run_advanced_metrics():
    """Calculate advanced metrics."""
    
    print("\n" + "="*60)
    print("ADVANCED METRICS")
    print("="*60)
    
    loader = load_data(data_dir="./data/Statsbomb")
    events_path = str(loader.available_files['events'])
    
    # xG Chain
    print("\n1. xG Chain (all players in goal possessions)")
    xg_chain = calculate_xg_chain(events_path, loader.conn)
    print(f"   Top 5 by xG Chain:")
    top = xg_chain.nlargest(5, 'xg_chain')[['player', 'team', 'xg_chain', 'goals_in_chain']]
    print(top.to_string(index=False))
    
    # xG Buildup
    print("\n2. xG Buildup (excludes shooter/assister)")
    xg_buildup = calculate_xg_buildup(events_path, loader.conn)
    print(f"   Top 5 build-up players:")
    top = xg_buildup.nlargest(5, 'xg_buildup')[['player', 'team', 'xg_buildup']]
    print(top.to_string(index=False))
    
    # Player roles
    print("\n3. Player Role Classification")
    comparison = compare_xg_chain_vs_buildup(events_path, loader.conn)
    counts = comparison['player_role'].value_counts()
    for role, count in counts.items():
        print(f"   {role}: {count}")
    
    # Packing (if available)
    if 'three_sixty' in loader.available_files:
        print("\n4. Packing (line-breaking passes)")
        three_sixty_path = str(loader.available_files['three_sixty'])
        try:
            packing = calculate_packing(events_path, three_sixty_path, loader.conn)
            print(f"   Top 5 by avg packing:")
            top = packing.nlargest(5, 'avg_packing_per_pass')[['player', 'team', 'avg_packing_per_pass']]
            print(top.to_string(index=False))
        except Exception as e:
            print(f"   Error: {e}")
            packing = None
    else:
        print("\n4. Packing - skipped (no 360 data)")
        packing = None
    
    loader.close()
    return xg_chain, xg_buildup, comparison, packing


def show_sample_match():
    """Analyze a single match in detail."""
    
    print("\n" + "="*60)
    print("SINGLE MATCH EXAMPLE")
    print("="*60)
    
    loader = load_data(data_dir="./data/Statsbomb")
    events_path = str(loader.available_files['events'])
    
    matches = loader.load_matches()
    sample = matches.iloc[0]
    mid = sample['match_id']
    
    print(f"\nMatch {mid}: {sample['home_team']} vs {sample['away_team']}")
    print(f"Score: {sample['home_score']}-{sample['away_score']}")
    
    # PPDA
    ppda = calculate_ppda(events_path, loader.conn, match_id=mid)
    print("\nPPDA:")
    for _, row in ppda.iterrows():
        print(f"  {row['team']}: {row['ppda']:.2f}")
    
    # xG
    xg = aggregate_xg_by_team(events_path, loader.conn, match_id=mid)
    print("\nxG:")
    for _, row in xg.iterrows():
        print(f"  {row['team']}: {row['xg']:.2f} ({row['goals']} goals)")
    
    loader.close()


def main():
    
    print("\n" + "="*60)
    print("Soccer Analytics Metrics Demo")
    print("="*60)
    
    # Run all metrics
    basic_results = run_basic_metrics()
    advanced_results = run_advanced_metrics()
    show_sample_match()
    
    print("\n" + "="*60)
    print("Complete")
    print("="*60)
    
    print("\nTo save results:")
    print("  ppda.to_csv('outputs/ppda.csv', index=False)")
    print("  xg_chain.to_csv('outputs/xg_chain.csv', index=False)")

if __name__ == "__main__":
    main()