"""Run all soccer analytics metrics and display results in terminal."""

from pathlib import Path
from src.data import load_data
from src.metrics import (
    # Basic metrics
    aggregate_xg_by_team,
    aggregate_xg_by_player,
    # Possession metrics
    calculate_ppda,
    calculate_field_tilt,
    calculate_possession_percentage, 
    calculate_possession_by_zone, 
    calculate_high_turnovers, 
    calculate_possession_value, 
    calculate_sequence_length, 
    calculate_counter_attack_speed, 
    calculate_defensive_actions_by_zone,
    analyze_possession_quality,
    # Progression metrics
    calculate_progressive_passes,
    calculate_progressive_carries,
    calculate_progressive_passes_received,
    calculate_progressive_actions,
    analyze_progression_profile,
    # Advanced metrics
    calculate_xg_chain,
    calculate_xg_buildup,
    compare_xg_chain_vs_buildup,
    calculate_packing,
)


def print_dataframe(df, title, max_rows=10):
    """Pretty print a dataframe with title."""
    print(f"\n{title}")
    print("=" * 80)
    if len(df) == 0:
        print("  (No data)")
    else:
        print(df.head(max_rows).to_string(index=False))
        if len(df) > max_rows:
            print(f"\n  ... and {len(df) - max_rows} more rows")
    print()


def main():
    """Run all metrics and display in terminal."""
    
    print("="*80)
    print(" " * 25 + "SOCCER ANALYTICS METRICS")
    print("="*80)
    
    # Setup
    loader = load_data(data_dir="./data/Statsbomb")
    events_path = str(loader.available_files['events'])
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    
    # Dataset info
    summary = loader.get_data_summary()
    print(f"\n Dataset: {summary.get('total_matches', 'N/A')} matches, "
          f"{summary.get('total_events', 'N/A'):,} events")
    print(f" Data directory: ./data/Statsbomb")
    
    # ==================================================================
    # POSSESSION METRICS
    # ==================================================================
    print("\n" + "="*80)
    print(" " * 30 + "POSSESSION METRICS")
    print("="*80)
    
    # 1. Possession Percentage
    print("\nPossession Percentage (Pass Count Method)")
    poss_pct = calculate_possession_percentage(events_path, loader.conn)
    print_dataframe(poss_pct, "Team Possession %", max_rows=20)
    poss_pct.to_csv(output_dir / "possession_percentage.csv", index=False)
    
    # 2. Possession by Zone
    print("Possession by Zone")
    poss_zones = calculate_possession_by_zone(events_path, loader.conn)
    print_dataframe(poss_zones, "Possession Distribution by Zone", max_rows=20)
    poss_zones.to_csv(output_dir / "possession_by_zone.csv", index=False)
    
    # 3. High Turnovers
    print("High Turnovers (Gegenpressing)")
    high_turnovers = calculate_high_turnovers(events_path, loader.conn)
    print_dataframe(high_turnovers, "High Turnovers (x >= 72)", max_rows=20)
    high_turnovers.to_csv(output_dir / "high_turnovers.csv", index=False)
    
    # 4. PPDA
    print("PPDA (Pressing Intensity)")
    ppda = calculate_ppda(events_path, loader.conn)
    print_dataframe(ppda, "Passes Per Defensive Action", max_rows=20)
    ppda.to_csv(output_dir / "ppda.csv", index=False)
    
    # 5. Field Tilt
    print("Field Tilt (Final Third Dominance)")
    field_tilt = calculate_field_tilt(events_path, loader.conn)
    print_dataframe(field_tilt, "Field Tilt % (Final Third x>80)", max_rows=20)
    field_tilt.to_csv(output_dir / "field_tilt.csv", index=False)
    
    # 6. Possession Value (EPR)
    print("Possession Value (Efficient Possession Ratio)")
    poss_value = calculate_possession_value(events_path, loader.conn)
    print_dataframe(poss_value, "Possession Efficiency (EPR = Possession% / xG)", max_rows=20)
    poss_value.to_csv(output_dir / "possession_value.csv", index=False)
    
    # 7. Sequence Length
    print("Sequence Length (Build-up Style)")
    seq_length = calculate_sequence_length(events_path, loader.conn)
    print_dataframe(seq_length, "Average Passes per Sequence", max_rows=20)
    seq_length.to_csv(output_dir / "sequence_length.csv", index=False)
    
    # 8. Counter-Attack Speed
    print("Counter-Attack Speed")
    counter_speed = calculate_counter_attack_speed(events_path, loader.conn)
    print_dataframe(counter_speed, "Counter-Attack Speed (Distance/Time)", max_rows=20)
    if len(counter_speed) > 0 and 'note' not in counter_speed.columns:
        counter_speed.to_csv(output_dir / "counter_attack_speed.csv", index=False)
    
    # 9. Defensive Actions by Zone
    print("Defensive Actions by Zone")
    def_zones = calculate_defensive_actions_by_zone(events_path, loader.conn)
    print_dataframe(def_zones, "Defensive Strategy by Zone", max_rows=20)
    def_zones.to_csv(output_dir / "defensive_actions_by_zone.csv", index=False)

    # 10. Possession Quality 
    print(" Possession Quality (Possession vs Tilt)")
    poss_quality = analyze_possession_quality(events_path, loader.conn)
    print_dataframe(poss_quality, "Possession Quality Analysis", max_rows=20)
    poss_quality.to_csv(output_dir / "possession_quality.csv", index=False)
    
    # ==================================================================
    # BASIC METRICS
    # ==================================================================
    print("\n" + "="*80)
    print(" " * 32 + "BASIC METRICS")
    print("="*80)
    
    print("\n xG by Team")
    team_xg = aggregate_xg_by_team(events_path, loader.conn)
    print_dataframe(team_xg, "Expected Goals by Team", max_rows=20)
    team_xg.to_csv(output_dir / "xg_team.csv", index=False)
    
    print("xG by Player")
    player_xg = aggregate_xg_by_player(events_path, loader.conn, min_shots=5)
    print_dataframe(player_xg, "Expected Goals by Player (min 5 shots)", max_rows=15)
    player_xg.to_csv(output_dir / "xg_player.csv", index=False)
    
    # ==================================================================
    # PROGRESSION METRICS
    # ==================================================================
    print("\n" + "="*80)
    print(" " * 29 + "PROGRESSION METRICS")
    print("="*80)
    
    print("\nProgressive Passes")
    prog_passes = calculate_progressive_passes(events_path, loader.conn)
    print_dataframe(prog_passes, "Progressive Passes (≥10 units forward)", max_rows=15)
    prog_passes.to_csv(output_dir / "progressive_passes.csv", index=False)
    
    print("Progressive Carries")
    prog_carries = calculate_progressive_carries(events_path, loader.conn)
    print_dataframe(prog_carries, "Progressive Carries (Split-Zone Thresholds)", max_rows=15)
    prog_carries.to_csv(output_dir / "progressive_carries.csv", index=False)
    
    print("Progressive Passes Received")
    prog_received = calculate_progressive_passes_received(events_path, loader.conn)
    print_dataframe(prog_received, "Progressive Passes Received (Movement)", max_rows=15)
    prog_received.to_csv(output_dir / "progressive_passes_received.csv", index=False)
    
    print("Progressive Actions (Combined)")
    prog_actions = calculate_progressive_actions(events_path, loader.conn)
    print_dataframe(prog_actions, "Progressive Actions (Passes + Carries + Received)", max_rows=15)
    prog_actions.to_csv(output_dir / "progressive_actions.csv", index=False)
    
    print("Progression Profiles (Player Archetypes)")
    prog_profile = analyze_progression_profile(events_path, loader.conn, min_minutes=30)
    print_dataframe(prog_profile, "Progression Profiles (min 30 mins)", max_rows=15)
    prog_profile.to_csv(output_dir / "progression_profiles.csv", index=False)
    
    # ==================================================================
    # ADVANCED METRICS
    # ==================================================================
    print("\n" + "="*80)
    print(" " * 30 + "ADVANCED METRICS")
    print("="*80)

    comparison = compare_xg_chain_vs_buildup(events_path, loader.conn, is_season_data=False)
    
    comparison.to_csv(output_dir / "player_roles.csv", index=False)
    print_dataframe(comparison, "Player Role Classification (Flagged)", max_rows=15)

    buildup_cols = [
        'match_id', 'player', 'team', 'possessions_with_buildup', 
        'xg_buildup', 'minutes_played', 'xg_buildup_per90', 
        'is_reliable', 'player_role'
    ]

    xg_buildup_updated = comparison[[c for c in buildup_cols if c in comparison.columns]]
    xg_buildup_updated.to_csv(output_dir / "xg_buildup.csv", index=False)

    chain_cols = [
        'match_id', 'player', 'team', 'possessions_with_shot', 
        'xg_chain', 'xg_chain_per90', 'is_reliable', 
        'player_role', 'team_involvement_pct'
    ]

    xg_chain_updated = comparison[[c for c in chain_cols if c in comparison.columns]]
    xg_chain_updated.to_csv(output_dir / "xg_chain.csv", index=False)
    
    # Packing (if available)
    if 'three_sixty' in loader.available_files:
        print("Packing")
        three_sixty_path = str(loader.available_files['three_sixty'])
        try:
            packing = calculate_packing(events_path, three_sixty_path, loader.conn)
            print_dataframe(packing, "Packing (Opponents Bypassed)", max_rows=15)
            packing.to_csv(output_dir / "packing.csv", index=False)
        except Exception as e:
            print(f"   ✗ Error: {e}\n")
    else:
        print("Packing - SKIPPED (no 360 data available)\n")
    
    loader.close()
    
    # ==================================================================
    # SUMMARY
    # ==================================================================
    print("\n" + "="*80)
    print(" " * 35 + "SUMMARY")
    print("="*80)
    print(f"\n All metrics calculated successfully!")
    print(f"CSV files saved to: {output_dir.absolute()}/\n")
    
    print("Files created:")
    print("\n  Possession Metrics:")
    print("    • possession_percentage.csv")
    print("    • possession_by_zone.csv")
    print("    • high_turnovers.csv")
    print("    • ppda.csv")
    print("    • field_tilt.csv")
    print("    • possession_value.csv")
    print("    • sequence_length.csv")
    if len(counter_speed) > 0 and 'note' not in counter_speed.columns:
        print("    • counter_attack_speed.csv")
    print("    • defensive_actions_by_zone.csv")
    
    print("\n  Basic Metrics:")
    print("    • xg_team.csv")
    print("    • xg_player.csv")
    
    print("\n  Progression Metrics:")
    print("    • progressive_passes.csv")
    print("    • progressive_carries.csv")
    print("    • progressive_passes_received.csv")
    print("    • progressive_actions.csv")
    print("    • progression_profiles.csv")
    
    print("\n  Advanced Metrics:")
    print("    • xg_chain.csv")
    print("    • xg_buildup.csv")
    print("    • player_roles.csv")
    if 'three_sixty' in loader.available_files:
        print("    • packing.csv")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()