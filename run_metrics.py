"""Run all soccer analytics metrics and save results."""

from pathlib import Path
from src.data import load_data
from src.metrics import (
    # Basic metrics
    calculate_ppda,
    calculate_field_tilt,
    aggregate_xg_by_team,
    aggregate_xg_by_player,
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


def main():
    """Run all metrics and save to CSV."""
    
    print("="*60)
    print("Running Soccer Analytics Metrics")
    print("="*60)
    
    # Setup
    loader = load_data(data_dir="./data/Statsbomb")
    events_path = str(loader.available_files['events'])
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    
    # Dataset info
    summary = loader.get_data_summary()
    print(f"\nDataset: {summary.get('total_matches', 'N/A')} matches, "
          f"{summary.get('total_events', 'N/A'):,} events\n")
    
    # Basic Metrics
    print("BASIC METRICS")
    print("-" * 60)
    
    print("1. PPDA...")
    ppda = calculate_ppda(events_path, loader.conn)
    ppda.to_csv(output_dir / "ppda.csv", index=False)
    print(f"   ✓ Saved {len(ppda)} teams")
    
    print("2. Field Tilt...")
    field_tilt = calculate_field_tilt(events_path, loader.conn)
    field_tilt.to_csv(output_dir / "field_tilt.csv", index=False)
    print(f"   ✓ Saved {len(field_tilt)} teams")
    
    print("3. xG by Team...")
    team_xg = aggregate_xg_by_team(events_path, loader.conn)
    team_xg.to_csv(output_dir / "xg_team.csv", index=False)
    print(f"   ✓ Saved {len(team_xg)} teams")
    
    print("4. xG by Player...")
    player_xg = aggregate_xg_by_player(events_path, loader.conn, min_shots=5)
    player_xg.to_csv(output_dir / "xg_player.csv", index=False)
    print(f"   ✓ Saved {len(player_xg)} players")
    
    # Progression Metrics
    print("\nPROGRESSION METRICS")
    print("-" * 60)
    
    print("1. Progressive Passes...")
    prog_passes = calculate_progressive_passes(events_path, loader.conn)
    prog_passes.to_csv(output_dir / "progressive_passes.csv", index=False)
    print(f"   ✓ Saved {len(prog_passes)} players")
    
    print("2. Progressive Carries...")
    prog_carries = calculate_progressive_carries(events_path, loader.conn)
    prog_carries.to_csv(output_dir / "progressive_carries.csv", index=False)
    print(f"   ✓ Saved {len(prog_carries)} players")
    
    print("3. Progressive Passes Received...")
    prog_received = calculate_progressive_passes_received(events_path, loader.conn)
    prog_received.to_csv(output_dir / "progressive_passes_received.csv", index=False)
    print(f"   ✓ Saved {len(prog_received)} players")
    
    print("4. Progressive Actions (Combined)...")
    prog_actions = calculate_progressive_actions(events_path, loader.conn)
    prog_actions.to_csv(output_dir / "progressive_actions.csv", index=False)
    print(f"   ✓ Saved {len(prog_actions)} players")
    
    print("5. Progression Profiles...")
    prog_profile = analyze_progression_profile(events_path, loader.conn, min_minutes=30)
    prog_profile.to_csv(output_dir / "progression_profiles.csv", index=False)
    print(f"   ✓ Saved {len(prog_profile)} players")
    
    # Advanced Metrics
    print("\nADVANCED METRICS")
    print("-" * 60)
    
    print("1. xG Chain...")
    xg_chain = calculate_xg_chain(events_path, loader.conn)
    xg_chain.to_csv(output_dir / "xg_chain.csv", index=False)
    print(f"   ✓ Saved {len(xg_chain)} players")
    
    print("2. xG Buildup...")
    xg_buildup = calculate_xg_buildup(events_path, loader.conn)
    xg_buildup.to_csv(output_dir / "xg_buildup.csv", index=False)
    print(f"   ✓ Saved {len(xg_buildup)} players")
    
    print("3. Player Roles...")
    comparison = compare_xg_chain_vs_buildup(events_path, loader.conn)
    comparison.to_csv(output_dir / "player_roles.csv", index=False)
    print(f"   ✓ Saved {len(comparison)} players")
    
    # Packing (if available)
    if 'three_sixty' in loader.available_files:
        print("4. Packing...")
        three_sixty_path = str(loader.available_files['three_sixty'])
        try:
            packing = calculate_packing(events_path, three_sixty_path, loader.conn)
            packing.to_csv(output_dir / "packing.csv", index=False)
            print(f"   ✓ Saved {len(packing)} players")
        except Exception as e:
            print(f"   ✗ Error: {e}")
    else:
        print("4. Packing - skipped (no 360 data)")
    
    loader.close()
    
    print("\n" + "="*60)
    print(f"Complete! Results saved to {output_dir}/")
    print("="*60)
    print("\nFiles created:")
    print("  Basic: ppda.csv, field_tilt.csv, xg_team.csv, xg_player.csv")
    print("  Progression: progressive_passes.csv, progressive_carries.csv,")
    print("               progressive_passes_received.csv, progressive_actions.csv,")
    print("               progression_profiles.csv")
    print("  Advanced: xg_chain.csv, xg_buildup.csv, player_roles.csv, packing.csv")


if __name__ == "__main__":
    main()