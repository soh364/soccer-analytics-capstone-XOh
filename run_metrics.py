"""Run full analytics pipeline and export grouped CSV results"""

from pathlib import Path
from src.data import load_data
from src.metrics import (
    aggregate_xg_by_team, aggregate_xg_by_player,
    calculate_ppda, calculate_field_tilt,
    calculate_possession_percentage, calculate_possession_by_zone, 
    calculate_high_turnovers, calculate_possession_value, 
    calculate_sequence_length, calculate_counter_attack_speed, 
    calculate_defensive_actions_by_zone, analyze_possession_quality,
    calculate_progressive_passes, calculate_progressive_carries,
    calculate_progressive_passes_received, calculate_progressive_actions,
    calculate_progressive_actions_no_overlap, analyze_progression_profile, 
    calculate_xg_chain, calculate_xg_buildup, compare_xg_chain_vs_buildup, 
    calculate_packing, calculate_team_progression_summary, calculate_team_progression_detail,
    calculate_team_xg_buildup, calculate_team_defensive_line_height
)

def log_comp(df, name):
    print(f" [+] Exported: {name} ({len(df)} records)")

def main():
    loader = load_data(data_dir="./data/Statsbomb")
    events_path = str(loader.available_files['events'])
    out = Path("outputs")
    out.mkdir(exist_ok=True)
    
    summary = loader.get_data_summary()
    print(f"--- Running Pipeline: {summary.get('total_matches', 'N/A')} Matches ---")

    # --- 1. POSSESSION & DEFENSIVE METRICS ---
    print("\n[Group: Possession]")
    
    pos_pct = calculate_possession_percentage(events_path, loader.conn)
    pos_pct.to_csv(out / "possession_overall_pct.csv", index=False)
    
    pos_zone = calculate_possession_by_zone(events_path, loader.conn)
    pos_zone.to_csv(out / "possession_zone_dist.csv", index=False)
    
    tilt = calculate_field_tilt(events_path, loader.conn)
    tilt.to_csv(out / "possession_field_tilt.csv", index=False)
    
    val = calculate_possession_value(events_path, loader.conn)
    val.to_csv(out / "possession_efficiency_epr.csv", index=False)
    
    qual = analyze_possession_quality(events_path, loader.conn)
    qual.to_csv(out / "possession_quality_analysis.csv", index=False)
    
    seq = calculate_sequence_length(events_path, loader.conn)
    seq.to_csv(out / "possession_sequence_style.csv", index=False)
    
    log_comp(pos_pct, "possession_group")

    print("\n[Group: Defensive & Pressing]")
    
    ppda = calculate_ppda(events_path, loader.conn)
    ppda.to_csv(out / "defensive_ppda.csv", index=False)
    
    turnovers = calculate_high_turnovers(events_path, loader.conn)
    turnovers.to_csv(out / "defensive_high_turnovers.csv", index=False)
    
    def_zones = calculate_defensive_actions_by_zone(events_path, loader.conn)
    def_zones.to_csv(out / "defensive_actions_by_zone.csv", index=False)

    def_line_height = calculate_team_defensive_line_height(events_path, loader.conn)
    def_line_height.to_csv(out / "defensive_line_height_team.csv", index=False)
    
    counter = calculate_counter_attack_speed(events_path, loader.conn)
    if not counter.empty and 'note' not in counter.columns:
        counter.to_csv(out / "defensive_counter_speed.csv", index=False)
        
    log_comp(ppda, "defensive_group")

    # --- 2. XG & SCORING ---
    print("\n[Group: xG Metrics]")
    
    team_xg = aggregate_xg_by_team(events_path, loader.conn)
    team_xg.to_csv(out / "xg_team_totals.csv", index=False)
    
    player_xg = aggregate_xg_by_player(events_path, loader.conn, min_shots=5)
    player_xg.to_csv(out / "xg_player_totals.csv", index=False)
    
    log_comp(player_xg, "xg_group")

    # --- 3. PROGRESSION ---
    print("\n[Group: Progression]")
    
    p_pass = calculate_progressive_passes(events_path, loader.conn)
    p_pass.to_csv(out / "progression_passes.csv", index=False)
    
    p_carry = calculate_progressive_carries(events_path, loader.conn)
    p_carry.to_csv(out / "progression_carries.csv", index=False)
    
    p_recv = calculate_progressive_passes_received(events_path, loader.conn)
    p_recv.to_csv(out / "progression_received.csv", index=False)
    
    p_act = calculate_progressive_actions(events_path, loader.conn)
    p_act.to_csv(out / "progression_actions_all.csv", index=False)

    p_act_no = calculate_progressive_actions_no_overlap(events_path, loader.conn)
    p_act_no.to_csv(out / "progression_actions_no_overlap.csv", index=False)

    p_prof = analyze_progression_profile(events_path, loader.conn, min_minutes=30)
    p_prof.to_csv(out / "progression_player_profiles.csv", index=False)

    prog_team_summary = calculate_team_progression_summary(events_path, loader.conn)
    prog_team_summary.to_csv(out / "progression_team_summary.csv", index=False)

    prog_team_detail = calculate_team_progression_detail(events_path, loader.conn)
    prog_team_detail.to_csv(out / "progression_team_detail.csv", index=False)
    
    log_comp(p_act, "progression_group")

    # --- 4. ADVANCED ---
    print("\n[Group: Advanced]")
    
    # Standalone Chain calculation
    chain_df = calculate_xg_chain(events_path, loader.conn, per_90=True)
    chain_df.to_csv(out / "advanced_xg_chain_raw.csv", index=False)
    log_comp(chain_df, "xg_chain_standalone")

    # Standalone Buildup calculation
    buildup_df = calculate_xg_buildup(events_path, loader.conn, per_90=True)
    buildup_df.to_csv(out / "advanced_xg_buildup_raw.csv", index=False)
    log_comp(buildup_df, "xg_buildup_standalone")

    xg_buildup_team = calculate_team_xg_buildup(events_path, loader.conn)
    xg_buildup_team.to_csv(out / "advanced_xg_buildup_team.csv", index=False)
    log_comp(xg_buildup_team, "xg_buildup_team")

    # Role Classification (The comparison of the two)
    roles_df = compare_xg_chain_vs_buildup(events_path, loader.conn, is_season_data=False)
    roles_df.to_csv(out / "advanced_player_roles_master.csv", index=False)
    log_comp(roles_df, "player_role_classification")

    # 360 Data check
    if 'three_sixty' in loader.available_files:
        packing = calculate_packing(events_path, str(loader.available_files['three_sixty']), loader.conn)
        packing.to_csv(out / "advanced_packing_stats.csv", index=False)
        log_comp(packing, "packing_stats")

    log_comp(roles_df, "advanced_group")

    loader.close()
    print(f"\nProcessing Complete. All CSVs in: {out.absolute()}/\n")

if __name__ == "__main__":
    main()