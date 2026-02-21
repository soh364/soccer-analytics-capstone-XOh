"""Complete analytics pipeline with all metrics

NAMING CONVENTION: {category}__{level}__{metric}.csv
    
    Categories:
        possession__ = Possession & territorial control
        defensive__  = Defensive actions & intensity
        progression__ = Ball advancement
        xg__         = Expected goals
        advanced__   = Advanced metrics (xG Chain, Packing, Network)
    
    Levels:
        __team__     = Team-level aggregation
        __player__   = Player-level
"""

from pathlib import Path
from src.data import load_data
from src.metrics import (
    # POSSESSION & DEFENSIVE (TEAM)
    calculate_ppda,
    calculate_field_tilt,
    calculate_possession_percentage,
    calculate_possession_by_zone,
    calculate_high_turnovers,
    calculate_possession_value,
    calculate_sequence_length,
    calculate_counter_attack_speed,
    calculate_defensive_actions_by_zone,
    calculate_team_defensive_line_height,
    analyze_possession_quality,
    
    # DEFENSIVE (PLAYER)
    calculate_pressure_metrics,
    calculate_defensive_actions,
    calculate_defensive_profile,
    
    # PROGRESSION (PLAYER)
    calculate_progressive_passes,
    calculate_progressive_carries,
    calculate_progressive_passes_received,
    calculate_progressive_actions,
    calculate_progressive_actions_no_overlap,
    analyze_progression_profile,
    
    # PROGRESSION (TEAM)
    calculate_team_progression_summary,
    calculate_team_progression_detail,
    
    # XG (TEAM & PLAYER)
    aggregate_xg_by_team,
    aggregate_xg_by_player,
    calculate_pass_completion_by_zone,
    
    # ADVANCED: XG CHAIN (PLAYER & TEAM)
    calculate_xg_chain,
    calculate_xg_buildup,
    calculate_team_xg_buildup,
    compare_xg_chain_vs_buildup,
    
    # ADVANCED: NETWORK (PLAYER)
    calculate_pass_network_centrality,
    classify_network_role,
    
    # ADVANCED: PACKING (PLAYER)
    calculate_packing,
)

SCOPES = {
    # Baseline archetypes from TOURNAMENTS 
    'men_tournaments_2022_24': {
        'season_name': ['2022', '2024'],
        'competition': ['FIFA World Cup', 'UEFA Euro', 'Copa America', 'African Cup of Nations'],
        'output_suffix': 'men_tourn_2022_24',
        # Purpose: Build archetypes + define required trait profiles
    },
    
    # Player performance data (for quality scores)
    'recent_club_players': {
        'season_name': ['2021/2022', '2022/2023', '2023/2024', '2024/2025'],  # Recent form
        'competition': None,
        'output_suffix': 'recent_club_players',
        # Purpose: Calculate player quality scores with time decay
    },
    
    # Same-era CMI baseline (for compression analysis)
    'recent_club_validation': {
        'season_name': ['2023/2024'],
        'competition': ['1. Bundesliga', 'Ligue 1'],
        'output_suffix': 'recent_club_val',
    },
}

def filter_events_by_scope(events_path: str, conn, scope: dict, loader) -> str:
    """
    Create a filtered view of events based on scope.
    Returns path to filtered parquet file.
    """
    import duckdb
    from pathlib import Path
    
    # If both filters are None, no filtering needed
    if scope['season_name'] is None and scope['competition'] is None:
        return events_path
    
    # Get matches path
    matches_path = str(loader.available_files['matches'])
    
    # Create outputs directory if needed
    Path("outputs").mkdir(exist_ok=True)
    
    # Create filtered output path
    filtered_path = f"outputs/temp_filtered_{scope['output_suffix']}.parquet"
    
    # Build WHERE conditions dynamically
    conditions = []
    params = []
    
    # Add season filter if specified
    if scope['season_name'] is not None:
        season_placeholders = ', '.join(['?' for _ in scope['season_name']])
        conditions.append(f"m.season_name IN ({season_placeholders})")
        params.extend(scope['season_name'])
    
    # Add competition filter if specified
    if scope['competition'] is not None:
        comp_placeholders = ', '.join(['?' for _ in scope['competition']])
        conditions.append(f"m.competition IN ({comp_placeholders})")
        params.extend(scope['competition'])
    
    # Combine conditions
    where_clause = " AND ".join(conditions) if conditions else "1=1"
    
    query = f"""
    COPY (
        SELECT e.*
        FROM '{events_path}' e
        INNER JOIN '{matches_path}' m
            ON e.match_id = m.match_id
        WHERE {where_clause}
    ) TO '{filtered_path}' (FORMAT PARQUET)
    """
    
    conn.execute(query, params)
    
    print(f"  ✓ Filtered events to: {scope['output_suffix']}")
    
    return filtered_path


def log_metric(name, count):
    """Log metric completion."""
    print(f"   [+] {name}: {count:,} records")


def run_pipeline_for_scope(scope_name: str, scope: dict):
    """Run complete metrics pipeline for a specific scope."""
    
    print(f"\n{'='*70}")
    print(f"SCOPE: {scope_name.upper()}")
    print(f"{'='*70}")
    print(f"Seasons: {scope['season_name']}")
    print(f"Competitions: {scope['competition']}")
    
    loader = load_data(data_dir="./data/Statsbomb")
    events_path = str(loader.available_files['events'])
    
    # Filter events
    if scope['season_name'] is not None:
        events_path = filter_events_by_scope(events_path, loader.conn, scope, loader)
    
    # Create output directory
    out = Path("outputs") / scope['output_suffix']
    out.mkdir(exist_ok=True, parents=True)
    
    print(f"\nOutput: {out}\n")
    
    # ========================================================================
    # POSSESSION & DEFENSIVE METRICS (TEAM LEVEL)
    # ========================================================================
    print("[1/6] Possession & Defensive - TEAM")
    
    ppda = calculate_ppda(events_path, loader.conn)
    ppda.to_csv(out / "possession__team__ppda.csv", index=False)
    log_metric("PPDA", len(ppda))
    
    tilt = calculate_field_tilt(events_path, loader.conn)
    tilt.to_csv(out / "possession__team__field_tilt.csv", index=False)
    log_metric("Field Tilt", len(tilt))
    
    poss_pct = calculate_possession_percentage(events_path, loader.conn)
    poss_pct.to_csv(out / "possession__team__percentage.csv", index=False)
    log_metric("Possession %", len(poss_pct))
    
    '''
    poss_zone = calculate_possession_by_zone(events_path, loader.conn)
    poss_zone.to_csv(out / "possession__team__by_zone.csv", index=False)
    log_metric("Possession by Zone", len(poss_zone))
    
    high_turn = calculate_high_turnovers(events_path, loader.conn)
    high_turn.to_csv(out / "defensive__team__high_turnovers.csv", index=False)
    log_metric("High Turnovers", len(high_turn))
    '''

    poss_val = calculate_possession_value(events_path, loader.conn)
    poss_val.to_csv(out / "possession__team__value_epr.csv", index=False)
    log_metric("Possession Value (EPR)", len(poss_val))
    
    '''
    seq_len = calculate_sequence_length(events_path, loader.conn)
    seq_len.to_csv(out / "possession__team__sequence_length.csv", index=False)
    log_metric("Sequence Length", len(seq_len))
    '''

    counter = calculate_counter_attack_speed(events_path, loader.conn)
    if not counter.empty and 'note' not in counter.columns:
        counter.to_csv(out / "possession__team__counter_speed.csv", index=False)
        log_metric("Counter Attack Speed", len(counter))
    
    '''
    def_zones = calculate_defensive_actions_by_zone(events_path, loader.conn)
    def_zones.to_csv(out / "defensive__team__actions_by_zone.csv", index=False)
    log_metric("Defensive Actions by Zone", len(def_zones))
    '''

    def_line = calculate_team_defensive_line_height(events_path, loader.conn)
    def_line.to_csv(out / "defensive__team__line_height.csv", index=False)
    log_metric("Defensive Line Height", len(def_line))
    
    '''
    poss_qual = analyze_possession_quality(events_path, loader.conn)
    poss_qual.to_csv(out / "possession__team__quality_analysis.csv", index=False)
    log_metric("Possession Quality", len(poss_qual))
    '''

    # ========================================================================
    # DEFENSIVE METRICS (PLAYER LEVEL)
    # ========================================================================
    print("\n[2/6] Defensive - PLAYER")
    
    pressure = calculate_pressure_metrics(events_path, loader.conn)
    pressure.to_csv(out / "defensive__player__pressures.csv", index=False)
    log_metric("Pressure Metrics", len(pressure))
    
    '''
    def_actions = calculate_defensive_actions(events_path, loader.conn)
    def_actions.to_csv(out / "defensive__player__actions.csv", index=False)
    log_metric("Defensive Actions", len(def_actions))
    '''
    
    def_profile = calculate_defensive_profile(events_path, loader.conn, min_actions=15)
    def_profile.to_csv(out / "defensive__player__profile.csv", index=False)
    log_metric("Defensive Profiles", len(def_profile))
    
    # ========================================================================
    # XG METRICS
    # ========================================================================
    print("\n[3/6] xG Metrics")
    
    team_xg = aggregate_xg_by_team(events_path, loader.conn)
    team_xg.to_csv(out / "xg__team__totals.csv", index=False)
    log_metric("Team xG", len(team_xg))
    
    player_xg = aggregate_xg_by_player(events_path, loader.conn, min_shots=5)
    player_xg.to_csv(out / "xg__player__totals.csv", index=False)
    log_metric("Player xG", len(player_xg))
    
    '''
    pass_comp = calculate_pass_completion_by_zone(events_path, loader.conn)
    pass_comp.to_csv(out / "xg__team__pass_completion_by_zone.csv", index=False)
    log_metric("Pass Completion by Zone", len(pass_comp))
    '''
    
    # ========================================================================
    # PROGRESSION METRICS
    # ========================================================================
    print("\n[4/6] Progression")
    
    # PLAYER
    '''
    prog_pass = calculate_progressive_passes(events_path, loader.conn)
    prog_pass.to_csv(out / "progression__player__passes.csv", index=False)
    log_metric("Progressive Passes", len(prog_pass))
    
    prog_carry = calculate_progressive_carries(events_path, loader.conn)
    prog_carry.to_csv(out / "progression__player__carries.csv", index=False)
    log_metric("Progressive Carries", len(prog_carry))
    
    prog_recv = calculate_progressive_passes_received(events_path, loader.conn)
    prog_recv.to_csv(out / "progression__player__received.csv", index=False)
    log_metric("Progressive Received", len(prog_recv))
    
    prog_act = calculate_progressive_actions(events_path, loader.conn)
    prog_act.to_csv(out / "progression__player__actions_all.csv", index=False)
    log_metric("Progressive Actions (All)", len(prog_act))
    
    prog_no_overlap = calculate_progressive_actions_no_overlap(events_path, loader.conn)
    prog_no_overlap.to_csv(out / "progression__player__actions_unique.csv", index=False)
    log_metric("Progressive Actions (No Overlap)", len(prog_no_overlap))
    '''
    
    prog_profile = analyze_progression_profile(events_path, loader.conn, min_minutes=30)
    prog_profile.to_csv(out / "progression__player__profile.csv", index=False)
    log_metric("Progression Profiles", len(prog_profile))
    
    # TEAM
    team_prog_summary = calculate_team_progression_summary(events_path, loader.conn)
    team_prog_summary.to_csv(out / "progression__team__summary.csv", index=False)
    log_metric("Team Progression Summary", len(team_prog_summary))
    
    '''
    team_prog_detail = calculate_team_progression_detail(events_path, loader.conn)
    team_prog_detail.to_csv(out / "progression__team__detail.csv", index=False)
    log_metric("Team Progression Detail", len(team_prog_detail))
    '''
    
    # ========================================================================
    # ADVANCED: XG CHAIN
    # ========================================================================
    print("\n[5/6] Advanced - xG Chain")
    
    # PLAYER
    xg_chain = calculate_xg_chain(events_path, loader.conn, per_90=True)
    xg_chain.to_csv(out / "advanced__player__xg_chain.csv", index=False)
    log_metric("xG Chain", len(xg_chain))
    
    xg_buildup = calculate_xg_buildup(events_path, loader.conn, per_90=True)
    xg_buildup.to_csv(out / "advanced__player__xg_buildup.csv", index=False)
    log_metric("xG Buildup", len(xg_buildup))
    
    '''
    player_roles = compare_xg_chain_vs_buildup(events_path, loader.conn, is_season_data=False)
    player_roles.to_csv(out / "advanced__player__roles.csv", index=False)
    log_metric("Player Roles", len(player_roles))
    '''
    
    # TEAM
    team_xg_buildup = calculate_team_xg_buildup(events_path, loader.conn)
    team_xg_buildup.to_csv(out / "advanced__team__xg_buildup.csv", index=False)
    log_metric("Team xG Buildup", len(team_xg_buildup))
    
    # ========================================================================
    # ADVANCED: NETWORK & PACKING
    # ========================================================================
    print("\n[6/6] Advanced - Network & Packing")
    
    # NETWORK
    network = calculate_pass_network_centrality(events_path, loader.conn)
    network.to_csv(out / "advanced__player__network_centrality.csv", index=False)
    log_metric("Network Centrality", len(network))
    
    '''
    network_roles = classify_network_role(events_path, loader.conn, min_involvement=5.0)
    network_roles.to_csv(out / "advanced__player__network_roles.csv", index=False)
    log_metric("Network Roles", len(network_roles))
    '''
    
    # PACKING (if 360 available)
    if 'three_sixty' in loader.available_files:
        try:
            three_sixty_path = str(loader.available_files['three_sixty'])
            packing = calculate_packing(events_path, three_sixty_path, loader.conn)
            packing.to_csv(out / "advanced__player__packing.csv", index=False)
            log_metric("Packing", len(packing))
        except Exception as e:
            print(f"   [!] Packing skipped: {e}")
    else:
        print("   [!] Packing skipped (no 360 data)")
    
    # ========================================================================
    # CLEANUP
    # ========================================================================
    
    # Remove temp file
    if scope['season_name'] is not None:
        temp_file = Path(f"outputs/temp_filtered_{scope['output_suffix']}.parquet")
        if temp_file.exists():
            temp_file.unlink()
    
    loader.close()
    
    print(f"\n COMPLETE: {scope_name}")
    print(f"   Files: {out.absolute()}\n")


def main():
    """Main entry point."""
    
    import sys
    
    if len(sys.argv) > 1:
        scope_names = sys.argv[1:]
    else:
        scope_names = ['men_club_2015']
    
    print(f"\n{'='*70}")
    print(f"SOCCER ANALYTICS PIPELINE - COMPLETE")
    print(f"{'='*70}")
    print(f"Scopes: {', '.join(scope_names)}")
    print(f"{'='*70}\n")
    
    for scope_name in scope_names:
        if scope_name not in SCOPES:
            print(f"Unknown scope: {scope_name}")
            print(f"Available: {list(SCOPES.keys())}")
            continue
        
        run_pipeline_for_scope(scope_name, SCOPES[scope_name])
    
    print(f"\n{'='*70}")
    print(f"ALL SCOPES COMPLETE")
    print(f"{'='*70}\n")
    
    print("OUTPUT STRUCTURE:")
    print("outputs/{scope}/")
    print("  ")
    print("  NAMING: {category}__{level}__{metric}.csv")
    print("  ")
    print("  Categories: possession, defensive, progression, xg, advanced")
    print("  Levels: team, player")


if __name__ == "__main__":
    main()