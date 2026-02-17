"""Run full analytics pipeline and export grouped CSV results with scope filtering"""

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
    calculate_team_xg_buildup, calculate_team_defensive_line_height, calculate_pressure_metrics
)

# ============================================================================
# SCOPE DEFINITIONS - Define which data to process
# ============================================================================
# 
# RESEARCH DESIGN OVERVIEW:
# -------------------------
# Men's Analysis:
#   - BUILD archetypes from 2015/16 Big 5 leagues (most complete dataset)
#   - VALIDATE temporal stability on 2023/24 recent clubs
#   - CALCULATE compression (CMI) using 2022-24 tournaments
#   - PREDICT 2026 Men's World Cup
#
# Women's Analysis:
#   - BUILD archetypes from 2018-21 FA WSL (only available club data)
#   - CALCULATE compression (CMI) using 2022-23 tournaments
#   - VALIDATE predictions on 2025 Women's Euro
#   - PREDICT 2026 Women's Euro
# ============================================================================

SCOPES = {
    # ========================================================================
    # MEN'S CLUB BASELINE - Build Tactical Archetypes
    # ========================================================================
    'men_club_2015': {
        'season_name': ['2015/2016'],
        'competition': ['Serie A', 'Premier League', 'La Liga', 'Ligue 1', '1. Bundesliga'],
        'output_suffix': 'men_club_2015',
        
        # Purpose: BUILD men's tactical archetypes (K-means clustering)
        # Expected: ~1,823 matches, ~98 teams
        # Usage: 
        #   - Build 12-dimensional team profiles
        #   - K-means clustering (k=6) to identify archetypes
        #   - Calculate cluster centers for tournament assignment
        #   - Calculate club-level variance for CMI
    },
    
    # ========================================================================
    # MEN'S TOURNAMENTS - Calculate Compression & Archetype Success
    # ========================================================================
    'men_tournaments_2022_24': {
        'season_name': ['2022', '2024'],
        'competition': ['FIFA World Cup', 'UEFA Euro', 'Copa America'],
        'output_suffix': 'men_tourn_2022_24',
        
        # Purpose: CALCULATE tournament compression (CMI) and archetype success
        # Expected: ~147 matches (64 WC + 51 Euro + 32 Copa), ~72 teams
        # Usage:
        #   - Assign tournament teams to nearest 2015/16 archetype
        #   - Calculate tournament-level variance for CMI
        #   - CMI = tournament_variance / club_variance per dimension
        #   - Analyze which archetypes succeed in tournaments
        #   - Calculate archetype success rates for predictions
    },
    
    # ========================================================================
    # MEN'S TEMPORAL VALIDATION - Verify Archetypes Still Valid
    # ========================================================================
    'recent_club_validation': {
        'season_name': ['2023/2024', '2022/2023'],
        'competition': ['1. Bundesliga', 'Ligue 1'],
        'output_suffix': 'recent_club_val',
        
        # Purpose: VALIDATE temporal stability of 2015/16 archetypes
        # Expected: ~66 matches (34 Bundesliga + 32 Ligue 1), ~30 teams
        # Usage:
        #   - Assign 2023/24 teams to nearest 2015/16 archetype
        #   - Calculate distance from archetype centers
        #   - Test: Average distance < 3.0? → Taxonomy still valid
        #   - Show: All 6 archetypes still represented in modern game
        # Limitation: Only 2 leagues, not comprehensive
    },
    
    # ========================================================================
    # WOMEN'S CLUB BASELINE - Build Tactical Archetypes
    # ========================================================================
    'women_club_2018_21': {
        'season_name': ['2018/2019', '2019/2020', '2020/2021'],
        'competition': ['FA Women\'s Super League'],
        'output_suffix': 'women_club_2018_21',
        
        # Purpose: BUILD women's tactical archetypes
        # Expected: ~326 matches, ~40-50 teams
        # Usage:
        #   - Build 12-dimensional team profiles
        #   - K-means clustering (k=5-6) to identify archetypes
        #   - Calculate cluster centers for tournament assignment
        #   - Calculate club-level variance for CMI
        # Note: No recent women's club data available for temporal validation
    },
    
    # ========================================================================
    # WOMEN'S TOURNAMENTS - Compression, Validation, & Prediction
    # ========================================================================
    'women_tournaments_2022_25': {
        'season_name': ['2022', '2023', '2025'],
        'competition': ['UEFA Women\'s Euro', 'Women\'s World Cup'],
        'output_suffix': 'women_tourn_2022_25',
        
        # Purpose: CALCULATE CMI, VALIDATE predictions, PREDICT 2026
        # Expected: ~126 matches (31 Euro22 + 64 WWC23 + 31 Euro25), ~64 teams
        # Usage:
        #   Phase 1 (2022-23 tournaments - 95 matches):
        #     - Calculate women's CMI (2018-21 clubs → 2022-23 tournaments)
        #     - Analyze archetype success rates
        #   
        #   Phase 2 (2025 Euro - 31 matches):
        #     - VALIDATION: Test predictions on 2025 Euro (out-of-sample)
        #     - Assign teams to archetypes
        #     - Predict success by archetype
        #     - Compare predictions to actual results
        #     - Calculate prediction error (MAE)
        #   
        #   Phase 3 (2026 Euro):
        #     - PREDICTION: Apply validated framework to 2026
        # Note: 2025 Euro serves as validation since no recent club data exists
    },
    
    # ========================================================================
    # ALL DATA (Exploratory Only - Not Used in Analysis)
    # ========================================================================
    'all': {
        'season_name': None,  # Process everything
        'competition': None,
        'output_suffix': 'all',

    }
}


def filter_events_by_scope(events_path: str, conn, scope: dict, loader) -> str:
    """
    Create a filtered view of events based on scope.
    Returns path to filtered parquet file.
    """
    import duckdb
    
    if scope['season_name'] is None:
        # No filtering needed
        return events_path
    
    # Get matches path
    matches_path = str(loader.available_files['matches'])
    
    # Create filtered output path
    filtered_path = f"outputs/temp_filtered_{scope['output_suffix']}.parquet"
    
    # Build filter conditions using list comprehension
    # This avoids SQL injection and apostrophe issues
    season_placeholders = ', '.join(['?' for _ in scope['season_name']])
    comp_placeholders = ', '.join(['?' for _ in scope['competition']])
    
    query = f"""
    COPY (
        SELECT e.*
        FROM '{events_path}' e
        INNER JOIN '{matches_path}' m
            ON e.match_id = m.match_id
        WHERE m.season_name IN ({season_placeholders})
          AND m.competition IN ({comp_placeholders})
    ) TO '{filtered_path}' (FORMAT PARQUET)
    """
    
    # Combine parameters in correct order
    params = scope['season_name'] + scope['competition']
    
    conn.execute(query, params)
    
    print(f"  ✓ Filtered events to: {scope['output_suffix']}")
    
    return filtered_path

def log_comp(df, name):
    print(f"   [+] {name}: {len(df):,} records")

def run_pipeline_for_scope(scope_name: str, scope: dict):
    """Run the full metrics pipeline for a specific scope."""
    
    print(f"\n{'='*70}")
    print(f"PROCESSING SCOPE: {scope_name.upper()}")
    print(f"{'='*70}")
    print(f"Seasons: {scope['season_name']}")
    print(f"Competitions: {scope['competition']}")
    
    loader = load_data(data_dir="./data/Statsbomb")
    events_path = str(loader.available_files['events'])
    
    # Filter events to scope
    if scope['season_name'] is not None:
        events_path = filter_events_by_scope(events_path, loader.conn, scope, loader)
    
    # Create output directory for this scope
    out = Path("outputs") / "raw_metrics" / scope['output_suffix']
    out.mkdir(exist_ok=True, parents=True)
    
    print(f"\nOutput directory: {out}\n")
    
    # --- 1. POSSESSION & DEFENSIVE METRICS ---
    print("[Group 1/4: Possession & Defensive]")
    
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
    
    log_comp(pos_pct, "Possession metrics")

    press_df = calculate_pressure_metrics(events_path, loader.conn)
    press_df.to_csv(out / "defensive_pressures_player.csv", index=False)
    
    # Optional: Team Aggregation
    press_team = press_df.groupby(['match_id', 'team']).agg({
        'total_pressures': 'sum',
        'counterpresses': 'sum',
        'pressure_regains': 'sum',
        'high_pressures': 'sum'
    }).reset_index()
    press_team.to_csv(out / "defensive_pressures_team.csv", index=False)
    
    log_comp(press_df, "Pressure metrics")

    # --- 2. XG & SCORING ---
    print("\n[Group 2/4: xG Metrics]")
    
    team_xg = aggregate_xg_by_team(events_path, loader.conn)
    team_xg.to_csv(out / "xg_team_totals.csv", index=False)
    
    player_xg = aggregate_xg_by_player(events_path, loader.conn, min_shots=5)
    player_xg.to_csv(out / "xg_player_totals.csv", index=False)
    
    log_comp(team_xg, "xG metrics")

    # --- 3. PROGRESSION ---
    print("\n[Group 3/4: Progression]")
    
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
    
    log_comp(p_act, "Progression metrics")

    # --- 4. ADVANCED ---
    print("\n[Group 4/4: Advanced]")
    
    chain_df = calculate_xg_chain(events_path, loader.conn, per_90=True)
    chain_df.to_csv(out / "advanced_xg_chain_raw.csv", index=False)
    
    buildup_df = calculate_xg_buildup(events_path, loader.conn, per_90=True)
    buildup_df.to_csv(out / "advanced_xg_buildup_raw.csv", index=False)

    xg_buildup_team = calculate_team_xg_buildup(events_path, loader.conn)
    xg_buildup_team.to_csv(out / "advanced_xg_buildup_team.csv", index=False)

    roles_df = compare_xg_chain_vs_buildup(events_path, loader.conn, is_season_data=False)
    roles_df.to_csv(out / "advanced_player_roles_master.csv", index=False)

    if 'three_sixty' in loader.available_files:
        packing = calculate_packing(events_path, str(loader.available_files['three_sixty']), loader.conn)
        packing.to_csv(out / "advanced_packing_stats.csv", index=False)
        log_comp(packing, "Packing stats")

    log_comp(roles_df, "Advanced metrics")
    
    # Cleanup temp file
    if scope['season_name'] is not None:
        temp_file = Path(f"outputs/temp_filtered_{scope['output_suffix']}.parquet")
        if temp_file.exists():
            temp_file.unlink()
    
    loader.close()
    
    print(f"\n SCOPE COMPLETE: {scope_name}")
    print(f"   Output: {out.absolute()}\n")

def main():
    """Main entry point - process specified scopes."""
    
    import sys
    
    # Allow command-line scope selection
    if len(sys.argv) > 1:
        scope_names = sys.argv[1:]
    else:
        # Default: process just men's club 2015/16 (for EDA)
        scope_names = ['men_club_2015']
    
    print(f"\n{'='*70}")
    print(f"TACTICAL METRICS PIPELINE")
    print(f"{'='*70}")
    print(f"Scopes to process: {', '.join(scope_names)}")
    print(f"{'='*70}\n")
    
    for scope_name in scope_names:
        if scope_name not in SCOPES:
            print(f"❌ Unknown scope: {scope_name}")
            print(f"Available scopes: {list(SCOPES.keys())}")
            continue
        
        run_pipeline_for_scope(scope_name, SCOPES[scope_name])
    
    print(f"\n{'='*70}")
    print(f" ALL SCOPES COMPLETE")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()