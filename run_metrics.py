"""Complete analytics pipeline with all metrics"""

from pathlib import Path
from src.data import load_data
from src.metrics import (
    calculate_ppda, calculate_field_tilt, calculate_possession_percentage,
    calculate_possession_value, calculate_counter_attack_speed,
    calculate_team_defensive_line_height,
    calculate_pressure_metrics, calculate_defensive_profile,
    aggregate_xg_by_team, aggregate_xg_by_player,
    analyze_progression_profile, calculate_team_progression_summary,
    calculate_xg_chain, calculate_xg_buildup, calculate_team_xg_buildup,
    calculate_pass_network_centrality,
    calculate_packing,
)

SCOPES = {
    'men_tournaments_2022_24': {
        'season_name': ['2022', '2023', '2024', '2025'],
        'competition': ['FIFA World Cup', 'UEFA Euro', 'Copa America', 'African Cup of Nations'],
        'output_suffix': 'men_tourn_2022_24',
        'per_season': False,
    },
    'recent_club_players': {
        'seasons': [
            (['2021/2022', '2022'],          '2021_2022'),
            (['2022/2023', '2023'],          '2022_2023'),
            (['2023/2024', '2024', '2025'],  '2023_2024'),
        ],
        # Explicitly exclude all women's competitions
        'exclude_competitions': [
            "FA Women's Super League",
            "NWSL",
            "UEFA Women's Euro",
            "Women's World Cup",
        ],
        'output_suffix': 'recent_club_players',
        'per_season': True,
    },
}


def filter_events_by_seasons(events_path, conn, season_names, loader, suffix, exclude_competitions=None):
    """Filter events to a list of season names, write to temp parquet."""
    matches_path = str(loader.available_files['matches'])
    Path("outputs").mkdir(exist_ok=True)
    filtered_path = f"outputs/temp_filtered_{suffix}.parquet"

    placeholders = ', '.join(['?' for _ in season_names])
    params = list(season_names)

    exclude_clause = ""
    if exclude_competitions:
        excl_placeholders = ', '.join(['?' for _ in exclude_competitions])
        exclude_clause = f"AND m.competition NOT IN ({excl_placeholders})"
        params.extend(exclude_competitions)

    conn.execute(f"""
        COPY (
            SELECT e.*
            FROM '{events_path}' e
            INNER JOIN '{matches_path}' m ON e.match_id = m.match_id
            WHERE m.season_name IN ({placeholders})
            {exclude_clause}
        ) TO '{filtered_path}' (FORMAT PARQUET)
    """, params)
    return filtered_path


def filter_events_by_scope(events_path, conn, scope, loader):
    """Filter events for non-per_season scopes (competition + season filter)."""
    matches_path = str(loader.available_files['matches'])
    Path("outputs").mkdir(exist_ok=True)
    suffix = scope['output_suffix']
    filtered_path = f"outputs/temp_filtered_{suffix}.parquet"

    conditions, params = [], []
    if scope.get('season_name'):
        placeholders = ', '.join(['?' for _ in scope['season_name']])
        conditions.append(f"m.season_name IN ({placeholders})")
        params.extend(scope['season_name'])
    if scope.get('competition'):
        placeholders = ', '.join(['?' for _ in scope['competition']])
        conditions.append(f"m.competition IN ({placeholders})")
        params.extend(scope['competition'])

    where_clause = " AND ".join(conditions) if conditions else "1=1"
    conn.execute(f"""
        COPY (
            SELECT e.*
            FROM '{events_path}' e
            INNER JOIN '{matches_path}' m ON e.match_id = m.match_id
            WHERE {where_clause}
        ) TO '{filtered_path}' (FORMAT PARQUET)
    """, params)
    print(f"  ✓ Filtered: {scope.get('season_name')} x {scope.get('competition')}")
    return filtered_path


def log_metric(name, count):
    print(f"     [+] {name}: {count:,} records")


def _run_metrics(events_path, matches_path, conn, out):
    print("  [1/6] Possession & Defensive - TEAM")
    for fn, path, label in [
        (calculate_ppda,                    "possession__team__ppda.csv",         "PPDA"),
        (calculate_field_tilt,              "possession__team__field_tilt.csv",   "Field Tilt"),
        (calculate_possession_percentage,   "possession__team__percentage.csv",   "Possession %"),
        (calculate_possession_value,        "possession__team__value_epr.csv",    "Possession Value (EPR)"),
        (calculate_team_defensive_line_height, "defensive__team__line_height.csv","Defensive Line Height"),
    ]:
        df = fn(events_path, conn, matches=matches_path)
        df.to_csv(out / path, index=False)
        log_metric(label, len(df))

    counter = calculate_counter_attack_speed(events_path, conn, matches=matches_path)
    if not counter.empty and 'note' not in counter.columns:
        counter.to_csv(out / "possession__team__counter_speed.csv", index=False)
        log_metric("Counter Attack Speed", len(counter))

    print("  [2/6] Defensive - PLAYER")
    pressure = calculate_pressure_metrics(events_path, conn, matches=matches_path)
    pressure.to_csv(out / "defensive__player__pressures.csv", index=False)
    log_metric("Pressure Metrics", len(pressure))

    def_profile = calculate_defensive_profile(events_path, conn, matches=matches_path, min_actions=15)
    def_profile.to_csv(out / "defensive__player__profile.csv", index=False)
    log_metric("Defensive Profiles", len(def_profile))

    print("  [3/6] xG Metrics")
    team_xg = aggregate_xg_by_team(events_path, conn, matches=matches_path)
    team_xg.to_csv(out / "xg__team__totals.csv", index=False)
    log_metric("Team xG", len(team_xg))

    player_xg = aggregate_xg_by_player(events_path, conn, matches=matches_path, min_shots=5)
    player_xg.to_csv(out / "xg__player__totals.csv", index=False)
    log_metric("Player xG", len(player_xg))

    print("  [4/6] Progression")
    prog_profile = analyze_progression_profile(events_path, conn, matches=matches_path, min_minutes=30)
    prog_profile.to_csv(out / "progression__player__profile.csv", index=False)
    log_metric("Progression Profiles", len(prog_profile))

    team_prog = calculate_team_progression_summary(events_path, conn, matches=matches_path)
    team_prog.to_csv(out / "progression__team__summary.csv", index=False)
    log_metric("Team Progression Summary", len(team_prog))

    print("  [5/6] Advanced - xG Chain")
    xg_chain = calculate_xg_chain(events_path, conn, matches=matches_path, per_90=True)
    xg_chain.to_csv(out / "advanced__player__xg_chain.csv", index=False)
    log_metric("xG Chain", len(xg_chain))

    xg_buildup = calculate_xg_buildup(events_path, conn, matches=matches_path, per_90=True)
    xg_buildup.to_csv(out / "advanced__player__xg_buildup.csv", index=False)
    log_metric("xG Buildup", len(xg_buildup))

    team_xg_buildup = calculate_team_xg_buildup(events_path, conn, matches=matches_path)
    team_xg_buildup.to_csv(out / "advanced__team__xg_buildup.csv", index=False)
    log_metric("Team xG Buildup", len(team_xg_buildup))

    print("  [6/6] Advanced - Network & Packing")
    network = calculate_pass_network_centrality(events_path, conn, matches=matches_path)
    network.to_csv(out / "advanced__player__network_centrality.csv", index=False)
    log_metric("Network Centrality", len(network))


def run_pipeline_for_scope(scope_name, scope):
    print(f"\n{'='*70}")
    print(f"SCOPE: {scope_name.upper()}")
    print(f"{'='*70}")

    loader = load_data(data_dir="./data/Statsbomb")
    events_path = str(loader.available_files['events'])
    matches_path = str(loader.available_files['matches'])
    temp_files = []

    try:
        BASE = Path("outputs") / "raw_metrics"

        if scope.get('per_season'):
            # New format: list of (season_names, folder_name) tuples
            for season_names, folder_name in scope['seasons']:
                print(f"\n  ── {folder_name} ({season_names}) ──")
                filtered = filter_events_by_seasons(
                    events_path, loader.conn, season_names, loader,
                    suffix=f"{scope['output_suffix']}__{folder_name}",
                    exclude_competitions=scope.get('exclude_competitions')
                )
                temp_files.append(filtered)
                out = BASE / scope['output_suffix'] / folder_name
                out.mkdir(exist_ok=True, parents=True)
                print(f"  ✓ Filtered: {season_names}")
                _run_metrics(filtered, matches_path, loader.conn, out)
        else:
            filtered = filter_events_by_scope(events_path, loader.conn, scope, loader)
            temp_files.append(filtered)
            out = BASE / scope['output_suffix']
            out.mkdir(exist_ok=True, parents=True)
            _run_metrics(filtered, matches_path, loader.conn, out)

    finally:
        import os
        for f in temp_files:
            if os.path.exists(f):
                os.remove(f)
        loader.close()

    print(f"\n✓ COMPLETE: {scope_name}\n")


def main():
    import sys
    scope_names = sys.argv[1:] if len(sys.argv) > 1 else list(SCOPES.keys())
    for scope_name in scope_names:
        if scope_name not in SCOPES:
            print(f"Unknown scope: {scope_name}. Available: {list(SCOPES.keys())}")
            continue
        run_pipeline_for_scope(scope_name, SCOPES[scope_name])


if __name__ == "__main__":
    main()