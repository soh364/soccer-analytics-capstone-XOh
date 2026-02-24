"""Possession-based metrics: PPDA, Field Tilt, Possession %, High Turnovers, and Advanced Possession Analysis"""

import pandas as pd
import numpy as np
import duckdb
from typing import Union, Optional


def calculate_ppda(events, conn=None, matches=None, match_id=None) -> pd.DataFrame:
    """Calculate PPDA (Passes Per Defensive Action) for teams."""
    if isinstance(events, str):
        if conn is None:
            conn = duckdb.connect()

        match_filter = f"WHERE match_id = {match_id}" if match_id else ""
        season_join = f"LEFT JOIN '{matches}' m ON tm.match_id = m.match_id" if matches else ""
        season_select = "m.season_name," if matches else ""

        query = f"""
        WITH team_matches AS (
            SELECT DISTINCT match_id, team
            FROM '{events}'
            {match_filter}
        ),
        defensive_actions AS (
            SELECT e.match_id, e.team, COUNT(*) as def_actions
            FROM '{events}' e
            WHERE e.type IN ('Interception', 'Tackle', 'Foul Committed', 'Duel')
              AND e.location_x > 48
              {f"AND e.match_id = {match_id}" if match_id else ""}
            GROUP BY e.match_id, e.team
        ),
        opponent_passes AS (
            SELECT e.match_id, e.team as opponent, COUNT(*) as opp_passes
            FROM '{events}' e
            WHERE e.type = 'Pass'
              AND e.location_x > 48
              {f"AND e.match_id = {match_id}" if match_id else ""}
            GROUP BY e.match_id, e.team
        )
        SELECT 
            {season_select}
            tm.match_id, tm.team,
            COALESCE(op.opp_passes, 0) as opponent_passes,
            COALESCE(da.def_actions, 0) as defensive_actions,
            CASE 
                WHEN COALESCE(da.def_actions, 0) = 0 THEN NULL
                ELSE ROUND(COALESCE(op.opp_passes, 0)::FLOAT / da.def_actions, 2)
            END as ppda
        FROM team_matches tm
        LEFT JOIN defensive_actions da ON tm.match_id = da.match_id AND tm.team = da.team
        LEFT JOIN opponent_passes op ON tm.match_id = op.match_id AND tm.team != op.opponent
        {season_join}
        ORDER BY tm.match_id, tm.team
        """
        return conn.execute(query).df()

    else:
        df = events.copy()
        if match_id is not None:
            df = df[df['match_id'] == match_id]

        team_matches = df[['match_id', 'team']].drop_duplicates()
        results = []

        for _, row in team_matches.iterrows():
            mid, team = row['match_id'], row['team']
            match_teams = df[df['match_id'] == mid]['team'].unique()
            opponent = [t for t in match_teams if t != team][0] if len(match_teams) > 1 else None
            if opponent is None:
                continue

            defensive_actions = df[
                (df['match_id'] == mid) & (df['team'] == team) &
                (df['type'].isin(['Interception', 'Tackle', 'Foul Committed', 'Duel'])) &
                (df['location_x'] > 48)
            ].shape[0]

            opponent_passes = df[
                (df['match_id'] == mid) & (df['team'] == opponent) &
                (df['type'] == 'Pass') & (df['location_x'] > 48)
            ].shape[0]

            ppda = round(opponent_passes / defensive_actions, 2) if defensive_actions > 0 else None
            results.append({'match_id': mid, 'team': team, 'opponent_passes': opponent_passes,
                           'defensive_actions': defensive_actions, 'ppda': ppda})

        return pd.DataFrame(results)


def calculate_field_tilt(events, conn=None, matches=None, match_id=None) -> pd.DataFrame:
    """Calculate Field Tilt - percentage of play in opponent's final third (x > 80)."""
    if isinstance(events, str):
        if conn is None:
            conn = duckdb.connect()

        season_join = f"LEFT JOIN '{matches}' m ON e.match_id = m.match_id" if matches else ""
        season_select = "m.season_name," if matches else ""
        season_group = "m.season_name," if matches else ""

        query = f"""
        SELECT 
            {season_select}
            e.match_id, e.team,
            COUNT(*) as total_actions,
            COUNT(*) FILTER (WHERE e.location_x > 80) as final_third_actions,
            ROUND(COUNT(*) FILTER (WHERE e.location_x > 80) * 100.0 / COUNT(*), 2) as field_tilt_pct
        FROM '{events}' e
        {season_join}
        WHERE e.type IN ('Pass', 'Carry', 'Shot', 'Dribble')
          AND e.location_x IS NOT NULL
          {f"AND e.match_id = {match_id}" if match_id else ""}
        GROUP BY {season_group} e.match_id, e.team
        ORDER BY e.match_id, field_tilt_pct DESC
        """
        return conn.execute(query).df()

    else:
        df = events.copy()
        if match_id is not None:
            df = df[df['match_id'] == match_id]

        df = df[df['type'].isin(['Pass', 'Carry', 'Shot', 'Dribble']) & df['location_x'].notna()]
        result = df.groupby(['match_id', 'team']).agg(
            total_actions=('type', 'count'),
            final_third_actions=('location_x', lambda x: (x > 80).sum())
        ).reset_index()
        result['field_tilt_pct'] = round(result['final_third_actions'] * 100.0 / result['total_actions'], 2)
        return result.sort_values(['match_id', 'field_tilt_pct'], ascending=[True, False])


def calculate_possession_percentage(events, conn=None, matches=None, match_id=None) -> pd.DataFrame:
    """Calculate possession % using pass count as proxy."""
    if isinstance(events, str):
        if conn is None:
            conn = duckdb.connect()

        season_join = f"LEFT JOIN '{matches}' m ON tp.match_id = m.match_id" if matches else ""
        season_select = "m.season_name," if matches else ""
        season_group = "m.season_name," if matches else ""

        query = f"""
        WITH team_passes AS (
            SELECT match_id, team, COUNT(*) as passes
            FROM '{events}'
            WHERE type = 'Pass'
              {f"AND match_id = {match_id}" if match_id else ""}
            GROUP BY match_id, team
        ),
        match_totals AS (
            SELECT match_id, SUM(passes) as total_passes
            FROM team_passes
            GROUP BY match_id
        )
        SELECT 
            {season_select}
            tp.match_id, tp.team, tp.passes, mt.total_passes,
            ROUND(tp.passes * 100.0 / mt.total_passes, 2) as possession_pct
        FROM team_passes tp
        JOIN match_totals mt ON tp.match_id = mt.match_id
        {season_join}
        ORDER BY tp.match_id, possession_pct DESC
        """
        return conn.execute(query).df()

    else:
        df = events.copy()
        if match_id is not None:
            df = df[df['match_id'] == match_id]

        team_passes = df[df['type'] == 'Pass'].groupby(['match_id', 'team']).size().reset_index(name='passes')
        match_totals = team_passes.groupby('match_id')['passes'].sum().reset_index(name='total_passes')
        result = team_passes.merge(match_totals, on='match_id')
        result['possession_pct'] = round(result['passes'] * 100.0 / result['total_passes'], 2)
        return result.sort_values(['match_id', 'possession_pct'], ascending=[True, False])


def calculate_possession_by_zone(events, conn=None, matches=None, match_id=None) -> pd.DataFrame:
    """Calculate possession distribution across pitch zones."""
    if isinstance(events, str):
        if conn is None:
            conn = duckdb.connect()

        season_join = f"LEFT JOIN '{matches}' m ON zp.match_id = m.match_id" if matches else ""
        season_select = "m.season_name," if matches else ""

        query = f"""
        WITH zone_passes AS (
            SELECT match_id, team,
                CASE 
                    WHEN location_x < 40 THEN 'Defensive Third'
                    WHEN location_x < 80 THEN 'Middle Third'
                    ELSE 'Final Third'
                END as zone,
                COUNT(*) as passes
            FROM '{events}'
            WHERE type = 'Pass' AND location_x IS NOT NULL
              {f"AND match_id = {match_id}" if match_id else ""}
            GROUP BY match_id, team, zone
        ),
        team_totals AS (
            SELECT match_id, team, SUM(passes) as total_passes
            FROM zone_passes GROUP BY match_id, team
        )
        SELECT 
            {season_select}
            zp.match_id, zp.team, zp.zone, zp.passes, tt.total_passes,
            ROUND(zp.passes * 100.0 / tt.total_passes, 2) as zone_pct
        FROM zone_passes zp
        JOIN team_totals tt ON zp.match_id = tt.match_id AND zp.team = tt.team
        {season_join}
        ORDER BY zp.match_id, zp.team, zp.zone
        """
        return conn.execute(query).df()

    else:
        df = events.copy()
        if match_id is not None:
            df = df[df['match_id'] == match_id]

        df = df[(df['type'] == 'Pass') & df['location_x'].notna()].copy()
        df['zone'] = pd.cut(df['location_x'], bins=[0, 40, 80, 120],
                           labels=['Defensive Third', 'Middle Third', 'Final Third'])
        zone_passes = df.groupby(['match_id', 'team', 'zone']).size().reset_index(name='passes')
        team_totals = zone_passes.groupby(['match_id', 'team'])['passes'].sum().reset_index(name='total_passes')
        result = zone_passes.merge(team_totals, on=['match_id', 'team'])
        result['zone_pct'] = round(result['passes'] * 100.0 / result['total_passes'], 2)
        return result.sort_values(['match_id', 'team', 'zone'])


def calculate_high_turnovers(events, conn=None, matches=None, match_id=None) -> pd.DataFrame:
    """Calculate high turnovers (ball recoveries in attacking 40%, x >= 72)."""
    if isinstance(events, str):
        if conn is None:
            conn = duckdb.connect()

        season_join = f"LEFT JOIN '{matches}' m ON e.match_id = m.match_id" if matches else ""
        season_select = "m.season_name," if matches else ""
        season_group = "m.season_name," if matches else ""

        query = f"""
        SELECT 
            {season_select}
            e.match_id, e.team,
            COUNT(*) as high_turnovers,
            COUNT(*) FILTER (WHERE e.location_x >= 88) as final_third_turnovers,
            COUNT(*) FILTER (WHERE e.location_x >= 102) as box_turnovers
        FROM '{events}' e
        {season_join}
        WHERE e.type IN ('Interception', 'Tackle', 'Ball Recovery')
          AND e.location_x >= 72
          AND e.location_x IS NOT NULL
          {f"AND e.match_id = {match_id}" if match_id else ""}
        GROUP BY {season_group} e.match_id, e.team
        ORDER BY e.match_id, high_turnovers DESC
        """
        return conn.execute(query).df()

    else:
        df = events.copy()
        if match_id is not None:
            df = df[df['match_id'] == match_id]

        df = df[df['type'].isin(['Interception', 'Tackle', 'Ball Recovery']) &
                (df['location_x'] >= 72) & df['location_x'].notna()].copy()
        result = df.groupby(['match_id', 'team']).agg(
            high_turnovers=('type', 'count'),
            final_third_turnovers=('location_x', lambda x: (x >= 88).sum()),
            box_turnovers=('location_x', lambda x: (x >= 102).sum())
        ).reset_index()
        return result.sort_values(['match_id', 'high_turnovers'], ascending=[True, False])


def calculate_possession_value(events, conn=None, matches=None, match_id=None) -> pd.DataFrame:
    """Calculate Efficient Possession Ratio (EPR): possession_pct / xG."""
    possession = calculate_possession_percentage(events, conn, matches=matches, match_id=match_id)

    if isinstance(events, str):
        if conn is None:
            conn = duckdb.connect()

        query = f"""
        SELECT match_id, team, COALESCE(SUM(shot_statsbomb_xg), 0) as total_xg
        FROM '{events}'
        WHERE type = 'Shot' AND shot_statsbomb_xg IS NOT NULL
          {f"AND match_id = {match_id}" if match_id else ""}
        GROUP BY match_id, team
        """
        xg = conn.execute(query).df()

    else:
        df = events.copy()
        if match_id is not None:
            df = df[df['match_id'] == match_id]

        xg = df[(df['type'] == 'Shot') & df['shot_statsbomb_xg'].notna()].groupby(
            ['match_id', 'team'])['shot_statsbomb_xg'].sum().reset_index(name='total_xg')

    result = possession.merge(xg, on=['match_id', 'team'], how='left')
    result['total_xg'] = result['total_xg'].fillna(0)
    result['epr'] = round(result['possession_pct'] / result['total_xg'].replace(0, np.nan), 2)

    def classify_style(row):
        epr, xg, poss = row['epr'], row['total_xg'], row['possession_pct']
        if xg == 0 or pd.isna(epr):
            return 'All Bark, No Bite' if poss > 50 else 'No Threat Created'
        if epr < 20: return 'Clinical/Direct'
        elif epr <= 40: return 'Balanced'
        else: return 'Patient Build-up'

    result['style'] = result.apply(classify_style, axis=1)
    return result[['match_id', 'team', 'possession_pct', 'total_xg', 'epr', 'style']]


def calculate_sequence_length(events, conn=None, matches=None, match_id=None) -> pd.DataFrame:
    """Calculate average passes per possession sequence."""
    if isinstance(events, str):
        if conn is None:
            conn = duckdb.connect()

        season_join = f"LEFT JOIN '{matches}' m ON ps.match_id = m.match_id" if matches else ""
        season_select = "m.season_name," if matches else ""
        season_group = "m.season_name," if matches else ""

        query = f"""
        WITH possession_sequences AS (
            SELECT match_id, team, possession,
                COUNT(*) FILTER (WHERE type = 'Pass') as passes_in_sequence
            FROM '{events}'
            {f"WHERE match_id = {match_id}" if match_id else ""}
            GROUP BY match_id, team, possession
        ),
        ps AS (
            SELECT match_id, team,
                COUNT(*) as total_sequences,
                SUM(passes_in_sequence) as total_passes,
                ROUND(AVG(passes_in_sequence), 2) as avg_passes_per_sequence,
                MAX(passes_in_sequence) as longest_sequence
            FROM possession_sequences
            WHERE passes_in_sequence > 0
            GROUP BY match_id, team
        )
        SELECT {season_select} ps.*
        FROM ps
        {season_join}
        ORDER BY ps.match_id, avg_passes_per_sequence DESC
        """
        return conn.execute(query).df()

    else:
        df = events.copy()
        if match_id is not None:
            df = df[df['match_id'] == match_id]

        df['possession_id'] = df['match_id'].astype(str) + '_' + df['possession'].astype(str)
        possession_passes = df[df['type'] == 'Pass'].groupby(
            ['match_id', 'team', 'possession_id']).size().reset_index(name='passes_in_sequence')
        possession_passes = possession_passes[possession_passes['passes_in_sequence'] > 0]

        result = possession_passes.groupby(['match_id', 'team']).agg(
            total_sequences=('possession_id', 'count'),
            total_passes=('passes_in_sequence', 'sum'),
            avg_passes_per_sequence=('passes_in_sequence', lambda x: round(x.mean(), 2)),
            longest_sequence=('passes_in_sequence', 'max')
        ).reset_index()
        return result.sort_values(['match_id', 'avg_passes_per_sequence'], ascending=[True, False])


def calculate_counter_attack_speed(events, conn=None, matches=None, match_id=None, max_time_window=10) -> pd.DataFrame:
    """Calculate counter-attack speed: distance gained per second after turnovers."""
    if isinstance(events, str):
        if conn is None:
            conn = duckdb.connect()

        try:
            check_query = f"SELECT COUNT(*) as has_data FROM '{events}' WHERE timestamp IS NOT NULL LIMIT 1"
            has_timestamp_data = conn.execute(check_query).df()['has_data'].iloc[0] > 0
        except:
            has_timestamp_data = False

        if not has_timestamp_data:
            return pd.DataFrame(columns=['match_id', 'team', 'counter_attacks',
                                         'avg_distance_gained', 'avg_time_elapsed',
                                         'avg_speed_units_per_sec'])

        season_join = f"LEFT JOIN '{matches}' m ON r.match_id = m.match_id" if matches else ""
        season_select = "m.season_name," if matches else ""

        query = f"""
        WITH turnovers AS (
            SELECT match_id, team, possession, index_num as turnover_index,
                location_x as turnover_x,
                CASE WHEN timestamp IS NOT NULL AND timestamp != '' 
                THEN CAST(CAST(SPLIT_PART(timestamp, ':', 1) AS INTEGER) * 3600 +
                          CAST(SPLIT_PART(timestamp, ':', 2) AS INTEGER) * 60 +
                          CAST(SPLIT_PART(timestamp, ':', 3) AS FLOAT) AS FLOAT)
                ELSE NULL END as turnover_time_seconds
            FROM '{events}'
            WHERE type IN ('Interception', 'Tackle', 'Ball Recovery')
              AND location_x IS NOT NULL AND timestamp IS NOT NULL
              {f"AND match_id = {match_id}" if match_id else ""}
        ),
        subsequent_actions AS (
            SELECT e.match_id, e.team, e.possession, e.index_num, e.location_x,
                t.turnover_x, t.turnover_time_seconds, t.turnover_index,
                (e.location_x - t.turnover_x) as distance_gained,
                CASE WHEN e.timestamp IS NOT NULL AND t.turnover_time_seconds IS NOT NULL
                THEN CAST(CAST(SPLIT_PART(e.timestamp, ':', 1) AS INTEGER) * 3600 +
                          CAST(SPLIT_PART(e.timestamp, ':', 2) AS INTEGER) * 60 +
                          CAST(SPLIT_PART(e.timestamp, ':', 3) AS FLOAT) AS FLOAT) - t.turnover_time_seconds
                ELSE NULL END as time_elapsed
            FROM '{events}' e
            JOIN turnovers t ON e.match_id = t.match_id AND e.team = t.team
                AND e.possession = t.possession AND e.index_num > t.turnover_index
            WHERE e.type IN ('Pass', 'Carry', 'Shot')
              AND e.location_x IS NOT NULL AND e.timestamp IS NOT NULL
              {f"AND e.match_id = {match_id}" if match_id else ""}
        ),
        r AS (
            SELECT match_id, team,
                COUNT(DISTINCT possession) as counter_attacks,
                ROUND(AVG(distance_gained), 2) as avg_distance_gained,
                ROUND(AVG(time_elapsed), 2) as avg_time_elapsed,
                ROUND(AVG(distance_gained / NULLIF(time_elapsed, 0)), 2) as avg_speed_units_per_sec
            FROM subsequent_actions
            WHERE distance_gained > 0 AND time_elapsed > 0 AND time_elapsed <= {max_time_window}
            GROUP BY match_id, team
            HAVING COUNT(*) > 0
        )
        SELECT {season_select} r.*
        FROM r
        {season_join}
        ORDER BY r.match_id, avg_speed_units_per_sec DESC
        """
        try:
            return conn.execute(query).df()
        except Exception as e:
            print(f"   [!] Counter attack speed error: {e}")
            return pd.DataFrame(columns=['match_id', 'team', 'counter_attacks',
                                         'avg_distance_gained', 'avg_time_elapsed',
                                         'avg_speed_units_per_sec'])

    else:
        df = events.copy()
        if match_id is not None:
            df = df[df['match_id'] == match_id]

        if 'timestamp' not in df.columns or df['timestamp'].isna().all():
            return pd.DataFrame(columns=['match_id', 'team', 'counter_attacks',
                                         'avg_distance_gained', 'avg_time_elapsed',
                                         'avg_speed_units_per_sec'])

        df['possession_id'] = df['match_id'].astype(str) + '_' + df['possession'].astype(str)

        def parse_timestamp(ts):
            if pd.isna(ts) or ts == '': return np.nan
            if isinstance(ts, str):
                try:
                    parts = ts.split(':')
                    return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
                except: return np.nan
            return ts

        df['timestamp_seconds'] = df['timestamp'].apply(parse_timestamp)

        turnovers = df[df['type'].isin(['Interception', 'Tackle', 'Ball Recovery']) &
                      df['location_x'].notna() & df['timestamp_seconds'].notna()][
            ['match_id', 'team', 'possession_id', 'index_num', 'location_x', 'timestamp_seconds']].copy()
        turnovers.columns = ['match_id', 'team', 'possession_id', 'turnover_index', 'turnover_x', 'turnover_time']

        actions = df[df['type'].isin(['Pass', 'Carry', 'Shot']) &
                    df['location_x'].notna() & df['timestamp_seconds'].notna()][
            ['match_id', 'team', 'possession_id', 'index_num', 'location_x', 'timestamp_seconds']].copy()

        counters = actions.merge(turnovers, on=['match_id', 'team', 'possession_id'], how='inner')
        counters = counters[counters['index_num'] > counters['turnover_index']]
        counters['distance_gained'] = counters['location_x'] - counters['turnover_x']
        counters['time_elapsed'] = counters['timestamp_seconds'] - counters['turnover_time']
        counters = counters[(counters['distance_gained'] > 0) & (counters['time_elapsed'] > 0) &
                           (counters['time_elapsed'] <= max_time_window)]

        if len(counters) == 0:
            return pd.DataFrame({'match_id': [], 'team': [], 'counter_attacks': [],
                                'avg_distance_gained': [], 'avg_time_elapsed': [],
                                'avg_speed_units_per_sec': []})

        counters['speed'] = counters['distance_gained'] / counters['time_elapsed']
        result = counters.groupby(['match_id', 'team']).agg(
            counter_attacks=('possession_id', 'nunique'),
            avg_distance_gained=('distance_gained', lambda x: round(x.mean(), 2)),
            avg_time_elapsed=('time_elapsed', lambda x: round(x.mean(), 2)),
            avg_speed_units_per_sec=('speed', lambda x: round(x.mean(), 2))
        ).reset_index()
        return result.sort_values(['match_id', 'avg_speed_units_per_sec'], ascending=[True, False])


def calculate_defensive_actions_by_zone(events, conn=None, matches=None, match_id=None) -> pd.DataFrame:
    """Calculate defensive actions by pitch zone."""
    if isinstance(events, str):
        if conn is None:
            conn = duckdb.connect()

        season_join = f"LEFT JOIN '{matches}' m ON e.match_id = m.match_id" if matches else ""
        season_select = "m.season_name," if matches else ""
        season_group = "m.season_name," if matches else ""

        query = f"""
        SELECT 
            {season_select}
            e.match_id, e.team,
            CASE 
                WHEN e.location_x < 40 THEN 'Defensive Third'
                WHEN e.location_x < 80 THEN 'Middle Third'
                ELSE 'Attacking Third'
            END as zone,
            COUNT(*) as defensive_actions
        FROM '{events}' e
        {season_join}
        WHERE e.type IN ('Interception', 'Tackle', 'Block', 'Clearance')
          AND e.location_x IS NOT NULL
          {f"AND e.match_id = {match_id}" if match_id else ""}
        GROUP BY {season_group} e.match_id, e.team, zone
        ORDER BY e.match_id, e.team, zone
        """
        return conn.execute(query).df()

    else:
        df = events.copy()
        if match_id is not None:
            df = df[df['match_id'] == match_id]

        df = df[df['type'].isin(['Interception', 'Tackle', 'Block', 'Clearance']) & df['location_x'].notna()]
        df['zone'] = pd.cut(df['location_x'], bins=[0, 40, 80, 120],
                           labels=['Defensive Third', 'Middle Third', 'Attacking Third'])
        result = df.groupby(['match_id', 'team', 'zone']).size().reset_index(name='defensive_actions')
        return result.sort_values(['match_id', 'team', 'zone'])


def calculate_team_defensive_line_height(events, conn=None, matches=None, match_id=None) -> pd.DataFrame:
    """Calculate defensive line height metric from zone distribution."""
    zone_data = calculate_defensive_actions_by_zone(events, conn, matches=matches, match_id=match_id)

    zone_pivot = zone_data.pivot_table(
        index=['match_id', 'team'] + (['season_name'] if 'season_name' in zone_data.columns else []),
        columns='zone', values='defensive_actions', fill_value=0
    ).reset_index()

    # Ensure all three zone columns always exist even if filtered data has no actions in one zone
    for zone_col in ['Defensive Third', 'Middle Third', 'Attacking Third']:
        if zone_col not in zone_pivot.columns:
            zone_pivot[zone_col] = 0

    zone_pivot['total_defensive_actions'] = (
        zone_pivot['Defensive Third'] + zone_pivot['Middle Third'] + zone_pivot['Attacking Third']
    )
    zone_pivot['defensive_line_height'] = round(
        (zone_pivot['Defensive Third'] * 1 + zone_pivot['Middle Third'] * 2 + zone_pivot['Attacking Third'] * 3) /
        zone_pivot['total_defensive_actions'], 3
    )
    zone_pivot['defensive_third_pct'] = round(zone_pivot['Defensive Third'] * 100.0 / zone_pivot['total_defensive_actions'], 2)
    zone_pivot['middle_third_pct'] = round(zone_pivot['Middle Third'] * 100.0 / zone_pivot['total_defensive_actions'], 2)
    zone_pivot['attacking_third_pct'] = round(zone_pivot['Attacking Third'] * 100.0 / zone_pivot['total_defensive_actions'], 2)

    def classify_defensive_style(row):
        if row['attacking_third_pct'] > 30: return 'High Press'
        elif row['middle_third_pct'] > 50: return 'Mid-Block'
        elif row['defensive_third_pct'] > 50: return 'Low Block'
        else: return 'Balanced'

    zone_pivot['defensive_style'] = zone_pivot.apply(classify_defensive_style, axis=1)

    keep_cols = ['match_id', 'team', 'total_defensive_actions', 'defensive_line_height',
                 'defensive_third_pct', 'middle_third_pct', 'attacking_third_pct',
                 'defensive_style', 'Defensive Third', 'Middle Third', 'Attacking Third']
    if 'season_name' in zone_pivot.columns:
        keep_cols = ['season_name'] + keep_cols

    return zone_pivot[keep_cols]


def analyze_possession_quality(events, conn=None, matches=None, match_id=None) -> pd.DataFrame:
    """Compare Possession % vs Field Tilt to classify possession quality."""
    possession = calculate_possession_percentage(events, conn, matches=matches, match_id=match_id)
    tilt = calculate_field_tilt(events, conn, matches=matches, match_id=match_id)
    zones = calculate_possession_by_zone(events, conn, matches=matches, match_id=match_id)

    final_third_passes = zones[zones['zone'] == 'Final Third'][['match_id', 'team', 'passes']].copy()
    final_third_passes.columns = ['match_id', 'team', 'final_third_passes']

    result = possession.merge(tilt[['match_id', 'team', 'field_tilt_pct']], on=['match_id', 'team'])
    result = result.merge(final_third_passes, on=['match_id', 'team'], how='left')
    result['final_third_passes'] = result['final_third_passes'].fillna(0)
    result['pass_efficiency_pct'] = round(result['final_third_passes'] * 100.0 / result['passes'], 2)

    def classify_verticality(pct):
        if pct > 35: return 'Ultra-Vertical'
        elif pct > 28: return 'Vertical'
        elif pct > 22: return 'Balanced'
        elif pct > 18: return 'Patient'
        else: return 'Horizontal'

    result['verticality'] = result['pass_efficiency_pct'].apply(classify_verticality)

    tilt_diff = []
    for _, row in result.iterrows():
        opponent_tilt = result[(result['match_id'] == row['match_id']) &
                               (result['team'] != row['team'])]['field_tilt_pct'].values
        tilt_diff.append(round(row['field_tilt_pct'] - opponent_tilt[0], 2) if len(opponent_tilt) > 0 else 0)
    result['tilt_differential'] = tilt_diff

    def classify_territorial_control(diff):
        if diff > 15: return 'Total Dominance'
        elif diff > 8: return 'Strong Control'
        elif diff > -8: return 'Contested'
        elif diff > -15: return 'Under Pressure'
        else: return 'Pinned Back'

    result['territorial_control'] = result['tilt_differential'].apply(classify_territorial_control)
    result['possession_quality_gap'] = round(result['possession_pct'] - result['field_tilt_pct'] * 2, 2)

    def interpret_gap(gap):
        if gap > 15: return 'Sterile Possession'
        elif gap > 5: return 'Patient Build-up'
        elif gap > -5: return 'Balanced'
        elif gap > -15: return 'Direct & Efficient'
        else: return 'Ultra-Efficient'

    result['gap_interpretation'] = result['possession_quality_gap'].apply(interpret_gap)

    def classify_style(row):
        poss, tilt_pct, gap = row['possession_pct'], row['field_tilt_pct'], row['possession_quality_gap']
        high_poss, low_poss = poss > 55, poss < 45
        high_tilt, low_tilt = tilt_pct > 30, tilt_pct < 20
        if high_poss and high_tilt: return 'The Steamroller'
        elif high_poss and low_tilt: return 'The U-Shape (Sterile)' if gap > 15 else 'The U-Shape'
        elif low_poss and high_tilt: return 'Efficiency Experts (Elite)' if gap < -15 else 'Efficiency Experts'
        elif low_poss and low_tilt: return 'Counter-Punchers'
        elif high_tilt: return 'Attacking'
        elif low_tilt: return 'Defensive'
        else: return 'Contested'

    result['possession_style'] = result.apply(classify_style, axis=1)

    column_order = ['match_id', 'team', 'possession_pct', 'passes', 'total_passes',
                    'field_tilt_pct', 'tilt_differential', 'territorial_control',
                    'pass_efficiency_pct', 'verticality', 'final_third_passes',
                    'possession_quality_gap', 'gap_interpretation', 'possession_style']
    if 'season_name' in result.columns:
        column_order = ['season_name'] + column_order

    return result[column_order].sort_values(['match_id', 'possession_pct'], ascending=[True, False])