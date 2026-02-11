"""Possession-based metrics: PPDA, Field Tilt, Possession %, High Turnovers, and Advanced Possession Analysis

NOTE:
- PPDA uses x > 48 (attacking 60%) - industry standard for meaningful pressing
- Possession % based on pass counts (industry standard for event data)
- Counter-attack speed requires timestamp data (may not be available in all datasets)

Sources:
- https://dataglossary.wyscout.com/ppda/
- https://the-footballanalyst.com/field-tilt-football-statistics-explained/
- https://blogarchive.statsbomb.com/articles/soccer/splitting-possession-into-offensedefense/ 
- https://breakingthelines.com/data-analysis/efficient-possession-ratio-a-new-football-performance-metric/
- https://soccerment.com/soccerments-advanced-metrics/ 
"""

import pandas as pd
import numpy as np
import duckdb
from typing import Union, Optional


def calculate_ppda(events, conn=None, match_id=None) -> pd.DataFrame:
    """
    Calculate PPDA (Passes Per Defensive Action) for teams.
    PPDA = opponent passes / defensive actions in attacking 60% (x > 48)
    
    Lower PPDA = More intense pressing (< 8 = high press)
    Higher PPDA = More passive/sitting back (> 15 = low block)
    """

    if isinstance(events, str):
        if conn is None:
            conn = duckdb.connect()
        
        match_filter = f"WHERE match_id = {match_id}" if match_id else ""
        
        query = f"""
        WITH team_matches AS (
            SELECT DISTINCT match_id, team
            FROM '{events}'
            {match_filter}
        ),
        defensive_actions AS (
            SELECT 
                e.match_id,
                e.team,
                COUNT(*) as def_actions
            FROM '{events}' e
            WHERE e.type IN ('Interception', 'Tackle', 'Foul Committed', 'Duel')
              AND e.location_x > 48  -- Opponent's attacking 60%
              {f"AND e.match_id = {match_id}" if match_id else ""}
            GROUP BY e.match_id, e.team
        ),
        opponent_passes AS (
            SELECT 
                e.match_id,
                e.team as opponent,
                COUNT(*) as opp_passes
            FROM '{events}' e
            WHERE e.type = 'Pass'
              AND e.location_x > 48  -- Their attacking 60%
              {f"AND e.match_id = {match_id}" if match_id else ""}
            GROUP BY e.match_id, e.team
        )
        SELECT 
            tm.match_id,
            tm.team,
            COALESCE(op.opp_passes, 0) as opponent_passes,
            COALESCE(da.def_actions, 0) as defensive_actions,
            CASE 
                WHEN COALESCE(da.def_actions, 0) = 0 THEN NULL
                ELSE ROUND(COALESCE(op.opp_passes, 0)::FLOAT / da.def_actions, 2)
            END as ppda
        FROM team_matches tm
        LEFT JOIN defensive_actions da ON tm.match_id = da.match_id AND tm.team = da.team
        LEFT JOIN opponent_passes op ON tm.match_id = op.match_id AND tm.team != op.opponent
        ORDER BY tm.match_id, tm.team
        """
        
        return conn.execute(query).df()
    
    else:
        df = events.copy()
        
        if match_id is not None:
            df = df[df['match_id'] == match_id]
        
        # Get all unique team-match combinations
        team_matches = df[['match_id', 'team']].drop_duplicates()
        
        results = []
        
        for _, row in team_matches.iterrows():
            mid = row['match_id']
            team = row['team']
            
            # Get opponent
            match_teams = df[df['match_id'] == mid]['team'].unique()
            opponent = [t for t in match_teams if t != team][0] if len(match_teams) > 1 else None
            
            if opponent is None:
                continue
            
            # Defensive actions by this team high up the pitch (x > 48)
            defensive_actions = df[
                (df['match_id'] == mid) &
                (df['team'] == team) &
                (df['type'].isin(['Interception', 'Tackle', 'Foul Committed', 'Duel'])) &
                (df['location_x'] > 48)
            ].shape[0]
            
            # Opponent passes in their "attacking 60%" (same x > 48 zone)
            opponent_passes = df[
                (df['match_id'] == mid) &
                (df['team'] == opponent) &
                (df['type'] == 'Pass') &
                (df['location_x'] > 48)
            ].shape[0]
            
            ppda = round(opponent_passes / defensive_actions, 2) if defensive_actions > 0 else None
            
            results.append({
                'match_id': mid,
                'team': team,
                'opponent_passes': opponent_passes,
                'defensive_actions': defensive_actions,
                'ppda': ppda
            })
        
        return pd.DataFrame(results)


def calculate_field_tilt(events, conn=None, match_id=None) -> pd.DataFrame:
    """
    Calculate Field Tilt - percentage of play in opponent's final third (x > 80)
    
    Benchmarks:
    - >35% = Elite dominance
    - 25-35% = Strong control
    - 15-25% = Contested
    - <15% = Defensive/counter-attacking
    """

    if isinstance(events, str):
        if conn is None:
            conn = duckdb.connect()
        
        match_filter = f"WHERE match_id = {match_id}" if match_id else ""
        
        query = f"""
        SELECT 
            match_id,
            team,
            COUNT(*) as total_actions,
            COUNT(*) FILTER (WHERE location_x > 80) as final_third_actions,
            ROUND(COUNT(*) FILTER (WHERE location_x > 80) * 100.0 / COUNT(*), 2) as field_tilt_pct
        FROM '{events}'
        WHERE type IN ('Pass', 'Carry', 'Shot', 'Dribble')
          AND location_x IS NOT NULL
          {f"AND match_id = {match_id}" if match_id else ""}
        GROUP BY match_id, team
        ORDER BY match_id, field_tilt_pct DESC
        """
        
        return conn.execute(query).df()
    
    else:
        df = events.copy()
        
        if match_id is not None:
            df = df[df['match_id'] == match_id]
        
        # Filter to relevant action types
        df = df[
            df['type'].isin(['Pass', 'Carry', 'Shot', 'Dribble']) &
            df['location_x'].notna()
        ]
        
        # Calculate field tilt using FINAL THIRD (x > 80)
        result = df.groupby(['match_id', 'team']).agg(
            total_actions=('type', 'count'),
            final_third_actions=('location_x', lambda x: (x > 80).sum())
        ).reset_index()
        
        result['field_tilt_pct'] = round(
            result['final_third_actions'] * 100.0 / result['total_actions'], 2
        )
        
        return result.sort_values(['match_id', 'field_tilt_pct'], ascending=[True, False])
    
def calculate_possession_percentage(events, conn=None, match_id=None) -> pd.DataFrame:
    """
    Calculate possession % using pass count as proxy.
    
    Industry standard: Team's passes / Total match passes
    >55% = Possession dominant
    45-55% = Balanced
    <45% = Counter-attacking/defensive
    """
    
    if isinstance(events, str):
        if conn is None:
            conn = duckdb.connect()
        
        match_filter = f"WHERE match_id = {match_id}" if match_id else ""
        
        query = f"""
        WITH team_passes AS (
            SELECT 
                match_id,
                team,
                COUNT(*) as passes
            FROM '{events}'
            WHERE type = 'Pass'
              {f"AND match_id = {match_id}" if match_id else ""}
            GROUP BY match_id, team
        ),
        match_totals AS (
            SELECT 
                match_id,
                SUM(passes) as total_passes
            FROM team_passes
            GROUP BY match_id
        )
        SELECT 
            tp.match_id,
            tp.team,
            tp.passes,
            mt.total_passes,
            ROUND(tp.passes * 100.0 / mt.total_passes, 2) as possession_pct
        FROM team_passes tp
        JOIN match_totals mt ON tp.match_id = mt.match_id
        ORDER BY tp.match_id, possession_pct DESC
        """
        
        return conn.execute(query).df()
    
    else:
        df = events.copy()
        
        if match_id is not None:
            df = df[df['match_id'] == match_id]
        
        # Count passes per team
        team_passes = df[df['type'] == 'Pass'].groupby(['match_id', 'team']).size().reset_index(name='passes')
        
        # Get total passes per match
        match_totals = team_passes.groupby('match_id')['passes'].sum().reset_index(name='total_passes')
        
        # Calculate possession %
        result = team_passes.merge(match_totals, on='match_id')
        result['possession_pct'] = round(result['passes'] * 100.0 / result['total_passes'], 2)
        
        return result.sort_values(['match_id', 'possession_pct'], ascending=[True, False])


def calculate_possession_by_zone(events, conn=None, match_id=None) -> pd.DataFrame:
    """
    Calculate possession distribution across pitch zones.
    
    Shows WHERE teams have possession:
    - Defensive third (0-40): Building from back
    - Middle third (40-80): Midfield control
    - Final third (80-120): Attacking pressure
    
    High final third % (>35%) = Dominant attacking possession
    """
    
    if isinstance(events, str):
        if conn is None:
            conn = duckdb.connect()
        
        match_filter = f"AND match_id = {match_id}" if match_id else ""
        
        query = f"""
        WITH zone_passes AS (
            SELECT 
                match_id,
                team,
                CASE 
                    WHEN location_x < 40 THEN 'Defensive Third'
                    WHEN location_x < 80 THEN 'Middle Third'
                    ELSE 'Final Third'
                END as zone,
                COUNT(*) as passes
            FROM '{events}'
            WHERE type = 'Pass'
              AND location_x IS NOT NULL
              {match_filter}
            GROUP BY match_id, team, zone
        ),
        team_totals AS (
            SELECT 
                match_id,
                team,
                SUM(passes) as total_passes
            FROM zone_passes
            GROUP BY match_id, team
        )
        SELECT 
            zp.match_id,
            zp.team,
            zp.zone,
            zp.passes,
            tt.total_passes,
            ROUND(zp.passes * 100.0 / tt.total_passes, 2) as zone_pct
        FROM zone_passes zp
        JOIN team_totals tt ON zp.match_id = tt.match_id AND zp.team = tt.team
        ORDER BY zp.match_id, zp.team, zp.zone
        """
        
        return conn.execute(query).df()
    
    else:
        df = events.copy()
        
        if match_id is not None:
            df = df[df['match_id'] == match_id]
        
        # Filter to passes with location
        df = df[(df['type'] == 'Pass') & df['location_x'].notna()].copy()
        
        # Add zone column
        df['zone'] = pd.cut(
            df['location_x'],
            bins=[0, 40, 80, 120],
            labels=['Defensive Third', 'Middle Third', 'Final Third']
        )
        
        # Count passes per zone
        zone_passes = df.groupby(['match_id', 'team', 'zone']).size().reset_index(name='passes')
        
        # Get team totals
        team_totals = zone_passes.groupby(['match_id', 'team'])['passes'].sum().reset_index(name='total_passes')
        
        # Calculate percentages
        result = zone_passes.merge(team_totals, on=['match_id', 'team'])
        result['zone_pct'] = round(result['passes'] * 100.0 / result['total_passes'], 2)
        
        return result.sort_values(['match_id', 'team', 'zone'])


def calculate_high_turnovers(events, conn=None, match_id=None) -> pd.DataFrame:
    """
    Calculate high turnovers (ball recoveries in attacking 40%, x >= 72).
    
    Measures gegenpressing / high pressing effectiveness:
    >10 per match = Intense gegenpress (Klopp style)
    5-10 = Moderate high pressing
    <5 = Low/mid block
    """
    
    if isinstance(events, str):
        if conn is None:
            conn = duckdb.connect()
        
        match_filter = f"AND match_id = {match_id}" if match_id else ""
        
        query = f"""
        SELECT 
            match_id,
            team,
            COUNT(*) as high_turnovers,
            COUNT(*) FILTER (WHERE location_x >= 88) as final_third_turnovers,
            COUNT(*) FILTER (WHERE location_x >= 102) as box_turnovers
        FROM '{events}'
        WHERE type IN ('Interception', 'Tackle', 'Ball Recovery')
          AND location_x >= 72  -- Attacking 40%
          AND location_x IS NOT NULL
          {match_filter}
        GROUP BY match_id, team
        ORDER BY match_id, high_turnovers DESC
        """
        
        return conn.execute(query).df()
    
    else:
        df = events.copy()
        
        if match_id is not None:
            df = df[df['match_id'] == match_id]
        
        # Filter to ball recoveries in attacking 40%
        df = df[
            df['type'].isin(['Interception', 'Tackle', 'Ball Recovery']) &
            (df['location_x'] >= 72) &
            df['location_x'].notna()
        ].copy()
        
        # Count by zone
        result = df.groupby(['match_id', 'team']).agg(
            high_turnovers=('type', 'count'),
            final_third_turnovers=('location_x', lambda x: (x >= 88).sum()),
            box_turnovers=('location_x', lambda x: (x >= 102).sum())
        ).reset_index()
        
        return result.sort_values(['match_id', 'high_turnovers'], ascending=[True, False])


def calculate_possession_value(events, conn=None, match_id=None) -> pd.DataFrame:
    """
    Calculate Efficient Possession Ratio (EPR): possession_pct / xG
    
    Shows possession efficiency:
    - Low EPR (< 20) = Clinical/Direct (creates xG with less possession)
    - Medium EPR (20-40) = Balanced
    - High EPR (> 40) = Patient build-up (needs more possession to create xG)
    - Infinite/NaN = "All Bark, No Bite" (possession but no threat)
    
    Note: Requires shot_statsbomb_xg column in events data
    """
    
    # Get possession %
    possession = calculate_possession_percentage(events, conn, match_id)
    
    if isinstance(events, str):
        if conn is None:
            conn = duckdb.connect()
        
        match_filter = f"WHERE match_id = {match_id}" if match_id else ""
        
        query = f"""
        SELECT 
            match_id,
            team,
            COALESCE(SUM(shot_statsbomb_xg), 0) as total_xg
        FROM '{events}'
        WHERE type = 'Shot'
          AND shot_statsbomb_xg IS NOT NULL
          {match_filter}
        GROUP BY match_id, team
        """
        
        xg = conn.execute(query).df()
    
    else:
        df = events.copy()
        
        if match_id is not None:
            df = df[df['match_id'] == match_id]
        
        # Calculate xG
        xg = df[
            (df['type'] == 'Shot') &
            df['shot_statsbomb_xg'].notna()
        ].groupby(['match_id', 'team'])['shot_statsbomb_xg'].sum().reset_index(name='total_xg')
    
    # Merge possession and xG
    result = possession.merge(xg, on=['match_id', 'team'], how='left')
    result['total_xg'] = result['total_xg'].fillna(0)
    
    # Calculate EPR (Efficient Possession Ratio)
    # Replace 0 xG with NaN to avoid division issues
    result['epr'] = round(
        result['possession_pct'] / result['total_xg'].replace(0, np.nan), 2
    )
    
    # Add interpretation with proper handling of edge cases
    def classify_style(row):
        epr = row['epr']
        xg = row['total_xg']
        poss = row['possession_pct']
        
        # Handle no xG created
        if xg == 0 or pd.isna(epr):
            if poss > 50:
                return 'All Bark, No Bite'  # High possession, no threat
            else:
                return 'No Threat Created'  # Low possession, no threat
        
        # Normal cases
        if epr < 20:
            return 'Clinical/Direct'
        elif epr <= 40:
            return 'Balanced'
        else:
            return 'Patient Build-up'
    
    result['style'] = result.apply(classify_style, axis=1)
    
    return result[['match_id', 'team', 'possession_pct', 'total_xg', 'epr', 'style']]


def calculate_sequence_length(events, conn=None, match_id=None) -> pd.DataFrame:
    """
    Calculate average passes per possession sequence.
    
    Shows team build-up style:
    - High avg (8+) = Patient build-up (Man City, Barcelona)
    - Medium avg (4-7) = Balanced
    - Low avg (<4) = Direct (long balls, counters)
    """
    
    if isinstance(events, str):
        if conn is None:
            conn = duckdb.connect()
        
        match_filter = f"WHERE match_id = {match_id}" if match_id else ""
        
        query = f"""
        WITH possession_sequences AS (
            SELECT 
                match_id,
                team,
                possession,
                COUNT(*) FILTER (WHERE type = 'Pass') as passes_in_sequence
            FROM '{events}'
            {match_filter}
            GROUP BY match_id, team, possession
        )
        SELECT 
            match_id,
            team,
            COUNT(*) as total_sequences,
            SUM(passes_in_sequence) as total_passes,
            ROUND(AVG(passes_in_sequence), 2) as avg_passes_per_sequence,
            MAX(passes_in_sequence) as longest_sequence
        FROM possession_sequences
        WHERE passes_in_sequence > 0
        GROUP BY match_id, team
        ORDER BY match_id, avg_passes_per_sequence DESC
        """
        
        return conn.execute(query).df()
    
    else:
        df = events.copy()
        
        if match_id is not None:
            df = df[df['match_id'] == match_id]
        
        # Create unique possession identifier
        df['possession_id'] = df['match_id'].astype(str) + '_' + df['possession'].astype(str)
        
        # Count passes per possession
        possession_passes = df[df['type'] == 'Pass'].groupby(
            ['match_id', 'team', 'possession_id']
        ).size().reset_index(name='passes_in_sequence')
        
        # Filter out possessions with 0 passes
        possession_passes = possession_passes[possession_passes['passes_in_sequence'] > 0]
        
        # Calculate averages
        result = possession_passes.groupby(['match_id', 'team']).agg(
            total_sequences=('possession_id', 'count'),
            total_passes=('passes_in_sequence', 'sum'),
            avg_passes_per_sequence=('passes_in_sequence', lambda x: round(x.mean(), 2)),
            longest_sequence=('passes_in_sequence', 'max')
        ).reset_index()
        
        return result.sort_values(['match_id', 'avg_passes_per_sequence'], ascending=[True, False])


def calculate_counter_attack_speed(events, conn=None, match_id=None, max_time_window=10) -> pd.DataFrame:
    """
    Calculate counter-attack speed: distance gained per second after turnovers.
    
    Identifies fast counter-attacking teams:
    - High speed (> 10 units/sec) = Explosive counters (Klopp Liverpool, Ancelotti Real Madrid)
    - Medium speed (5-10 units/sec) = Balanced transitions
    - Low speed (< 5 units/sec) = Slow build-up after winning ball
    
    Note: Requires timestamp data. If unavailable, returns empty dataframe with note.
    """
    
    if isinstance(events, str):
        if conn is None:
            conn = duckdb.connect()
        
        match_filter = f"AND match_id = {match_id}" if match_id else ""
        
        # Check if timestamp column exists - FIX: Use direct query instead of pragma
        try:
            check_query = f"""
            SELECT COUNT(*) as has_data 
            FROM '{events}' 
            WHERE timestamp IS NOT NULL 
            LIMIT 1
            """
            has_timestamp_data = conn.execute(check_query).df()['has_data'].iloc[0] > 0
        except:
            # If timestamp column doesn't exist, query will fail
            has_timestamp_data = False
        
        if not has_timestamp_data:
            return pd.DataFrame({
                'match_id': [],
                'team': [],
                'counter_attacks': [],
                'avg_distance_gained': [],
                'avg_time_elapsed': [],
                'avg_speed_units_per_sec': [],
                'note': ['Timestamp data not available']
            })
        
        query = f"""
        WITH turnovers AS (
            SELECT 
                match_id,
                team,
                possession,
                index_num as turnover_index,
                location_x as turnover_x,
                -- Handle timestamp parsing for StatsBomb format (HH:MM:SS.mmm)
                CASE 
                    WHEN timestamp IS NOT NULL AND timestamp != '' 
                    THEN CAST(
                        CAST(SPLIT_PART(timestamp, ':', 1) AS INTEGER) * 3600 +
                        CAST(SPLIT_PART(timestamp, ':', 2) AS INTEGER) * 60 +
                        CAST(SPLIT_PART(timestamp, ':', 3) AS FLOAT)
                    AS FLOAT)
                    ELSE NULL
                END as turnover_time_seconds
            FROM '{events}'
            WHERE type IN ('Interception', 'Tackle', 'Ball Recovery')
              AND location_x IS NOT NULL
              AND timestamp IS NOT NULL
              {match_filter}
        ),
        subsequent_actions AS (
            SELECT 
                e.match_id,
                e.team,
                e.possession,
                e.index_num,
                e.location_x,
                CASE 
                    WHEN e.timestamp IS NOT NULL AND e.timestamp != '' 
                    THEN CAST(
                        CAST(SPLIT_PART(e.timestamp, ':', 1) AS INTEGER) * 3600 +
                        CAST(SPLIT_PART(e.timestamp, ':', 2) AS INTEGER) * 60 +
                        CAST(SPLIT_PART(e.timestamp, ':', 3) AS FLOAT)
                    AS FLOAT)
                    ELSE NULL
                END as action_time_seconds,
                t.turnover_x,
                t.turnover_time_seconds,
                t.turnover_index,
                (e.location_x - t.turnover_x) as distance_gained,
                CASE 
                    WHEN e.timestamp IS NOT NULL AND t.turnover_time_seconds IS NOT NULL
                    THEN CAST(
                        CAST(SPLIT_PART(e.timestamp, ':', 1) AS INTEGER) * 3600 +
                        CAST(SPLIT_PART(e.timestamp, ':', 2) AS INTEGER) * 60 +
                        CAST(SPLIT_PART(e.timestamp, ':', 3) AS FLOAT)
                    AS FLOAT) - t.turnover_time_seconds
                    ELSE NULL
                END as time_elapsed
            FROM '{events}' e
            JOIN turnovers t 
                ON e.match_id = t.match_id 
                AND e.team = t.team 
                AND e.possession = t.possession
                AND e.index_num > t.turnover_index
            WHERE e.type IN ('Pass', 'Carry', 'Shot')
              AND e.location_x IS NOT NULL
              AND e.timestamp IS NOT NULL
              {match_filter}
        )
        SELECT 
            match_id,
            team,
            COUNT(DISTINCT possession) as counter_attacks,
            ROUND(AVG(distance_gained), 2) as avg_distance_gained,
            ROUND(AVG(time_elapsed), 2) as avg_time_elapsed,
            ROUND(AVG(distance_gained / NULLIF(time_elapsed, 0)), 2) as avg_speed_units_per_sec
        FROM subsequent_actions
        WHERE distance_gained > 0 
          AND time_elapsed > 0 
          AND time_elapsed <= {max_time_window}
        GROUP BY match_id, team
        HAVING COUNT(*) > 0
        ORDER BY match_id, avg_speed_units_per_sec DESC
        """
        
        try:
            return conn.execute(query).df()
        except Exception as e:
            # If query fails (e.g., no valid data), return empty with note
            return pd.DataFrame({
                'match_id': [],
                'team': [],
                'counter_attacks': [],
                'avg_distance_gained': [],
                'avg_time_elapsed': [],
                'avg_speed_units_per_sec': [],
                'note': [f'Error: {str(e)}']
            })
    
    else:
        df = events.copy()
        
        if match_id is not None:
            df = df[df['match_id'] == match_id]
        
        # Check if timestamp exists
        if 'timestamp' not in df.columns or df['timestamp'].isna().all():
            return pd.DataFrame({
                'match_id': [],
                'team': [],
                'counter_attacks': [],
                'avg_distance_gained': [],
                'avg_time_elapsed': [],
                'avg_speed_units_per_sec': [],
                'note': ['Timestamp data not available']
            })
        
        # Create possession identifier
        df['possession_id'] = df['match_id'].astype(str) + '_' + df['possession'].astype(str)
        
        # Parse timestamp if it's string format (StatsBomb format: "00:00:12.450")
        def parse_timestamp(ts):
            if pd.isna(ts) or ts == '':
                return np.nan
            if isinstance(ts, str):
                try:
                    parts = ts.split(':')
                    hours = int(parts[0])
                    minutes = int(parts[1])
                    seconds = float(parts[2])
                    return hours * 3600 + minutes * 60 + seconds
                except:
                    return np.nan
            return ts
        
        df['timestamp_seconds'] = df['timestamp'].apply(parse_timestamp)
        
        # Find turnovers
        turnovers = df[
            df['type'].isin(['Interception', 'Tackle', 'Ball Recovery']) &
            df['location_x'].notna() &
            df['timestamp_seconds'].notna()
        ][['match_id', 'team', 'possession_id', 'index_num', 'location_x', 'timestamp_seconds']].copy()
        turnovers.columns = ['match_id', 'team', 'possession_id', 'turnover_index', 'turnover_x', 'turnover_time']
        
        # Find subsequent actions
        actions = df[
            df['type'].isin(['Pass', 'Carry', 'Shot']) &
            df['location_x'].notna() &
            df['timestamp_seconds'].notna()
        ][['match_id', 'team', 'possession_id', 'index_num', 'location_x', 'timestamp_seconds']].copy()
        
        # Merge
        counters = actions.merge(
            turnovers,
            on=['match_id', 'team', 'possession_id'],
            how='inner'
        )
        
        # Filter to actions after turnover
        counters = counters[counters['index_num'] > counters['turnover_index']]
        
        # Calculate metrics
        counters['distance_gained'] = counters['location_x'] - counters['turnover_x']
        counters['time_elapsed'] = counters['timestamp_seconds'] - counters['turnover_time']
        
        # Filter to valid counters
        counters = counters[
            (counters['distance_gained'] > 0) &
            (counters['time_elapsed'] > 0) &
            (counters['time_elapsed'] <= max_time_window)
        ]
        
        if len(counters) == 0:
            return pd.DataFrame({
                'match_id': [],
                'team': [],
                'counter_attacks': [],
                'avg_distance_gained': [],
                'avg_time_elapsed': [],
                'avg_speed_units_per_sec': []
            })
        
        # Calculate speed
        counters['speed'] = counters['distance_gained'] / counters['time_elapsed']
        
        # Aggregate
        result = counters.groupby(['match_id', 'team']).agg(
            counter_attacks=('possession_id', 'nunique'),
            avg_distance_gained=('distance_gained', lambda x: round(x.mean(), 2)),
            avg_time_elapsed=('time_elapsed', lambda x: round(x.mean(), 2)),
            avg_speed_units_per_sec=('speed', lambda x: round(x.mean(), 2))
        ).reset_index()
        
        return result.sort_values(['match_id', 'avg_speed_units_per_sec'], ascending=[True, False])


def calculate_defensive_actions_by_zone(events, conn=None, match_id=None) -> pd.DataFrame:
    """
    Calculate defensive actions (tackles, interceptions) by pitch zone.
    
    Shows defensive strategy:
    - High in attacking third (>30% of total) = High press
    - High in middle third (>50% of total) = Mid-block
    - High in defensive third (>50% of total) = Low block
    """

    if isinstance(events, str):
        if conn is None:
            conn = duckdb.connect()
        
        match_filter = f"AND match_id = {match_id}" if match_id else ""
        
        query = f"""
        SELECT 
            match_id,
            team,
            CASE 
                WHEN location_x < 40 THEN 'Defensive Third'
                WHEN location_x < 80 THEN 'Middle Third'
                ELSE 'Attacking Third'
            END as zone,
            COUNT(*) as defensive_actions
        FROM '{events}'
        WHERE type IN ('Interception', 'Tackle', 'Block', 'Clearance')
          AND location_x IS NOT NULL
          {match_filter}
        GROUP BY match_id, team, zone
        ORDER BY match_id, team, zone
        """
        
        return conn.execute(query).df()
    
    else:
        df = events.copy()
        
        if match_id is not None:
            df = df[df['match_id'] == match_id]
        
        df = df[
            df['type'].isin(['Interception', 'Tackle', 'Block', 'Clearance']) &
            df['location_x'].notna()
        ]
        
        # Add zone column
        df['zone'] = pd.cut(
            df['location_x'],
            bins=[0, 40, 80, 120],
            labels=['Defensive Third', 'Middle Third', 'Attacking Third']
        )
        
        result = df.groupby(['match_id', 'team', 'zone']).size().reset_index(name='defensive_actions')
        
        return result.sort_values(['match_id', 'team', 'zone'])
    
def calculate_team_defensive_line_height(events, conn=None, match_id=None) -> pd.DataFrame:
    """
    Calculate defensive line height metric from zone distribution.
    
    Returns match-team level with:
    - Weighted average defensive line (1=deep, 2=mid, 3=high)
    - Percentage of actions in each zone
    
    Interpretation:
    - Height > 2.0 = High press (lots of actions in attacking third)
    - Height 1.5-2.0 = Mid-block
    - Height < 1.5 = Low block (deep defense)
    """
    
    # Get zone-level data
    zone_data = calculate_defensive_actions_by_zone(events, conn, match_id)
    
    # Pivot to wide format
    zone_pivot = zone_data.pivot_table(
        index=['match_id', 'team'],
        columns='zone',
        values='defensive_actions',
        fill_value=0
    ).reset_index()
    
    # Calculate total actions
    zone_pivot['total_defensive_actions'] = (
        zone_pivot['Defensive Third'] + 
        zone_pivot['Middle Third'] + 
        zone_pivot['Attacking Third']
    )
    
    # Calculate weighted average (Defensive=1, Middle=2, Attacking=3)
    zone_pivot['defensive_line_height'] = round(
        (zone_pivot['Defensive Third'] * 1 + 
         zone_pivot['Middle Third'] * 2 + 
         zone_pivot['Attacking Third'] * 3) / 
        zone_pivot['total_defensive_actions'],
        3
    )
    
    # Calculate percentages
    zone_pivot['defensive_third_pct'] = round(
        zone_pivot['Defensive Third'] * 100.0 / zone_pivot['total_defensive_actions'], 2
    )
    zone_pivot['middle_third_pct'] = round(
        zone_pivot['Middle Third'] * 100.0 / zone_pivot['total_defensive_actions'], 2
    )
    zone_pivot['attacking_third_pct'] = round(
        zone_pivot['Attacking Third'] * 100.0 / zone_pivot['total_defensive_actions'], 2
    )
    
    # Classify defensive style
    def classify_defensive_style(row):
        if row['attacking_third_pct'] > 30:
            return 'High Press'
        elif row['middle_third_pct'] > 50:
            return 'Mid-Block'
        elif row['defensive_third_pct'] > 50:
            return 'Low Block'
        else:
            return 'Balanced'
    
    zone_pivot['defensive_style'] = zone_pivot.apply(classify_defensive_style, axis=1)
    
    # Select final columns
    result = zone_pivot[[
        'match_id', 'team', 
        'total_defensive_actions',
        'defensive_line_height',
        'defensive_third_pct', 'middle_third_pct', 'attacking_third_pct',
        'defensive_style',
        'Defensive Third', 'Middle Third', 'Attacking Third'  # Keep raw counts too
    ]]
    
    return result
    

def analyze_possession_quality(events, conn=None, match_id=None) -> pd.DataFrame:
    """
    Compare Possession % vs Field Tilt to classify possession quality.
    
    Identifies team styles:
    - The Steamroller: High possession + High tilt (dominant)
    - The U-Shape: High possession + Low tilt (sterile possession)
    - Efficiency Experts: Low possession + High tilt (direct/clinical)
    - Counter-Punchers: Low possession + Low tilt (defensive)
    
    Includes:
    - Pass efficiency (vertical vs horizontal)
    - Tilt differential (territorial gap)
    - Possession quality gap (efficiency metric)
    """
    
    # Get both metrics
    possession = calculate_possession_percentage(events, conn, match_id)
    tilt = calculate_field_tilt(events, conn, match_id)
    
    # Get pass distribution by zone for efficiency calculation
    zones = calculate_possession_by_zone(events, conn, match_id)
    
    # Extract final third passes
    final_third_passes = zones[zones['zone'] == 'Final Third'][['match_id', 'team', 'passes']].copy()
    final_third_passes.columns = ['match_id', 'team', 'final_third_passes']
    
    # Merge all data
    result = possession.merge(
        tilt[['match_id', 'team', 'field_tilt_pct']], 
        on=['match_id', 'team']
    )
    
    result = result.merge(
        final_third_passes,
        on=['match_id', 'team'],
        how='left'
    )
    
    result['final_third_passes'] = result['final_third_passes'].fillna(0)
    
    # Pass efficiency (verticality)
    # What % of passes are in the final third
    result['pass_efficiency_pct'] = round(
        result['final_third_passes'] * 100.0 / result['passes'], 2
    )
    
    # Classify verticality
    def classify_verticality(pct):
        if pct > 35:
            return 'Ultra-Vertical'
        elif pct > 28:
            return 'Vertical'
        elif pct > 22:
            return 'Balanced'
        elif pct > 18:
            return 'Patient'
        else:
            return 'Horizontal'
    
    result['verticality'] = result['pass_efficiency_pct'].apply(classify_verticality)
    
    # Tilt differential (territorial gap)
    # Calculate opponent's tilt for each team
    tilt_diff = []
    for _, row in result.iterrows():
        match_id = row['match_id']
        team = row['team']
        
        # Get opponent's tilt in same match
        opponent_tilt = result[
            (result['match_id'] == match_id) & 
            (result['team'] != team)
        ]['field_tilt_pct'].values
        
        if len(opponent_tilt) > 0:
            diff = round(row['field_tilt_pct'] - opponent_tilt[0], 2)
        else:
            diff = 0
        
        tilt_diff.append(diff)
    
    result['tilt_differential'] = tilt_diff
    
    # Classify territorial control
    def classify_territorial_control(diff):
        if diff > 15:
            return 'Total Dominance'
        elif diff > 8:
            return 'Strong Control'
        elif diff > -8:
            return 'Contested'
        elif diff > -15:
            return 'Under Pressure'
        else:
            return 'Pinned Back'
    
    result['territorial_control'] = result['tilt_differential'].apply(classify_territorial_control)

    # Possession quality gap 
    # Logic: Possession should roughly equal 2x field tilt for balanced play
    # If possession is 60% and tilt is 30%, that's balanced (60 ≈ 2*30)
    # If possession is 60% and tilt is 20%, gap is +20 (sterile possession)
    # If possession is 40% and tilt is 30%, gap is -20 (ultra-efficient)
    
    expected_possession = result['field_tilt_pct'] * 2
    result['possession_quality_gap'] = round(
        result['possession_pct'] - expected_possession, 2
    )
    
    # Gap interpretation
    def interpret_gap(gap):
        if gap > 15:
            return 'Sterile Possession'
        elif gap > 5:
            return 'Patient Build-up'
        elif gap > -5:
            return 'Balanced'
        elif gap > -15:
            return 'Direct & Efficient'
        else:
            return 'Ultra-Efficient'
    
    result['gap_interpretation'] = result['possession_quality_gap'].apply(interpret_gap)

    # Possession style 
    def classify_style(row):
        poss = row['possession_pct']
        tilt = row['field_tilt_pct']
        gap = row['possession_quality_gap']
        
        # Thresholds
        high_poss = poss > 55
        low_poss = poss < 45
        high_tilt = tilt > 30
        low_tilt = tilt < 20
        sterile = gap > 15
        efficient = gap < -15
        
        # Primary classifications based on possession + territory
        if high_poss and high_tilt:
            return 'The Steamroller'
        elif high_poss and low_tilt:
            if sterile:
                return 'The U-Shape (Sterile)'
            else:
                return 'The U-Shape'
        elif low_poss and high_tilt:
            if efficient:
                return 'Efficiency Experts (Elite)'
            else:
                return 'Efficiency Experts'
        elif low_poss and low_tilt:
            return 'Counter-Punchers'
        # Secondary classifications for balanced possession
        elif high_tilt:
            return 'Attacking'
        elif low_tilt:
            return 'Defensive'
        else:
            return 'Contested'
    
    result['possession_style'] = result.apply(classify_style, axis=1)
    
    # Reorder columns for clarity
    column_order = [
        'match_id', 'team',
        # Possession metrics
        'possession_pct', 'passes', 'total_passes',
        # Territory metrics  
        'field_tilt_pct', 'tilt_differential', 'territorial_control',
        # Efficiency metrics
        'pass_efficiency_pct', 'verticality', 'final_third_passes',
        # Quality metrics
        'possession_quality_gap', 'gap_interpretation', 'possession_style'
    ]
    
    return result[column_order].sort_values(
        ['match_id', 'possession_pct'], 
        ascending=[True, False]
    )