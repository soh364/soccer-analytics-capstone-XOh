"""Possession-based metrics: PPDA, Field Tilt"""

import pandas as pd
import duckdb
from typing import Union


def calculate_ppda(events, conn=None, match_id=None) -> pd.DataFrame:
    # Calculate PPDA (Passes Per Defensive Action) for teams
    # PPDA = opponent passes / defensive actions

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
              AND e.location_x > 48  -- Opponent's attacking 60% (120 * 0.4 = 48)
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
    # Calculate Field Tilt - percentage of play in opponent's half

    if isinstance(events, str):
        if conn is None:
            conn = duckdb.connect()
        
        match_filter = f"WHERE match_id = {match_id}" if match_id else ""
        
        query = f"""
        SELECT 
            match_id,
            team,
            COUNT(*) as total_actions,
            COUNT(*) FILTER (WHERE location_x > 60) as attacking_half_actions,
            ROUND(COUNT(*) FILTER (WHERE location_x > 60) * 100.0 / COUNT(*), 2) as field_tilt_pct
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
        
        # Calculate field tilt
        result = df.groupby(['match_id', 'team']).agg(
            total_actions=('type', 'count'),
            attacking_half_actions=('location_x', lambda x: (x > 60).sum())
        ).reset_index()
        
        result['field_tilt_pct'] = round(
            result['attacking_half_actions'] * 100.0 / result['total_actions'], 2
        )
        
        return result.sort_values(['match_id', 'field_tilt_pct'], ascending=[True, False])


def calculate_defensive_actions_by_zone(events, conn=None, match_id=None) -> pd.DataFrame:
    # Calculate defensive actions (tackles, interceptions) by pitch zone

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