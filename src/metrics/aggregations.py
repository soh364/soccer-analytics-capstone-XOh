"""Aggregation metrics: xG summaries, match statistics"""

import pandas as pd
import duckdb
from typing import Union, Optional


def aggregate_xg_by_team(events, conn=None, match_id=None) -> pd.DataFrame:
    # Aggregate xG by team and match

    if isinstance(events, str):
        if conn is None:
            conn = duckdb.connect()
        
        match_filter = f"AND match_id = {match_id}" if match_id else ""
        
        query = f"""
        SELECT 
            match_id,
            team,
            COUNT(*) as shots,
            SUM(shot_statsbomb_xg) as xg,
            ROUND(AVG(shot_statsbomb_xg), 3) as avg_xg_per_shot,
            SUM(CASE WHEN shot_outcome = 'Goal' THEN 1 ELSE 0 END) as goals
        FROM '{events}'
        WHERE type = 'Shot'
          AND shot_statsbomb_xg IS NOT NULL
          {match_filter}
        GROUP BY match_id, team
        ORDER BY match_id, xg DESC
        """
        
        return conn.execute(query).df()
    
    else:

        df = events.copy()
        
        df = df[
            (df['type'] == 'Shot') &
            (df['shot_statsbomb_xg'].notna())
        ]
        
        if match_id is not None:
            df = df[df['match_id'] == match_id]
        
        result = df.groupby(['match_id', 'team']).agg(
            shots=('type', 'count'),
            xg=('shot_statsbomb_xg', 'sum'),
            avg_xg_per_shot=('shot_statsbomb_xg', lambda x: round(x.mean(), 3)),
            goals=('shot_outcome', lambda x: (x == 'Goal').sum())
        ).reset_index()
        
        return result.sort_values(['match_id', 'xg'], ascending=[True, False])


def aggregate_xg_by_player(events, conn=None, match_id=None, min_shots=1) -> pd.DataFrame:
    # Aggregate xG by player
    # min_shots is there so we don't get a ton of 1-shot noise

    if isinstance(events, str):
        if conn is None:
            conn = duckdb.connect()
        
        match_filter = f"AND match_id = {match_id}" if match_id else ""
        
        query = f"""
        SELECT 
            player,
            team,
            COUNT(DISTINCT match_id) as matches_played,
            COUNT(*) as shots,
            SUM(shot_statsbomb_xg) as xg,
            ROUND(AVG(shot_statsbomb_xg), 3) as avg_xg_per_shot,
            SUM(CASE WHEN shot_outcome = 'Goal' THEN 1 ELSE 0 END) as goals,
            ROUND(SUM(CASE WHEN shot_outcome = 'Goal' THEN 1 ELSE 0 END)::FLOAT / 
                  SUM(shot_statsbomb_xg), 3) as goals_vs_xg_ratio
        FROM '{events}'
        WHERE type = 'Shot'
          AND shot_statsbomb_xg IS NOT NULL
          AND player IS NOT NULL
          {match_filter}
        GROUP BY player, team
        HAVING COUNT(*) >= {min_shots}
        ORDER BY xg DESC
        """
        
        return conn.execute(query).df()
    
    else:

        df = events.copy()
        
        df = df[
            (df['type'] == 'Shot') &
            (df['shot_statsbomb_xg'].notna()) &
            (df['player'].notna())
        ]
        
        if match_id is not None:
            df = df[df['match_id'] == match_id]
        
        result = df.groupby(['player', 'team']).agg(
            matches_played=('match_id', 'nunique'),
            shots=('type', 'count'),
            xg=('shot_statsbomb_xg', 'sum'),
            avg_xg_per_shot=('shot_statsbomb_xg', lambda x: round(x.mean(), 3)),
            goals=('shot_outcome', lambda x: (x == 'Goal').sum())
        ).reset_index()
        
        # Calculate goals vs xG ratio
        result['goals_vs_xg_ratio'] = round(result['goals'] / result['xg'], 3)
        
        # Filter out tiny sample sizes by min_shots
        result = result[result['shots'] >= min_shots]
        
        return result.sort_values('xg', ascending=False)


def get_match_summary(events, matches, conn=None, match_id=None) -> pd.DataFrame:
    # Generate match summary statistics: shots, xG, passing, and a rough possession proxy

    if isinstance(events, str):
        if conn is None:
            conn = duckdb.connect()
        
        query = f"""
        WITH match_stats AS (
            SELECT 
                e.match_id,
                e.team,
                COUNT(DISTINCT CASE WHEN e.type = 'Shot' THEN e.id END) as shots,
                SUM(CASE WHEN e.type = 'Shot' THEN e.shot_statsbomb_xg ELSE 0 END) as xg,
                COUNT(DISTINCT CASE WHEN e.type = 'Pass' THEN e.id END) as passes,
                COUNT(DISTINCT CASE WHEN e.type = 'Pass' AND e.pass_outcome IS NULL THEN e.id END) as completed_passes,
                COUNT(DISTINCT CASE WHEN e.type = 'Carry' THEN e.id END) as carries
            FROM '{events}' e
            WHERE e.match_id = {match_id}
            GROUP BY e.match_id, e.team
        ),
        possession_est AS (
            SELECT 
                e.match_id,
                e.team,
                COUNT(*) as touches
            FROM '{events}' e
            WHERE e.match_id = {match_id}
              AND e.type IN ('Pass', 'Carry', 'Dribble')
            GROUP BY e.match_id, e.team
        )
        SELECT 
            m.match_id,
            m.match_date,
            m.home_team,
            m.away_team,
            m.home_score,
            m.away_score,
            ms.team,
            ms.shots,
            ROUND(ms.xg, 2) as xg,
            ms.passes,
            ms.completed_passes,
            ROUND(ms.completed_passes * 100.0 / NULLIF(ms.passes, 0), 1) as pass_completion_pct,
            ROUND(p.touches * 100.0 / SUM(p.touches) OVER (PARTITION BY p.match_id), 1) as possession_est_pct
        FROM '{matches}' m
        JOIN match_stats ms ON m.match_id = ms.match_id
        LEFT JOIN possession_est p ON ms.match_id = p.match_id AND ms.team = p.team
        WHERE m.match_id = {match_id}
        ORDER BY ms.team
        """
        
        return conn.execute(query).df()
    
    else:

        events_df = events[events['match_id'] == match_id].copy()
        match_info = matches[matches['match_id'] == match_id].iloc[0]
        
        results = []
        
        for team in events_df['team'].unique():
            team_events = events_df[events_df['team'] == team]
            
            shots = team_events[team_events['type'] == 'Shot']
            passes = team_events[team_events['type'] == 'Pass']
            
            # Calculate stats
            stats = {
                'match_id': match_id,
                'match_date': match_info['match_date'],
                'home_team': match_info['home_team'],
                'away_team': match_info['away_team'],
                'home_score': match_info['home_score'],
                'away_score': match_info['away_score'],
                'team': team,
                'shots': len(shots),
                'xg': round(shots['shot_statsbomb_xg'].sum(), 2),
                'passes': len(passes),
                'completed_passes': len(passes[passes['pass_outcome'].isna()]),
            }
            
            stats['pass_completion_pct'] = round(
                stats['completed_passes'] * 100.0 / stats['passes'] if stats['passes'] > 0 else 0, 1
            )
            
            results.append(stats)
        
        result_df = pd.DataFrame(results)
        
        # Calculate possession estimate
        touches = events_df[events_df['type'].isin(['Pass', 'Carry', 'Dribble'])].groupby('team').size()
        total_touches = touches.sum()
        result_df['possession_est_pct'] = result_df['team'].map(
            lambda t: round(touches.get(t, 0) * 100.0 / total_touches, 1)
        )
        
        return result_df


def calculate_pass_completion_by_zone(events, conn=None, match_id=None) -> pd.DataFrame:
    # Calculate pass completion percentage by pitch zone

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
            COUNT(*) as total_passes,
            COUNT(*) FILTER (WHERE pass_outcome IS NULL) as completed_passes,
            ROUND(COUNT(*) FILTER (WHERE pass_outcome IS NULL) * 100.0 / COUNT(*), 2) as completion_pct
        FROM '{events}'
        WHERE type = 'Pass'
          AND location_x IS NOT NULL
          {match_filter}
        GROUP BY match_id, team, zone
        ORDER BY match_id, team, zone
        """
        
        return conn.execute(query).df()
    
    else:

        df = events.copy()
        
        df = df[
            (df['type'] == 'Pass') &
            (df['location_x'].notna())
        ]
        
        if match_id is not None:
            df = df[df['match_id'] == match_id]
        
        # Add zone
        df['zone'] = pd.cut(
            df['location_x'],
            bins=[0, 40, 80, 120],
            labels=['Defensive Third', 'Middle Third', 'Attacking Third']
        )
        
        result = df.groupby(['match_id', 'team', 'zone']).agg(
            total_passes=('type', 'count'),
            completed_passes=('pass_outcome', lambda x: x.isna().sum())
        ).reset_index()
        
        result['completion_pct'] = round(
            result['completed_passes'] * 100.0 / result['total_passes'], 2
        )
        
        return result.sort_values(['match_id', 'team', 'zone'])