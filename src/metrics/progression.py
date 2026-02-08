"""Ball progression metrics: Progressive Passes, Progressive Carries"""

import pandas as pd
import duckdb
from typing import Union, Optional


def calculate_progressive_passes(events, conn=None, match_id=None, player=None) -> pd.DataFrame:
    # Progressive pass heuristic:
    # 1. Completed pass into the box (end_x >= 102 and 18 <= end_y <= 62), OR
    # 2. Gains >= 10 in x-direction, and the pass doesn't start too deep (start_x <= 72)
    
    if isinstance(events, str):
        if conn is None:
            conn = duckdb.connect()
        
        filters = ["e.type = 'Pass'", "e.pass_outcome IS NULL"]  # Completed passes only
        
        if match_id is not None:
            filters.append(f"e.match_id = {match_id}")
        if player is not None:
            filters.append(f"e.player = '{player}'")
        
        where_clause = " AND ".join(filters)
        
        query = f"""
        WITH progressive_passes_calc AS (
            SELECT 
                e.match_id,
                e.team,
                e.player,
                e.location_x,
                e.location_y,
                e.pass_end_location_x,
                e.pass_end_location_y,
                (e.pass_end_location_x - e.location_x) as yards_forward,
                CASE 
                    WHEN e.pass_end_location_x >= 102 
                         AND e.pass_end_location_y BETWEEN 18 AND 62 THEN 1
                    WHEN (e.pass_end_location_x - e.location_x) >= 10 
                         AND e.location_x <= 72 THEN 1
                    ELSE 0
                END as is_progressive
            FROM '{events}' e
            WHERE {where_clause}
              AND e.location_x IS NOT NULL
              AND e.pass_end_location_x IS NOT NULL
        )
        SELECT 
            match_id,
            team,
            player,
            COUNT(*) as total_passes,
            SUM(is_progressive) as progressive_passes,
            ROUND(SUM(is_progressive) * 100.0 / COUNT(*), 2) as progressive_pass_pct,
            ROUND(AVG(CASE WHEN is_progressive = 1 THEN yards_forward END), 2) as avg_progressive_distance
        FROM progressive_passes_calc
        GROUP BY match_id, team, player
        HAVING SUM(is_progressive) > 0
        ORDER BY progressive_passes DESC
        """
        
        return conn.execute(query).df()
    
    else:

        df = events.copy()
        
        # Filter to completed passes
        df = df[
            (df['type'] == 'Pass') &
            (df['pass_outcome'].isna()) &  # Completed passes
            (df['location_x'].notna()) &
            (df['pass_end_location_x'].notna())
        ]
        
        if match_id is not None:
            df = df[df['match_id'] == match_id]
        if player is not None:
            df = df[df['player'] == player]
        
        # Calculate yards forward
        df['yards_forward'] = df['pass_end_location_x'] - df['location_x']
        
        # Determine if progressive
        df['is_progressive'] = (
            # Into penalty area
            ((df['pass_end_location_x'] >= 102) & 
             (df['pass_end_location_y'] >= 18) & 
             (df['pass_end_location_y'] <= 62)) |
            # Or at least 10 yards forward from non-defending third
            ((df['yards_forward'] >= 10) & (df['location_x'] <= 72))
        ).astype(int)
        
        # Aggregate by player
        result = df.groupby(['match_id', 'team', 'player']).agg(
            total_passes=('type', 'count'),
            progressive_passes=('is_progressive', 'sum'),
            avg_progressive_distance=('yards_forward', lambda x: round(x[df.loc[x.index, 'is_progressive'] == 1].mean(), 2))
        ).reset_index()
        
        result['progressive_pass_pct'] = round(
            result['progressive_passes'] * 100.0 / result['total_passes'], 2
        )
        
        # Filter to only players with at least one progressive pass
        result = result[result['progressive_passes'] > 0]
        
        return result.sort_values('progressive_passes', ascending=False)


def calculate_progressive_carries(events, conn=None, match_id=None, player=None) -> pd.DataFrame:
    # Progressive carry heuristic:
    # 1. Carry into the box (end_x >= 102 and 18 <= end_y <= 62), OR
    # 2. Gains >= 10 in x-direction and ends past x > 60 (so you don't count carries in deep zones)

    if isinstance(events, str):
        if conn is None:
            conn = duckdb.connect()
        
        filters = ["e.type = 'Carry'"]
        
        if match_id is not None:
            filters.append(f"e.match_id = {match_id}")
        if player is not None:
            filters.append(f"e.player = '{player}'")
        
        where_clause = " AND ".join(filters)
        
        query = f"""
        WITH progressive_carries_calc AS (
            SELECT 
                e.match_id,
                e.team,
                e.player,
                e.location_x,
                e.location_y,
                e.carry_end_location_x,
                e.carry_end_location_y,
                (e.carry_end_location_x - e.location_x) as yards_forward,
                CASE 
                    WHEN e.carry_end_location_x >= 102 
                         AND e.carry_end_location_y BETWEEN 18 AND 62 THEN 1
                    WHEN (e.carry_end_location_x - e.location_x) >= 10 
                         AND e.carry_end_location_x > 60 THEN 1
                    ELSE 0
                END as is_progressive
            FROM '{events}' e
            WHERE {where_clause}
              AND e.location_x IS NOT NULL
              AND e.carry_end_location_x IS NOT NULL
        )
        SELECT 
            match_id,
            team,
            player,
            COUNT(*) as total_carries,
            SUM(is_progressive) as progressive_carries,
            ROUND(SUM(is_progressive) * 100.0 / COUNT(*), 2) as progressive_carry_pct,
            ROUND(AVG(CASE WHEN is_progressive = 1 THEN yards_forward END), 2) as avg_progressive_distance
        FROM progressive_carries_calc
        GROUP BY match_id, team, player
        HAVING SUM(is_progressive) > 0
        ORDER BY progressive_carries DESC
        """
        
        return conn.execute(query).df()
    
    else:

        df = events.copy()
        
        # Filter to carries
        df = df[
            (df['type'] == 'Carry') &
            (df['location_x'].notna()) &
            (df['carry_end_location_x'].notna())
        ]
        
        if match_id is not None:
            df = df[df['match_id'] == match_id]
        if player is not None:
            df = df[df['player'] == player]
        
        # Calculate yards forward
        df['yards_forward'] = df['carry_end_location_x'] - df['location_x']
        
        # Determine if progressive
        df['is_progressive'] = (
            # Into penalty area
            ((df['carry_end_location_x'] >= 102) & 
             (df['carry_end_location_y'] >= 18) & 
             (df['carry_end_location_y'] <= 62)) |
            # Or at least 10 yards forward and not ending in defending half
            ((df['yards_forward'] >= 10) & (df['carry_end_location_x'] > 60))
        ).astype(int)
        
        # Aggregate by player
        result = df.groupby(['match_id', 'team', 'player']).agg(
            total_carries=('type', 'count'),
            progressive_carries=('is_progressive', 'sum'),
            avg_progressive_distance=('yards_forward', lambda x: round(x[df.loc[x.index, 'is_progressive'] == 1].mean(), 2))
        ).reset_index()
        
        result['progressive_carry_pct'] = round(
            result['progressive_carries'] * 100.0 / result['total_carries'], 2
        )
        
        # Filter to only players with at least one progressive carry
        result = result[result['progressive_carries'] > 0]
        
        return result.sort_values('progressive_carries', ascending=False)


def calculate_progressive_actions(events, conn=None, match_id=None, player=None) -> pd.DataFrame:
    # Combined progressive actions = progressive passes + progressive carries 
    
    # Get progressive passes
    prog_passes = calculate_progressive_passes(events, conn, match_id, player)
    
    # Get progressive carries
    prog_carries = calculate_progressive_carries(events, conn, match_id, player)
    
    # Merge
    result = prog_passes.merge(
        prog_carries[['match_id', 'player', 'progressive_carries']],
        on=['match_id', 'player'],
        how='outer',
        suffixes=('_pass', '_carry')
    ).fillna(0)
    
    result['progressive_actions'] = (
        result['progressive_passes'] + result['progressive_carries']
    )
    
    # Sort by total progressive actions
    result = result.sort_values('progressive_actions', ascending=False)
    
    return result[['match_id', 'team', 'player', 'progressive_passes', 
                   'progressive_carries', 'progressive_actions']]