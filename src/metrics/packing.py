"""Packing metric - counts opponents bypassed by passes using 360 tracking data"""

import pandas as pd
import duckdb
import numpy as np
from typing import Union, Optional


def calculate_packing(events, three_sixty, conn=None, match_id=None, min_passes=5) -> pd.DataFrame:
    # Packing: number of opponents "eliminated" by a pass
    # An opponent is "packed" if they are inside the rectangle formed by the pass start and end points (x/y min-max box)

    if isinstance(events, str):
        if conn is None:
            conn = duckdb.connect()
        
        match_filter = f"AND e.match_id = {match_id}" if match_id else ""
        
        query = f"""
        WITH completed_passes AS (
            -- Get all completed passes with their locations
            SELECT 
                e.id as event_id,
                e.match_id,
                e.team,
                e.player,
                e.location_x as pass_start_x,
                e.location_y as pass_start_y,
                e.pass_end_location_x as pass_end_x,
                e.pass_end_location_y as pass_end_y,
                CASE 
                    WHEN (e.pass_end_location_x - e.location_x) >= 10 
                         OR (e.pass_end_location_x >= 102 AND e.pass_end_location_y BETWEEN 18 AND 62)
                    THEN 1 ELSE 0 
                END as is_progressive
            FROM '{events}' e
            WHERE e.type = 'Pass'
              AND e.pass_outcome IS NULL  -- Completed passes only
              AND e.location_x IS NOT NULL
              AND e.pass_end_location_x IS NOT NULL
              {match_filter}
        ),
        opponent_positions AS (
            -- Get opponent positions at moment of each pass
            SELECT 
                t.event_uuid,
                t.location_x as opponent_x,
                t.location_y as opponent_y
            FROM '{three_sixty}' t
            WHERE t.teammate = FALSE  -- Only opponents
              AND t.actor = FALSE     -- Not the passer
              AND t.keeper = FALSE    -- Not the goalkeeper
        ),
        packing_calculation AS (
            -- Calculate if opponent is between pass start and end
            SELECT 
                cp.event_id,
                cp.match_id,
                cp.team,
                cp.player,
                cp.is_progressive,
                COUNT(DISTINCT op.opponent_x || '_' || op.opponent_y) as opponents_packed
            FROM completed_passes cp
            LEFT JOIN opponent_positions op ON cp.event_id = op.event_uuid
            WHERE op.opponent_x IS NULL  -- No 360 data
               OR (
                   -- Opponent is "between" start and end
                   -- Simplified: check if opponent is in rectangular zone between pass points
                   op.opponent_x BETWEEN LEAST(cp.pass_start_x, cp.pass_end_x) 
                                     AND GREATEST(cp.pass_start_x, cp.pass_end_x)
                   AND op.opponent_y BETWEEN LEAST(cp.pass_start_y, cp.pass_end_y) 
                                         AND GREATEST(cp.pass_start_y, cp.pass_end_y)
               )
            GROUP BY cp.event_id, cp.match_id, cp.team, cp.player, cp.is_progressive
        )
        SELECT 
            player,
            team,
            COUNT(*) as total_passes,
            SUM(opponents_packed) as total_opponents_packed,
            ROUND(AVG(opponents_packed), 2) as avg_packing_per_pass,
            SUM(is_progressive) as progressive_passes,
            ROUND(
                AVG(CASE WHEN is_progressive = 1 THEN opponents_packed END), 2
            ) as avg_packing_progressive
        FROM packing_calculation
        GROUP BY player, team
        HAVING COUNT(*) >= {min_passes}
        ORDER BY avg_packing_per_pass DESC
        """
        
        return conn.execute(query).df()
    
    else:

        events_df = events.copy()
        three_sixty_df = three_sixty.copy()
        
        if match_id is not None:
            events_df = events_df[events_df['match_id'] == match_id]

        # Get completed passes with valid start/end locations
        passes = events_df[
            (events_df['type'] == 'Pass') &
            (events_df['pass_outcome'].isna()) &
            (events_df['location_x'].notna()) &
            (events_df['pass_end_location_x'].notna())
        ].copy()
        
        # Mark progressive passes (simple heuristic)
        passes['is_progressive'] = (
            ((passes['pass_end_location_x'] - passes['location_x']) >= 10) |
            ((passes['pass_end_location_x'] >= 102) & 
             (passes['pass_end_location_y'] >= 18) & 
             (passes['pass_end_location_y'] <= 62))
        ).astype(int)
        
        # Opponents only (no teammate/actor/keeper) for each pass 
        opponents = three_sixty_df[
            (three_sixty_df['teammate'] == False) &
            (three_sixty_df['actor'] == False) &
            (three_sixty_df['keeper'] == False)
        ].copy()
        
        # Calculate packing for each pass
        packing_results = []
        
        for _, pass_event in passes.iterrows():
            event_id = pass_event['id']
            
            # Get opponents at this moment
            pass_opponents = opponents[opponents['event_uuid'] == event_id]
            
            if len(pass_opponents) == 0:
                # No 360 data for this pass
                packed = 0
            else:
                # Check which opponents are "between" pass start and end
                # Simplified: rectangular zone between two points
                start_x, start_y = pass_event['location_x'], pass_event['location_y']
                end_x, end_y = pass_event['pass_end_location_x'], pass_event['pass_end_location_y']
                
                min_x, max_x = min(start_x, end_x), max(start_x, end_x)
                min_y, max_y = min(start_y, end_y), max(start_y, end_y)
                
                # Count opponents in this zone
                packed = len(pass_opponents[
                    (pass_opponents['location_x'] >= min_x) &
                    (pass_opponents['location_x'] <= max_x) &
                    (pass_opponents['location_y'] >= min_y) &
                    (pass_opponents['location_y'] <= max_y)
                ])
            
            packing_results.append({
                'player': pass_event['player'],
                'team': pass_event['team'],
                'is_progressive': pass_event['is_progressive'],
                'opponents_packed': packed
            })
        
        packing_df = pd.DataFrame(packing_results)

        # Aggregate to player/team level
        result = packing_df.groupby(['player', 'team']).agg(
            total_passes=('is_progressive', 'count'),
            total_opponents_packed=('opponents_packed', 'sum'),
            avg_packing_per_pass=('opponents_packed', 'mean'),
            progressive_passes=('is_progressive', 'sum'),
            avg_packing_progressive=('opponents_packed', lambda x: x[packing_df.loc[x.index, 'is_progressive'] == 1].mean())
        ).reset_index()
        
        result['avg_packing_per_pass'] = result['avg_packing_per_pass'].round(2)
        result['avg_packing_progressive'] = result['avg_packing_progressive'].round(2)
        
        # Filter
        result = result[result['total_passes'] >= min_passes]
        
        return result.sort_values('avg_packing_per_pass', ascending=False)


def calculate_packing_by_zone(events, three_sixty, conn=None, match_id=None) -> pd.DataFrame:
    # Calculate packing broken down by pitch zone (thirds): similar to calculate_packing but group by zone
    
    if isinstance(events, str):
        if conn is None:
            conn = duckdb.connect()

        # For now, return a placeholder - full implementation similar to above but with zone classification
        return pd.DataFrame({
            'player': [],
            'team': [],
            'zone': [],
            'passes': [],
            'avg_packing': []
        })
    
    else:
        # TODO: Implement Pandas version if needed
        return pd.DataFrame()


def compare_packing_vs_progression(events, three_sixty, conn=None, match_id=None) -> pd.DataFrame:
    # Compare packing vs progressive passes to find different passer types.

    # Import progressive passes function
    from .progression import calculate_progressive_passes
    
    # Calculate both metrics
    packing = calculate_packing(events, three_sixty, conn, match_id)
    prog_passes = calculate_progressive_passes(events, conn, match_id)
    
    # Merge
    comparison = packing.merge(
        prog_passes[['player', 'team', 'progressive_passes', 'progressive_pass_pct']],
        on=['player', 'team'],
        how='outer',
        suffixes=('_packing', '_prog')
    ).fillna(0)
    
    # Classify passer type
    def classify_passer_type(row):
        if row['avg_packing_per_pass'] > 2 and row['progressive_pass_pct'] > 30:
            return 'Elite Line-Breaker'
        elif row['progressive_pass_pct'] > 30:
            return 'Safe Progressor'
        elif row['avg_packing_per_pass'] > 2:
            return 'Short Incisive'
        else:
            return 'Conservative'
    
    comparison['passer_type'] = comparison.apply(classify_passer_type, axis=1)
    
    return comparison.sort_values('avg_packing_per_pass', ascending=False)