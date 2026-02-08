"""xG Chain and xG Buildup metrics - credits all players in goal-scoring possessions"""

import pandas as pd
import duckdb
from typing import Union, Optional


def calculate_xg_chain(events, conn=None, match_id=None, min_touches=1) -> pd.DataFrame:
    # xG Chain:
    # - find possessions that contain at least one shot
    # - assign the possession xG to every player who had a "touch" during that possession

    if isinstance(events, str):
        if conn is None:
            conn = duckdb.connect()
        
        match_filter = f"AND match_id = {match_id}" if match_id else ""
        
        query = f"""
        WITH shot_possessions AS (
            -- Find all possessions that ended in a shot
            SELECT DISTINCT
                match_id,
                possession,
                team,
                MAX(CASE WHEN type = 'Shot' THEN shot_statsbomb_xg ELSE 0 END) as possession_xg,
                SUM(CASE WHEN type = 'Shot' THEN 1 ELSE 0 END) as shots_in_possession,
                SUM(CASE WHEN type = 'Shot' AND shot_outcome = 'Goal' THEN 1 ELSE 0 END) as goals_in_possession
            FROM '{events}'
            WHERE possession IS NOT NULL
              {match_filter}
            GROUP BY match_id, possession, team
            HAVING SUM(CASE WHEN type = 'Shot' THEN 1 ELSE 0 END) > 0
        ),
        player_touches_in_shot_possessions AS (
            -- Find all players who touched the ball in these possessions
            SELECT 
                e.match_id,
                e.possession,
                e.team,
                e.player,
                COUNT(*) as touches_in_possession
            FROM '{events}' e
            INNER JOIN shot_possessions sp 
                ON e.match_id = sp.match_id 
                AND e.possession = sp.possession
                AND e.team = sp.team
            WHERE e.player IS NOT NULL
              AND e.type IN ('Pass', 'Carry', 'Dribble', 'Shot')
            GROUP BY e.match_id, e.possession, e.team, e.player
        )
        SELECT 
            pt.player,
            pt.team,
            COUNT(DISTINCT pt.possession) as possessions_with_shot,
            ROUND(SUM(sp.possession_xg), 3) as xg_chain,
            SUM(sp.shots_in_possession) as shots_in_chain,
            SUM(sp.goals_in_possession) as goals_in_chain,
            ROUND(AVG(sp.possession_xg), 3) as avg_xg_per_possession,
            SUM(pt.touches_in_possession) as total_touches_in_chains
        FROM player_touches_in_shot_possessions pt
        INNER JOIN shot_possessions sp 
            ON pt.match_id = sp.match_id 
            AND pt.possession = sp.possession
            AND pt.team = sp.team
        GROUP BY pt.player, pt.team
        HAVING SUM(pt.touches_in_possession) >= {min_touches}
        ORDER BY xg_chain DESC
        """
        
        return conn.execute(query).df()
    
    else:

        df = events.copy()
        
        if match_id is not None:
            df = df[df['match_id'] == match_id]
        
        # Step 1: Find possessions with shots
        shot_possessions = df[df['type'] == 'Shot'].groupby(['match_id', 'possession', 'team']).agg(
            possession_xg=('shot_statsbomb_xg', 'sum'),
            shots_in_possession=('type', 'count'),
            goals_in_possession=('shot_outcome', lambda x: (x == 'Goal').sum())
        ).reset_index()
        
        # Step 2: Find all players who touched ball in these possessions
        relevant_events = df[
            (df['player'].notna()) &
            (df['type'].isin(['Pass', 'Carry', 'Dribble', 'Shot']))
        ].copy()
        
        # Merge with shot possessions
        player_touches = relevant_events.merge(
            shot_possessions[['match_id', 'possession', 'team']],
            on=['match_id', 'possession', 'team'],
            how='inner'
        )
        
        # Count touches per player per possession
        touches_per_possession = player_touches.groupby(
            ['match_id', 'possession', 'team', 'player']
        ).size().reset_index(name='touches_in_possession')
        
        # Merge with xG values
        result = touches_per_possession.merge(
            shot_possessions,
            on=['match_id', 'possession', 'team']
        )
        
        # Aggregate by player
        final = result.groupby(['player', 'team']).agg(
            possessions_with_shot=('possession', 'nunique'),
            xg_chain=('possession_xg', 'sum'),
            shots_in_chain=('shots_in_possession', 'sum'),
            goals_in_chain=('goals_in_possession', 'sum'),
            avg_xg_per_possession=('possession_xg', 'mean'),
            total_touches_in_chains=('touches_in_possession', 'sum')
        ).reset_index()
        
        # Round
        final['xg_chain'] = final['xg_chain'].round(3)
        final['avg_xg_per_possession'] = final['avg_xg_per_possession'].round(3)
        
        # Filter by min touches
        final = final[final['total_touches_in_chains'] >= min_touches]
        
        return final.sort_values('xg_chain', ascending=False)


def calculate_xg_buildup(events, conn=None, match_id=None, min_touches=1) -> pd.DataFrame:
    # xG Buildup:
    # - start from shot events with xG
    # - find all earlier touches in the possession
    # - exclude the shooter and (if present) the assister (key pass)

    if isinstance(events, str):
        if conn is None:
            conn = duckdb.connect()
        
        match_filter = f"AND match_id = {match_id}" if match_id else ""
        
        query = f"""
        WITH shot_events AS (
            -- Get all shots with their details
            SELECT 
                match_id,
                possession,
                team,
                player as shooter,
                shot_statsbomb_xg,
                shot_key_pass_id  -- ID of the assist pass (if any)
            FROM '{events}'
            WHERE type = 'Shot'
              AND shot_statsbomb_xg IS NOT NULL
              {match_filter}
        ),
        assist_passes AS (
            -- Find the assister (player who made key pass)
            SELECT DISTINCT
                se.match_id,
                se.possession,
                se.team,
                e.player as assister
            FROM shot_events se
            INNER JOIN '{events}' e 
                ON se.match_id = e.match_id 
                AND se.shot_key_pass_id = e.id
            WHERE se.shot_key_pass_id IS NOT NULL
        ),
        buildup_touches AS (
            -- Find players who touched ball BEFORE shot (exclude shooter & assister)
            SELECT 
                e.match_id,
                e.possession,
                e.team,
                e.player,
                COUNT(*) as touches_in_buildup
            FROM '{events}' e
            INNER JOIN shot_events se 
                ON e.match_id = se.match_id 
                AND e.possession = se.possession
                AND e.team = se.team
            WHERE e.player IS NOT NULL
              AND e.type IN ('Pass', 'Carry', 'Dribble')
              AND e.player != se.shooter  -- Not the shooter
              AND NOT EXISTS (
                  -- Not the assister
                  SELECT 1 FROM assist_passes ap
                  WHERE ap.match_id = e.match_id
                    AND ap.possession = e.possession
                    AND ap.team = e.team
                    AND ap.assister = e.player
              )
            GROUP BY e.match_id, e.possession, e.team, e.player
        )
        SELECT 
            bt.player,
            bt.team,
            COUNT(DISTINCT bt.possession) as possessions_with_buildup,
            ROUND(SUM(se.shot_statsbomb_xg), 3) as xg_buildup,
            ROUND(AVG(se.shot_statsbomb_xg), 3) as avg_xg_per_possession,
            SUM(bt.touches_in_buildup) as total_touches_in_buildup
        FROM buildup_touches bt
        INNER JOIN shot_events se 
            ON bt.match_id = se.match_id 
            AND bt.possession = se.possession
            AND bt.team = se.team
        GROUP BY bt.player, bt.team
        HAVING SUM(bt.touches_in_buildup) >= {min_touches}
        ORDER BY xg_buildup DESC
        """
        
        return conn.execute(query).df()
    
    else:

        df = events.copy()
        
        if match_id is not None:
            df = df[df['match_id'] == match_id]
        
        # Get shots
        shots = df[
            (df['type'] == 'Shot') & 
            (df['shot_statsbomb_xg'].notna())
        ][['match_id', 'possession', 'team', 'player', 'shot_statsbomb_xg', 'shot_key_pass_id']].copy()
        shots.rename(columns={'player': 'shooter'}, inplace=True)
        
        # Get assisters
        assisters = shots[shots['shot_key_pass_id'].notna()].merge(
            df[['id', 'player']],
            left_on='shot_key_pass_id',
            right_on='id',
            how='left'
        )[['match_id', 'possession', 'team', 'player']].drop_duplicates()
        assisters.rename(columns={'player': 'assister'}, inplace=True)
        
        # Get all touches in shot possessions (excluding shooters and assisters)
        touches = df[
            (df['player'].notna()) &
            (df['type'].isin(['Pass', 'Carry', 'Dribble']))
        ][['match_id', 'possession', 'team', 'player']].copy()
        
        # Merge with shots to get xG
        buildup = touches.merge(shots, on=['match_id', 'possession', 'team'])
        
        # Exclude shooters
        buildup = buildup[buildup['player'] != buildup['shooter']]
        
        # Exclude assisters
        buildup = buildup.merge(
            assisters,
            on=['match_id', 'possession', 'team'],
            how='left'
        )
        buildup = buildup[buildup['player'] != buildup['assister']]
        
        # Count touches
        touch_counts = buildup.groupby(
            ['match_id', 'possession', 'team', 'player']
        ).size().reset_index(name='touches_in_buildup')
        
        # Merge back with xG
        buildup = touch_counts.merge(
            buildup[['match_id', 'possession', 'team', 'shot_statsbomb_xg']].drop_duplicates(),
            on=['match_id', 'possession', 'team']
        )
        
        # Aggregate
        result = buildup.groupby(['player', 'team']).agg(
            possessions_with_buildup=('possession', 'nunique'),
            xg_buildup=('shot_statsbomb_xg', 'sum'),
            avg_xg_per_possession=('shot_statsbomb_xg', 'mean'),
            total_touches_in_buildup=('touches_in_buildup', 'sum')
        ).reset_index()
        
        result['xg_buildup'] = result['xg_buildup'].round(3)
        result['avg_xg_per_possession'] = result['avg_xg_per_possession'].round(3)
        
        # Filter
        result = result[result['total_touches_in_buildup'] >= min_touches]
        
        return result.sort_values('xg_buildup', ascending=False)


def compare_xg_chain_vs_buildup(events, conn=None, match_id=None) -> pd.DataFrame:
    # Mostly a wrapper so it's easy to inspect roles:
    # ratio close to 1 => player mostly credited via buildup (deep involvement)
    # ratio close to 0 => player mostly credited via late involvement (finisher/assister)

    xg_chain = calculate_xg_chain(events, conn, match_id)
    xg_buildup = calculate_xg_buildup(events, conn, match_id)
    
    # Merge
    comparison = xg_chain.merge(
        xg_buildup[['player', 'team', 'xg_buildup', 'possessions_with_buildup']],
        on=['player', 'team'],
        how='outer'
    ).fillna(0)
    
    # Guard against divide-by-zero (outer merge can create rows with xg_chain == 0)
    comparison['buildup_to_chain_ratio'] = (
        comparison['xg_buildup'] / comparison['xg_chain']
    ).fillna(0).round(3)
    
    # Classify player type
    def classify_role(row):
        if row['xg_chain'] < 2:
            return 'Limited Involvement'
        elif row['buildup_to_chain_ratio'] > 0.7:
            return 'Deep Playmaker'
        elif row['buildup_to_chain_ratio'] < 0.3:
            return 'Finisher/Assister'
        else:
            return 'Complete Attacker'
    
    comparison['player_role'] = comparison.apply(classify_role, axis=1)
    
    return comparison.sort_values('xg_chain', ascending=False)