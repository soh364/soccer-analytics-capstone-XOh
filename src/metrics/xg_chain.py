"""xG Chain and xG Buildup metrics - credits all players in goal-scoring possessions

- Per-90 normalization for fair comparison
- Max xG per possession (avoids double-counting scrambles)
- Team involvement % (player's share of team threat)
- Context-aware role classification (single match vs season)
"""

import pandas as pd
import numpy as np
import duckdb
from typing import Union, Optional


def calculate_xg_chain(events, conn=None, match_id=None, min_touches=1, per_90=True) -> pd.DataFrame:
    # xG Chain: Total xG of possessions where player had a touch

    if isinstance(events, str):
        if conn is None:
            conn = duckdb.connect()
        
        match_filter = f"AND e.match_id = {match_id}" if match_id else ""
        
        query = f"""
        WITH shot_possessions AS (
            SELECT 
                match_id, possession, team,
                MAX(shot_statsbomb_xg) as possession_xg,
                COUNT(*) as shots_in_possession,
                SUM(CASE WHEN shot_outcome = 'Goal' THEN 1 ELSE 0 END) as goals_in_possession
            FROM '{events}'
            WHERE type = 'Shot' AND shot_statsbomb_xg IS NOT NULL {match_filter.replace('e.', '')}
            GROUP BY match_id, possession, team
        ),
        player_touches_in_shot_possessions AS (
            SELECT 
                e.match_id, e.possession, e.team, e.player,
                COUNT(DISTINCT e.id) as touches_in_possession
            FROM '{events}' e
            INNER JOIN shot_possessions sp ON e.match_id = sp.match_id AND e.possession = sp.possession AND e.team = sp.team
            WHERE e.player IS NOT NULL AND e.type IN ('Pass', 'Carry', 'Dribble', 'Shot') {match_filter}
            GROUP BY e.match_id, e.possession, e.team, e.player
        ),
        player_minutes AS (
            SELECT 
                e.match_id, e.team, e.player,
                MAX(e.minute) - MIN(e.minute) as minutes_played
            FROM '{events}' e
            WHERE e.player IS NOT NULL {match_filter}
            GROUP BY e.match_id, e.team, e.player
        ),
        team_xg AS (
            SELECT match_id, team, SUM(possession_xg) as team_total_xg
            FROM shot_possessions GROUP BY match_id, team
        ),
        player_xg_chain AS (
            SELECT 
                pt.match_id,   -- 1. ADDED THIS: Must select it to use it later
                pt.player,
                pt.team,
                COUNT(DISTINCT pt.possession) as possessions_with_shot,
                ROUND(SUM(sp.possession_xg), 3) as xg_chain,
                SUM(sp.shots_in_possession) as shots_in_chain,
                SUM(sp.goals_in_possession) as goals_in_chain,
                ROUND(AVG(sp.possession_xg), 3) as avg_xg_per_possession,
                SUM(pt.touches_in_possession) as total_touches_in_chains,
                SUM(pm.minutes_played) as minutes_played
            FROM player_touches_in_shot_possessions pt
            INNER JOIN shot_possessions sp ON pt.match_id = sp.match_id AND pt.possession = sp.possession AND pt.team = sp.team
            LEFT JOIN player_minutes pm ON pt.match_id = pm.match_id AND pt.team = pm.team AND pt.player = pm.player
            GROUP BY pt.player, pt.team, pt.match_id
            HAVING SUM(pt.touches_in_possession) >= {min_touches}
        )
        SELECT 
            pxc.*,
            ROUND(pxc.xg_chain * 90.0 / NULLIF(pxc.minutes_played, 0), 3) as xg_chain_per90,
            ROUND(pxc.xg_chain * 100.0 / NULLIF(txg.team_total_xg, 0), 2) as team_involvement_pct
        FROM player_xg_chain pxc
        LEFT JOIN team_xg txg 
            ON pxc.team = txg.team 
            AND pxc.match_id = txg.match_id -- 2. ADDED THIS: Prevents many-to-many duplicates
        ORDER BY {'xg_chain_per90' if per_90 else 'xg_chain'} DESC
        """
        
        return conn.execute(query).df()
    
    else:
        df = events.copy()
        
        if match_id is not None:
            df = df[df['match_id'] == match_id]
        
        # Find possessions with shots - take MAX xG 
        shot_possessions = df[
            (df['type'] == 'Shot') & 
            (df['shot_statsbomb_xg'].notna())
        ].groupby(['match_id', 'possession', 'team']).agg(
            possession_xg=('shot_statsbomb_xg', 'max'),  # MAX not SUM
            shots_in_possession=('type', 'count'),
            goals_in_possession=('shot_outcome', lambda x: (x == 'Goal').sum())
        ).reset_index()
        
        # Find all players who touched ball in these possessions
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
        
        # Count distinct touches per player per possession (avoid carry inflation)
        touches_per_possession = player_touches.groupby(
            ['match_id', 'possession', 'team', 'player']
        )['id'].nunique().reset_index(name='touches_in_possession')
        
        # Merge with xG values
        result = touches_per_possession.merge(
            shot_possessions,
            on=['match_id', 'possession', 'team']
        )
        
        # Calculate minutes played
        player_minutes = df[df['player'].notna()].groupby(
            ['match_id', 'team', 'player']
        )['minute'].agg(lambda x: x.max() - x.min()).reset_index(name='minutes_played')
        
        # Calculate team total xG
        team_xg = shot_possessions.groupby(['match_id', 'team'])['possession_xg'].sum().reset_index(name='team_total_xg')
        
        # Aggregate by player
        final = result.groupby(['match_id', 'player', 'team']).agg(
            possessions_with_shot=('possession', 'nunique'),
            xg_chain=('possession_xg', 'sum'),
            shots_in_chain=('shots_in_possession', 'sum'),
            goals_in_chain=('goals_in_possession', 'sum'),
            avg_xg_per_possession=('possession_xg', 'mean'),
            total_touches_in_chains=('touches_in_possession', 'sum')
        ).reset_index()
        
        # Merge minutes and team xG
        final = final.merge(player_minutes, on=['match_id', 'team', 'player'], how='left')
        final = final.merge(team_xg, on=['match_id', 'team'], how='left')
        
        # Calculate per-90 and involvement %
        final['minutes_played'] = final['minutes_played'].fillna(90)
        final['xg_chain_per90'] = (final['xg_chain'] * 90.0 / final['minutes_played']).round(3)
        final['team_involvement_pct'] = (final['xg_chain'] * 100.0 / final['team_total_xg']).round(2)
        
        # Round
        final['xg_chain'] = final['xg_chain'].round(3)
        final['avg_xg_per_possession'] = final['avg_xg_per_possession'].round(3)
        
        # Filter by min touches
        final = final[final['total_touches_in_chains'] >= min_touches]
        
        # Drop match_id if aggregating across matches
        if match_id is None:
            final = final.groupby(['player', 'team']).agg({
                'possessions_with_shot': 'sum',
                'xg_chain': 'sum',
                'shots_in_chain': 'sum',
                'goals_in_chain': 'sum',
                'avg_xg_per_possession': 'mean',
                'total_touches_in_chains': 'sum',
                'minutes_played': 'sum',
                'xg_chain_per90': 'mean',
                'team_involvement_pct': 'mean'
            }).reset_index()
            final['xg_chain'] = final['xg_chain'].round(3)
            final['xg_chain_per90'] = (final['xg_chain'] * 90.0 / final['minutes_played']).round(3)
        
        sort_col = 'xg_chain_per90' if per_90 else 'xg_chain'
        return final.sort_values(sort_col, ascending=False)


def calculate_xg_buildup(events, conn=None, match_id=None, min_touches=1, per_90=True) -> pd.DataFrame:
    # xG Buildup: xG of possessions where player contributed but didn't shoot or assist

    if isinstance(events, str):
        if conn is None:
            conn = duckdb.connect()
        
        match_filter = f"AND e.match_id = {match_id}" if match_id else ""
        
        query = f"""
        WITH shot_events AS (
            -- Get all shots with MAX xG per possession
            SELECT 
                match_id,
                possession,
                team,
                player as shooter,
                MAX(shot_statsbomb_xg) as possession_xg,
                shot_key_pass_id
            FROM '{events}'
            WHERE type = 'Shot'
              AND shot_statsbomb_xg IS NOT NULL
              {match_filter.replace('e.', '')}
            GROUP BY match_id, possession, team, player, shot_key_pass_id
        ),
        assist_passes AS (
            -- Find the assister (optimized with LEFT JOIN instead of NOT EXISTS)
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
            -- Find players in buildup (exclude shooter & assister via LEFT JOIN)
            SELECT 
                e.match_id,
                e.possession,
                e.team,
                e.player,
                COUNT(DISTINCT e.id) as touches_in_buildup
            FROM '{events}' e
            INNER JOIN shot_events se 
                ON e.match_id = se.match_id 
                AND e.possession = se.possession
                AND e.team = se.team
            LEFT JOIN assist_passes ap
                ON e.match_id = ap.match_id
                AND e.possession = ap.possession
                AND e.team = ap.team
                AND e.player = ap.assister
            WHERE e.player IS NOT NULL
              AND e.type IN ('Pass', 'Carry', 'Dribble')
              AND e.player != se.shooter
              AND ap.assister IS NULL  -- Optimized: LEFT JOIN where NULL
              {match_filter}
            GROUP BY e.match_id, e.possession, e.team, e.player
        ),
        player_minutes AS (
            SELECT 
                e.match_id,
                e.team,
                e.player,
                MAX(e.minute) - MIN(e.minute) as minutes_played
            FROM '{events}' e
            WHERE e.player IS NOT NULL
              {match_filter}
            GROUP BY e.match_id, e.team, e.player
        )
        SELECT 
            bt.player,
            bt.team,
            bt.match_id,
            COUNT(DISTINCT bt.possession) as possessions_with_buildup,
            ROUND(SUM(se.possession_xg), 3) as xg_buildup,
            ROUND(AVG(se.possession_xg), 3) as avg_xg_per_possession,
            SUM(bt.touches_in_buildup) as total_touches_in_buildup,
            SUM(pm.minutes_played) as minutes_played,
            ROUND(SUM(se.possession_xg) * 90.0 / NULLIF(SUM(pm.minutes_played), 0), 3) as xg_buildup_per90
        FROM buildup_touches bt
        INNER JOIN shot_events se 
            ON bt.match_id = se.match_id 
            AND bt.possession = se.possession
            AND bt.team = se.team
        LEFT JOIN player_minutes pm
            ON bt.match_id = pm.match_id
            AND bt.team = pm.team
            AND bt.player = pm.player
        GROUP BY bt.match_id, bt.player, bt.team
        HAVING SUM(bt.touches_in_buildup) >= {min_touches}
        ORDER BY {'xg_buildup_per90' if per_90 else 'xg_buildup'} DESC
        """
        
        return conn.execute(query).df()
    
    else:
        df = events.copy()
        
        if match_id is not None:
            df = df[df['match_id'] == match_id]
        
        # Get shots with MAX xG per possession
        shots = df[
            (df['type'] == 'Shot') & 
            (df['shot_statsbomb_xg'].notna())
        ].groupby(['match_id', 'possession', 'team', 'player']).agg(
            possession_xg=('shot_statsbomb_xg', 'max'),
            shot_key_pass_id=('shot_key_pass_id', 'first')
        ).reset_index()
        shots.rename(columns={'player': 'shooter'}, inplace=True)
        
        # Get assisters
        assisters = shots[shots['shot_key_pass_id'].notna()].merge(
            df[['id', 'player']],
            left_on='shot_key_pass_id',
            right_on='id',
            how='left'
        )[['match_id', 'possession', 'team', 'player']].drop_duplicates()
        assisters.rename(columns={'player': 'assister'}, inplace=True)
        
        # Get all touches in shot possessions
        touches = df[
            (df['player'].notna()) &
            (df['type'].isin(['Pass', 'Carry', 'Dribble']))
        ][['match_id', 'possession', 'team', 'player', 'id']].copy()
        
        # Merge with shots
        buildup = touches.merge(shots[['match_id', 'possession', 'team', 'shooter', 'possession_xg']], 
                                on=['match_id', 'possession', 'team'])
        
        # Exclude shooters
        buildup = buildup[buildup['player'] != buildup['shooter']]
        
        # Exclude assisters 
        buildup = buildup.merge(
            assisters,
            on=['match_id', 'possession', 'team'],
            how='left'
        )
        buildup = buildup[buildup['assister'].isna() | (buildup['player'] != buildup['assister'])]
        
        # Count distinct touches
        touch_counts = buildup.groupby(
            ['match_id', 'possession', 'team', 'player']
        )['id'].nunique().reset_index(name='touches_in_buildup')
        
        # Merge with xG
        buildup = touch_counts.merge(
            buildup[['match_id', 'possession', 'team', 'possession_xg']].drop_duplicates(),
            on=['match_id', 'possession', 'team']
        )
        
        # Calculate minutes
        player_minutes = df[df['player'].notna()].groupby(
            ['match_id', 'team', 'player']
        )['minute'].agg(lambda x: x.max() - x.min()).reset_index(name='minutes_played')
        
        # Aggregate
        result = buildup.groupby(['match_id', 'player', 'team']).agg(
            possessions_with_buildup=('possession', 'nunique'),
            xg_buildup=('possession_xg', 'sum'),
            avg_xg_per_possession=('possession_xg', 'mean'),
            total_touches_in_buildup=('touches_in_buildup', 'sum')
        ).reset_index()
        
        # Merge minutes
        result = result.merge(player_minutes, on=['match_id', 'team', 'player'], how='left')
        result['minutes_played'] = result['minutes_played'].fillna(90)
        
        # Calculate per-90
        result['xg_buildup_per90'] = (result['xg_buildup'] * 90.0 / result['minutes_played']).round(3)
        
        # Round
        result['xg_buildup'] = result['xg_buildup'].round(3)
        result['avg_xg_per_possession'] = result['avg_xg_per_possession'].round(3)
        
        # Filter
        result = result[result['total_touches_in_buildup'] >= min_touches]
        
        # Aggregate across matches if needed
        if match_id is None:
            result = result.groupby(['player', 'team']).agg({
                'possessions_with_buildup': 'sum',
                'xg_buildup': 'sum',
                'avg_xg_per_possession': 'mean',
                'total_touches_in_buildup': 'sum',
                'minutes_played': 'sum'
            }).reset_index()
            result['xg_buildup_per90'] = (result['xg_buildup'] * 90.0 / result['minutes_played']).round(3)
        
        sort_col = 'xg_buildup_per90' if per_90 else 'xg_buildup'
        return result.sort_values(sort_col, ascending=False)


def compare_xg_chain_vs_buildup(events, conn=None, match_id=None, is_season_data=False) -> pd.DataFrame:
    """
    Compare xG Chain vs xG Buildup to classify player roles
    Integrates reliability flags to avoid the 'small sample size' trap
    """
    
    # Fetch raw metrics
    xg_chain = calculate_xg_chain(events, conn, match_id, per_90=True)
    xg_buildup = calculate_xg_buildup(events, conn, match_id, per_90=True)
    
    # Merge dataframes
    comparison = xg_chain.merge(
        xg_buildup[['match_id', 'player', 'team', 'xg_buildup', 'xg_buildup_per90', 'possessions_with_buildup']],
        on=['match_id', 'player', 'team'], # Merge on match_id too to keep match-level context
        how='outer'
    ).fillna(0)

    # Handle edge cases for extremely low minutes
    # We force per-90s to 0 if they played less than a minute to avoid division by zero/infinity
    comparison.loc[comparison['minutes_played'] < 1, ['xg_chain_per90', 'xg_buildup_per90']] = 0

    # Add the Reliability Flag
    # 15 mins for a single game is enough to see a 'cameo', 270 mins (3 games) for a season
    min_mins = 270 if is_season_data else 15 
    comparison['is_reliable'] = comparison['minutes_played'] >= min_mins
    
    # Calculate derived metrics
    comparison['90s_played'] = (comparison['minutes_played'] / 90.0).round(2)
    
    # Ratio calculation with safety for zero division
    comparison['buildup_to_chain_ratio'] = (
        comparison['xg_buildup'] / comparison['xg_chain'].replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan).fillna(0).round(3)
    
    # Set Context-aware thresholds
    if is_season_data:
        involvement_threshold = 0.5 # Minimum xG Chain p90 to be considered 'involved' over a season
        talisman_threshold = 30.0    # % of team xG involvement
    else:
        involvement_threshold = 0.1 # Very low bar for a single match
        talisman_threshold = 40.0    # Higher bar for 'Talisman' in a single game
    
    # Role Classification Logic
    def classify_role(row):
        # Check sample size
        if not row['is_reliable']:
            return 'Insignificant Sample'
        
        # If they have 0 buildup but are involved in shots, they aren't playmaking/talismanic
        if row['xg_buildup'] == 0 and row['xg_chain'] > 0:
            return 'Pure Finisher'

        xg_chain_p90 = row['xg_chain_per90']
        ratio = row['buildup_to_chain_ratio']
        involvement = row['team_involvement_pct']
        
        # Talisman check (Requires high involvement AND at least some buildup)
        if involvement >= 30.0: 
            if ratio > 0.6:
                return 'Deep Talisman'
            else:
                return 'Attacking Talisman'
        
        # Check if they were actually involved in play
        if row['xg_chain_per90'] < involvement_threshold:
            return 'Limited Involvement'
        
        ratio = row['buildup_to_chain_ratio']
        involvement = row['team_involvement_pct']
        
        # Check for Talisman status (The 'Messi' or 'Miedema' role)
        if involvement >= talisman_threshold:
            return 'Deep Talisman' if ratio > 0.6 else 'Attacking Talisman'
        
        # Role based on the "Deepness" of their contribution
        if ratio > 0.8:
            return 'Deep Playmaker'     # Center-backs / Holding Mids
        elif ratio > 0.5:
            return 'Complete Attacker'  # Box-to-box or Creative 10s
        elif ratio > 0.2:
            return 'Advanced Playmaker'# Final third creators
        else:
            return 'Pure Finisher'      # Poachers / Target men
    
    comparison['player_role'] = comparison.apply(classify_role, axis=1)
    
    # Clean up and Sort
    # We sort by xG Chain p90 but keep the 'Unreliable' players at the bottom
    comparison = comparison.sort_values(
        by=['is_reliable', 'xg_chain_per90'], 
        ascending=[False, False]
    )
    
    return comparison