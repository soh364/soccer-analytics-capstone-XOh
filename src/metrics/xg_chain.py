"""xG Chain and xG Buildup metrics - credits all players in goal-scoring possessions"""

import pandas as pd
import numpy as np
import duckdb
from typing import Union, Optional


def calculate_xg_chain(events, conn=None, matches=None, match_id=None, min_touches=1, per_90=True) -> pd.DataFrame:
    # xG Chain: Total xG of possessions where player had a touch

    if isinstance(events, str):
        if conn is None:
            conn = duckdb.connect()

        match_filter = f"AND e.match_id = {match_id}" if match_id else ""
        season_join = f"LEFT JOIN '{matches}' m ON pxc.match_id = m.match_id" if matches else ""
        season_select = "m.season_name," if matches else ""

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
            SELECT e.match_id, e.team, e.player, MAX(e.minute) - MIN(e.minute) as minutes_played
            FROM '{events}' e
            WHERE e.player IS NOT NULL {match_filter}
            GROUP BY e.match_id, e.team, e.player
        ),
        team_xg AS (
            SELECT match_id, team, SUM(possession_xg) as team_total_xg
            FROM shot_possessions GROUP BY match_id, team
        ),
        pxc AS (
            SELECT 
                pt.match_id, pt.player, pt.team,
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
            {season_select}
            pxc.*,
            ROUND(pxc.xg_chain * 90.0 / NULLIF(pxc.minutes_played, 0), 3) as xg_chain_per90,
            ROUND(pxc.xg_chain * 100.0 / NULLIF(txg.team_total_xg, 0), 2) as team_involvement_pct
        FROM pxc
        LEFT JOIN team_xg txg ON pxc.team = txg.team AND pxc.match_id = txg.match_id
        {season_join}
        ORDER BY {'xg_chain_per90' if per_90 else 'xg_chain'} DESC
        """

        return conn.execute(query).df()

    else:
        df = events.copy()
        if match_id is not None:
            df = df[df['match_id'] == match_id]

        shot_possessions = df[
            (df['type'] == 'Shot') & (df['shot_statsbomb_xg'].notna())
        ].groupby(['match_id', 'possession', 'team']).agg(
            possession_xg=('shot_statsbomb_xg', 'max'),
            shots_in_possession=('type', 'count'),
            goals_in_possession=('shot_outcome', lambda x: (x == 'Goal').sum())
        ).reset_index()

        relevant_events = df[(df['player'].notna()) &
                             (df['type'].isin(['Pass', 'Carry', 'Dribble', 'Shot']))].copy()

        player_touches = relevant_events.merge(
            shot_possessions[['match_id', 'possession', 'team']], on=['match_id', 'possession', 'team'], how='inner')

        touches_per_possession = player_touches.groupby(
            ['match_id', 'possession', 'team', 'player'])['id'].nunique().reset_index(name='touches_in_possession')

        result = touches_per_possession.merge(shot_possessions, on=['match_id', 'possession', 'team'])

        player_minutes = df[df['player'].notna()].groupby(
            ['match_id', 'team', 'player'])['minute'].agg(lambda x: x.max() - x.min()).reset_index(name='minutes_played')

        team_xg = shot_possessions.groupby(['match_id', 'team'])['possession_xg'].sum().reset_index(name='team_total_xg')

        final = result.groupby(['match_id', 'player', 'team']).agg(
            possessions_with_shot=('possession', 'nunique'),
            xg_chain=('possession_xg', 'sum'),
            shots_in_chain=('shots_in_possession', 'sum'),
            goals_in_chain=('goals_in_possession', 'sum'),
            avg_xg_per_possession=('possession_xg', 'mean'),
            total_touches_in_chains=('touches_in_possession', 'sum')
        ).reset_index()

        final = final.merge(player_minutes, on=['match_id', 'team', 'player'], how='left')
        final = final.merge(team_xg, on=['match_id', 'team'], how='left')
        final['minutes_played'] = final['minutes_played'].fillna(90)
        final['xg_chain_per90'] = (final['xg_chain'] * 90.0 / final['minutes_played']).round(3)
        final['team_involvement_pct'] = (final['xg_chain'] * 100.0 / final['team_total_xg']).round(2)
        final['xg_chain'] = final['xg_chain'].round(3)
        final['avg_xg_per_possession'] = final['avg_xg_per_possession'].round(3)
        final = final[final['total_touches_in_chains'] >= min_touches]

        if match_id is None:
            final = final.groupby(['player', 'team']).agg({
                'possessions_with_shot': 'sum', 'xg_chain': 'sum', 'shots_in_chain': 'sum',
                'goals_in_chain': 'sum', 'avg_xg_per_possession': 'mean',
                'total_touches_in_chains': 'sum', 'minutes_played': 'sum',
                'xg_chain_per90': 'mean', 'team_involvement_pct': 'mean'
            }).reset_index()
            final['xg_chain'] = final['xg_chain'].round(3)
            final['xg_chain_per90'] = (final['xg_chain'] * 90.0 / final['minutes_played']).round(3)

        return final.sort_values('xg_chain_per90' if per_90 else 'xg_chain', ascending=False)


def calculate_xg_buildup(events, conn=None, matches=None, match_id=None, min_touches=1, per_90=True) -> pd.DataFrame:
    # xG Buildup: xG of possessions where player contributed but didn't shoot or assist

    if isinstance(events, str):
        if conn is None:
            conn = duckdb.connect()

        match_filter = f"AND e.match_id = {match_id}" if match_id else ""
        season_join = f"LEFT JOIN '{matches}' m ON r.match_id = m.match_id" if matches else ""
        season_select = "m.season_name," if matches else ""

        query = f"""
        WITH shot_events AS (
            SELECT match_id, possession, team, player as shooter,
                MAX(shot_statsbomb_xg) as possession_xg, shot_key_pass_id
            FROM '{events}'
            WHERE type = 'Shot' AND shot_statsbomb_xg IS NOT NULL {match_filter.replace('e.', '')}
            GROUP BY match_id, possession, team, player, shot_key_pass_id
        ),
        assist_passes AS (
            SELECT DISTINCT se.match_id, se.possession, se.team, e.player as assister
            FROM shot_events se
            INNER JOIN '{events}' e ON se.match_id = e.match_id AND se.shot_key_pass_id = e.id
            WHERE se.shot_key_pass_id IS NOT NULL
        ),
        buildup_touches AS (
            SELECT e.match_id, e.possession, e.team, e.player,
                COUNT(DISTINCT e.id) as touches_in_buildup
            FROM '{events}' e
            INNER JOIN shot_events se ON e.match_id = se.match_id AND e.possession = se.possession AND e.team = se.team
            LEFT JOIN assist_passes ap ON e.match_id = ap.match_id AND e.possession = ap.possession
                AND e.team = ap.team AND e.player = ap.assister
            WHERE e.player IS NOT NULL
              AND e.type IN ('Pass', 'Carry', 'Dribble')
              AND e.player != se.shooter
              AND ap.assister IS NULL
              {match_filter}
            GROUP BY e.match_id, e.possession, e.team, e.player
        ),
        player_minutes AS (
            SELECT e.match_id, e.team, e.player, MAX(e.minute) - MIN(e.minute) as minutes_played
            FROM '{events}' e
            WHERE e.player IS NOT NULL {match_filter}
            GROUP BY e.match_id, e.team, e.player
        ),
        r AS (
            SELECT 
                bt.match_id, bt.player, bt.team,
                COUNT(DISTINCT bt.possession) as possessions_with_buildup,
                ROUND(SUM(se.possession_xg), 3) as xg_buildup,
                ROUND(AVG(se.possession_xg), 3) as avg_xg_per_possession,
                SUM(bt.touches_in_buildup) as total_touches_in_buildup,
                SUM(pm.minutes_played) as minutes_played,
                ROUND(SUM(se.possession_xg) * 90.0 / NULLIF(SUM(pm.minutes_played), 0), 3) as xg_buildup_per90
            FROM buildup_touches bt
            INNER JOIN shot_events se ON bt.match_id = se.match_id AND bt.possession = se.possession AND bt.team = se.team
            LEFT JOIN player_minutes pm ON bt.match_id = pm.match_id AND bt.team = pm.team AND bt.player = pm.player
            GROUP BY bt.match_id, bt.player, bt.team
            HAVING SUM(bt.touches_in_buildup) >= {min_touches}
        )
        SELECT {season_select} r.*
        FROM r
        {season_join}
        ORDER BY {'xg_buildup_per90' if per_90 else 'xg_buildup'} DESC
        """

        return conn.execute(query).df()

    else:
        df = events.copy()
        if match_id is not None:
            df = df[df['match_id'] == match_id]

        shots = df[(df['type'] == 'Shot') & (df['shot_statsbomb_xg'].notna())].groupby(
            ['match_id', 'possession', 'team', 'player']).agg(
            possession_xg=('shot_statsbomb_xg', 'max'),
            shot_key_pass_id=('shot_key_pass_id', 'first')
        ).reset_index()
        shots.rename(columns={'player': 'shooter'}, inplace=True)

        assisters = shots[shots['shot_key_pass_id'].notna()].merge(
            df[['id', 'player']], left_on='shot_key_pass_id', right_on='id', how='left'
        )[['match_id', 'possession', 'team', 'player']].drop_duplicates()
        assisters.rename(columns={'player': 'assister'}, inplace=True)

        touches = df[(df['player'].notna()) & (df['type'].isin(['Pass', 'Carry', 'Dribble']))][
            ['match_id', 'possession', 'team', 'player', 'id']].copy()

        buildup = touches.merge(shots[['match_id', 'possession', 'team', 'shooter', 'possession_xg']],
                                on=['match_id', 'possession', 'team'])
        buildup = buildup[buildup['player'] != buildup['shooter']]
        buildup = buildup.merge(assisters, on=['match_id', 'possession', 'team'], how='left')
        buildup = buildup[buildup['assister'].isna() | (buildup['player'] != buildup['assister'])]

        touch_counts = buildup.groupby(['match_id', 'possession', 'team', 'player'])[
            'id'].nunique().reset_index(name='touches_in_buildup')

        buildup = touch_counts.merge(
            buildup[['match_id', 'possession', 'team', 'possession_xg']].drop_duplicates(),
            on=['match_id', 'possession', 'team'])

        player_minutes = df[df['player'].notna()].groupby(['match_id', 'team', 'player'])[
            'minute'].agg(lambda x: x.max() - x.min()).reset_index(name='minutes_played')

        result = buildup.groupby(['match_id', 'player', 'team']).agg(
            possessions_with_buildup=('possession', 'nunique'),
            xg_buildup=('possession_xg', 'sum'),
            avg_xg_per_possession=('possession_xg', 'mean'),
            total_touches_in_buildup=('touches_in_buildup', 'sum')
        ).reset_index()

        result = result.merge(player_minutes, on=['match_id', 'team', 'player'], how='left')
        result['minutes_played'] = result['minutes_played'].fillna(90)
        result['xg_buildup_per90'] = (result['xg_buildup'] * 90.0 / result['minutes_played']).round(3)
        result['xg_buildup'] = result['xg_buildup'].round(3)
        result['avg_xg_per_possession'] = result['avg_xg_per_possession'].round(3)
        result = result[result['total_touches_in_buildup'] >= min_touches]

        if match_id is None:
            result = result.groupby(['player', 'team']).agg({
                'possessions_with_buildup': 'sum', 'xg_buildup': 'sum',
                'avg_xg_per_possession': 'mean', 'total_touches_in_buildup': 'sum',
                'minutes_played': 'sum'
            }).reset_index()
            result['xg_buildup_per90'] = (result['xg_buildup'] * 90.0 / result['minutes_played']).round(3)

        return result.sort_values('xg_buildup_per90' if per_90 else 'xg_buildup', ascending=False)


def calculate_team_xg_buildup(events, conn=None, matches=None, match_id=None) -> pd.DataFrame:
    """Aggregate xG buildup to TEAM level per match."""

    player_buildup = calculate_xg_buildup(events, conn, matches=matches, match_id=match_id, min_touches=1, per_90=False)

    group_cols = ['match_id', 'team']
    if 'season_name' in player_buildup.columns:
        group_cols = ['season_name'] + group_cols

    team_buildup = player_buildup.groupby(group_cols).agg({
        'possessions_with_buildup': 'sum',
        'xg_buildup': 'sum',
        'total_touches_in_buildup': 'sum'
    }).reset_index()

    team_buildup['avg_xg_per_buildup_possession'] = round(
        team_buildup['xg_buildup'] / team_buildup['possessions_with_buildup'], 3)
    team_buildup['avg_touches_per_buildup_possession'] = round(
        team_buildup['total_touches_in_buildup'] / team_buildup['possessions_with_buildup'], 2)

    return team_buildup.rename(columns={
        'possessions_with_buildup': 'buildup_possessions',
        'xg_buildup': 'total_xg_from_buildup'
    })


def compare_xg_chain_vs_buildup(events, conn=None, matches=None, match_id=None, is_season_data=False) -> pd.DataFrame:
    """Compare xG Chain vs xG Buildup to classify player roles."""

    xg_chain = calculate_xg_chain(events, conn, matches=matches, match_id=match_id, per_90=True)
    xg_buildup = calculate_xg_buildup(events, conn, matches=matches, match_id=match_id, per_90=True)

    merge_cols = ['match_id', 'player', 'team']
    if 'season_name' in xg_chain.columns:
        merge_cols = ['season_name'] + merge_cols

    buildup_cols = merge_cols + ['xg_buildup', 'xg_buildup_per90', 'possessions_with_buildup']

    comparison = xg_chain.merge(
        xg_buildup[buildup_cols], on=merge_cols, how='outer'
    ).fillna(0)

    comparison.loc[comparison['minutes_played'] < 1, ['xg_chain_per90', 'xg_buildup_per90']] = 0

    min_mins = 270 if is_season_data else 15
    comparison['is_reliable'] = comparison['minutes_played'] >= min_mins
    comparison['90s_played'] = (comparison['minutes_played'] / 90.0).round(2)
    comparison['buildup_to_chain_ratio'] = (
        comparison['xg_buildup'] / comparison['xg_chain'].replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan).fillna(0).round(3)

    involvement_threshold = 0.5 if is_season_data else 0.1
    talisman_threshold = 30.0

    def classify_role(row):
        if not row['is_reliable']:
            return 'Insignificant Sample'
        if row['xg_buildup'] == 0 and row['xg_chain'] > 0:
            return 'Pure Finisher'

        xg_chain_p90 = row['xg_chain_per90']
        ratio = row['buildup_to_chain_ratio']
        involvement = row['team_involvement_pct']

        if involvement >= 30.0:
            return 'Deep Talisman' if ratio > 0.6 else 'Attacking Talisman'
        if xg_chain_p90 < involvement_threshold:
            return 'Limited Involvement'
        if involvement >= talisman_threshold:
            return 'Deep Talisman' if ratio > 0.6 else 'Attacking Talisman'
        if ratio > 0.8: return 'Deep Playmaker'
        elif ratio > 0.5: return 'Complete Attacker'
        elif ratio > 0.2: return 'Advanced Playmaker'
        else: return 'Pure Finisher'

    comparison['player_role'] = comparison.apply(classify_role, axis=1)

    return comparison.sort_values(by=['is_reliable', 'xg_chain_per90'], ascending=[False, False])