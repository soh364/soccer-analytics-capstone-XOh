"""Aggregation metrics: xG summaries, match statistics"""

import pandas as pd
import duckdb
from typing import Union, Optional


def aggregate_xg_by_team(events, conn=None, matches=None, match_id=None) -> pd.DataFrame:
    """Aggregates team performance focusing on Non-Penalty xG (npxG)."""
    if isinstance(events, str):
        if conn is None: conn = duckdb.connect()
        match_filter = f"AND e.match_id = {match_id}" if match_id else ""
        season_join = f"LEFT JOIN '{matches}' m ON e.match_id = m.match_id" if matches else ""
        season_select = "m.season_name," if matches else ""
        season_group = "m.season_name," if matches else ""

        query = f"""
        SELECT 
            {season_select}
            e.match_id,
            e.team,
            COUNT(*) as total_shots,
            ROUND(SUM(e.shot_statsbomb_xg), 3) as total_xg,
            ROUND(SUM(CASE WHEN e.shot_type != 'Penalty' THEN e.shot_statsbomb_xg ELSE 0 END), 3) as npxg,
            SUM(CASE WHEN e.shot_outcome = 'Goal' THEN 1 ELSE 0 END) as total_goals,
            SUM(CASE WHEN e.shot_outcome = 'Goal' AND e.shot_type != 'Penalty' THEN 1 ELSE 0 END) as np_goals,
            ROUND(AVG(CASE WHEN e.shot_type != 'Penalty' THEN e.shot_statsbomb_xg END), 3) as avg_npxg_per_shot
        FROM '{events}' e
        {season_join}
        WHERE e.type = 'Shot' AND e.shot_statsbomb_xg IS NOT NULL {match_filter}
        GROUP BY {season_group} e.match_id, e.team
        ORDER BY e.match_id, npxg DESC
        """
        return conn.execute(query).df()

    else:
        df = events[(events['type'] == 'Shot') & (events['shot_statsbomb_xg'].notna())].copy()
        if match_id: df = df[df['match_id'] == match_id]

        df['is_penalty'] = df['shot_type'] == 'Penalty'

        result = df.groupby(['match_id', 'team']).agg(
            total_shots=('type', 'count'),
            total_xg=('shot_statsbomb_xg', 'sum'),
            npxg=('shot_statsbomb_xg', lambda x: x[df.loc[x.index, 'is_penalty'] == False].sum()),
            total_goals=('shot_outcome', lambda x: (x == 'Goal').sum()),
            np_goals=('shot_outcome', lambda x: ((x == 'Goal') & (~df.loc[x.index, 'is_penalty'])).sum())
        ).reset_index()

        result['avg_npxg_per_shot'] = round(result['npxg'] / (result['total_shots'] - df.groupby(['match_id', 'team'])['is_penalty'].sum().values), 3)
        return result.sort_values(['match_id', 'npxg'], ascending=[True, False])


def aggregate_xg_by_player(events, conn=None, matches=None, match_id=None, min_shots=3) -> pd.DataFrame:
    """Calculates player efficiency using G-xG (Goals minus xG) differential."""
    if isinstance(events, str):
        if conn is None: conn = duckdb.connect()
        match_filter = f"AND e.match_id = {match_id}" if match_id else ""
        season_join = f"LEFT JOIN '{matches}' m ON e.match_id = m.match_id" if matches else ""
        season_select = "m.season_name," if matches else ""
        season_group = "m.season_name," if matches else ""

        query = f"""
        SELECT 
            {season_select}
            e.player,
            e.team,
            COUNT(DISTINCT e.match_id) as matches,
            COUNT(*) as shots,
            ROUND(SUM(e.shot_statsbomb_xg), 2) as xg,
            SUM(CASE WHEN e.shot_outcome = 'Goal' THEN 1 ELSE 0 END) as goals,
            ROUND(SUM(CASE WHEN e.shot_outcome = 'Goal' THEN 1 ELSE 0 END) - SUM(e.shot_statsbomb_xg), 2) as goals_minus_xg
        FROM '{events}' e
        {season_join}
        WHERE e.type = 'Shot' AND e.player IS NOT NULL {match_filter}
        GROUP BY {season_group} e.player, e.team
        HAVING COUNT(*) >= {min_shots}
        ORDER BY goals_minus_xg DESC
        """
        return conn.execute(query).df()

    else:
        df = events[(events['type'] == 'Shot') & (events['player'].notna())].copy()
        if match_id: df = df[df['match_id'] == match_id]

        res = df.groupby(['player', 'team']).agg(
            matches=('match_id', 'nunique'),
            shots=('type', 'count'),
            xg=('shot_statsbomb_xg', 'sum'),
            goals=('shot_outcome', lambda x: (x == 'Goal').sum())
        ).reset_index()

        res['goals_minus_xg'] = round(res['goals'] - res['xg'], 2)
        return res[res['shots'] >= min_shots].sort_values('goals_minus_xg', ascending=False)


def get_match_summary(events, matches, conn=None, match_id=None) -> pd.DataFrame:
    """Comprehensive match stats including a refined possession estimate."""
    if match_id is None: raise ValueError("match_id is required for summary.")

    if isinstance(events, str):
        if conn is None: conn = duckdb.connect()
        query = f"""
        WITH team_stats AS (
            SELECT 
                team,
                COUNT(CASE WHEN type = 'Shot' THEN 1 END) as shots,
                SUM(CASE WHEN type = 'Shot' THEN shot_statsbomb_xg ELSE 0 END) as xg,
                COUNT(CASE WHEN type = 'Pass' THEN 1 END) as passes,
                COUNT(CASE WHEN type = 'Pass' AND pass_outcome IS NULL THEN 1 END) as successful_passes,
                COUNT(CASE WHEN type IN ('Pass', 'Carry', 'Dribble') THEN 1 END) as touches
            FROM '{events}' WHERE match_id = {match_id} GROUP BY team
        )
        SELECT 
            m.season_name,
            m.match_id, m.home_team, m.away_team, m.home_score, m.away_score,
            ts.team, ts.shots, ROUND(ts.xg, 2) as xg,
            ROUND(ts.successful_passes * 100.0 / ts.passes, 1) as pass_accuracy_pct,
            ROUND(ts.touches * 100.0 / SUM(ts.touches) OVER (), 1) as possession_pct
        FROM '{matches}' m
        JOIN team_stats ts ON (ts.team = m.home_team OR ts.team = m.away_team)
        WHERE m.match_id = {match_id}
        """
        return conn.execute(query).df()

    else:
        m_info = matches[matches['match_id'] == match_id].iloc[0]
        ev = events[events['match_id'] == match_id]

        summary = []
        total_touches = ev[ev['type'].isin(['Pass', 'Carry', 'Dribble'])].shape[0]

        for team in [m_info['home_team'], m_info['away_team']]:
            t_ev = ev[ev['team'] == team]
            sh = t_ev[t_ev['type'] == 'Shot']
            ps = t_ev[t_ev['type'] == 'Pass']
            tch = t_ev[t_ev['type'].isin(['Pass', 'Carry', 'Dribble'])].shape[0]

            summary.append({
                'season_name': m_info.get('season_name'),
                'team': team,
                'shots': len(sh),
                'xg': round(sh['shot_statsbomb_xg'].sum(), 2),
                'pass_accuracy': round((ps['pass_outcome'].isna().sum() / len(ps)) * 100, 1) if len(ps) > 0 else 0,
                'possession_pct': round((tch / total_touches) * 100, 1) if total_touches > 0 else 0
            })
        return pd.DataFrame(summary)


def calculate_pass_completion_by_zone(events, conn=None, matches=None, match_id=None) -> pd.DataFrame:
    """Calculate pass completion percentage by pitch zone."""

    if isinstance(events, str):
        if conn is None:
            conn = duckdb.connect()

        match_filter = f"AND e.match_id = {match_id}" if match_id else ""
        season_join = f"LEFT JOIN '{matches}' m ON e.match_id = m.match_id" if matches else ""
        season_select = "m.season_name," if matches else ""
        season_group = "m.season_name," if matches else ""

        query = f"""
        SELECT 
            {season_select}
            e.match_id,
            e.team,
            CASE 
                WHEN e.location_x < 40 THEN 'Defensive Third'
                WHEN e.location_x < 80 THEN 'Middle Third'
                ELSE 'Attacking Third'
            END as zone,
            COUNT(*) as total_passes,
            COUNT(*) FILTER (WHERE e.pass_outcome IS NULL) as completed_passes,
            ROUND(COUNT(*) FILTER (WHERE e.pass_outcome IS NULL) * 100.0 / COUNT(*), 2) as completion_pct
        FROM '{events}' e
        {season_join}
        WHERE e.type = 'Pass'
          AND e.location_x IS NOT NULL
          {match_filter}
        GROUP BY {season_group} e.match_id, e.team, zone
        ORDER BY e.match_id, e.team, zone
        """

        return conn.execute(query).df()

    else:
        df = events.copy()
        df = df[(df['type'] == 'Pass') & (df['location_x'].notna())]
        if match_id is not None:
            df = df[df['match_id'] == match_id]

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