import pandas as pd
import numpy as np
import duckdb
from typing import Union, Optional


def calculate_pressure_metrics(events, conn=None, matches=None, match_id=None, player=None) -> pd.DataFrame:
    """
    Calculate Pressure and Counter-pressing metrics from StatsBomb data.
    """
    if isinstance(events, str):
        if conn is None:
            conn = duckdb.connect()

        filters = ["e.type = 'Pressure'"]
        params = []
        if match_id is not None:
            filters.append("e.match_id = ?")
            params.append(match_id)
        if player is not None:
            filters.append("e.player = ?")
            params.append(player)

        where_clause = " AND ".join(filters)
        season_join = f"LEFT JOIN '{matches}' m ON pb.match_id = m.match_id" if matches else ""
        season_select = "m.season_name," if matches else ""
        season_group = "m.season_name," if matches else ""

        query = f"""
        WITH pressure_base AS (
            SELECT 
                e.match_id, e.team, e.player, e.id, e.index_num, e.location_x, e.minute,
                COALESCE(e.counterpress, False) as is_counterpress,
                EXISTS (
                    SELECT 1 FROM '{events}' e2 
                    WHERE e2.match_id = e.match_id 
                      AND e2.team = e.team
                      AND e2.type IN ('Interception', 'Tackle', 'Ball Recovery', 'Duel', 'Clearance')
                      AND e2.index_num BETWEEN e.index_num AND e.index_num + 5
                ) as is_regain
            FROM '{events}' e
            WHERE {where_clause}
        ),
        player_minutes AS (
            SELECT match_id, team, player, (MAX(minute) - MIN(minute)) as mins
            FROM '{events}'
            WHERE player IS NOT NULL
            GROUP BY 1, 2, 3
        )
        SELECT 
            {season_select}
            pb.match_id, pb.team, pb.player,
            COUNT(*) as total_pressures,
            SUM(CASE WHEN pb.is_counterpress THEN 1 ELSE 0 END) as counterpresses,
            SUM(CASE WHEN pb.is_regain THEN 1 ELSE 0 END) as pressure_regains,
            COUNT(*) FILTER (WHERE pb.location_x >= 80) as high_pressures,
            ROUND(SUM(CASE WHEN pb.is_regain THEN 1 ELSE 0 END) * 100.0 / NULLIF(COUNT(*), 0), 2) as pressure_success_pct,
            pm.mins as minutes_played,
            ROUND(COUNT(*) * 90.0 / NULLIF(pm.mins, 0), 2) as pressures_per_90
        FROM pressure_base pb
        LEFT JOIN player_minutes pm 
            ON pb.match_id = pm.match_id 
            AND pb.team = pm.team 
            AND pb.player = pm.player
        {season_join}
        GROUP BY {season_group} pb.match_id, pb.team, pb.player, pm.mins
        ORDER BY total_pressures DESC
        """
        return conn.execute(query, params).df()

    else:
        df = events.copy()

        if match_id:
            df = df[df['match_id'] == match_id]
        if player:
            df = df[df['player'] == player]

        df = df.sort_values(['match_id', 'index_num'])

        regain_types = ['Interception', 'Tackle', 'Ball Recovery', 'Duel', 'Clearance']
        df['is_regain_event'] = df['type'].isin(regain_types).astype(int)

        df['regain_within_5'] = False
        for i in range(1, 6):
            df['regain_within_5'] = df['regain_within_5'] | (
                (df['is_regain_event'].shift(-i) == 1) &
                (df['team'].shift(-i) == df['team']) &
                (df['match_id'].shift(-i) == df['match_id'])
            )

        pressures = df[df['type'] == 'Pressure'].copy()

        mins = df.groupby(['match_id', 'team', 'player'])['minute'].agg(lambda x: x.max() - x.min()).reset_index(name='mins')

        res = pressures.groupby(['match_id', 'team', 'player']).agg(
            total_pressures=('id', 'count'),
            counterpresses=('counterpress', lambda x: x.fillna(False).sum()),
            pressure_regains=('regain_within_5', 'sum'),
            high_pressures=('location_x', lambda x: (x >= 80).sum())
        ).reset_index()

        res = res.merge(mins, on=['match_id', 'team', 'player'], how='left')
        res['pressure_success_pct'] = round(res['pressure_regains'] * 100 / res['total_pressures'], 2)
        res['pressures_per_90'] = round(res['total_pressures'] * 90 / res['mins'].replace(0, np.nan), 2)

        return res.sort_values('total_pressures', ascending=False)


def calculate_defensive_actions(
    events: Union[pd.DataFrame, str],
    conn: duckdb.DuckDBPyConnection = None,
    matches: Optional[str] = None,
    match_id: Optional[int] = None,
    player: Optional[str] = None,
) -> pd.DataFrame:
    """Calculate defensive contributions per player."""
    if isinstance(events, str):
        if conn is None:
            conn = duckdb.connect()

        filters = []
        if match_id is not None:
            filters.append(f"match_id = {match_id}")
        if player is not None:
            filters.append(f"player = '{player}'")

        where_clause = f"WHERE {' AND '.join(filters)}" if filters else ""
        season_join = f"LEFT JOIN '{matches}' m ON da.match_id = m.match_id" if matches else ""
        season_select = "m.season_name," if matches else ""
        season_group = "m.season_name," if matches else ""

        query = f"""
        WITH defensive_events AS (
            SELECT 
                match_id, team, player, type, location_x, location_y,
                duel_outcome,
                CASE 
                    WHEN location_x < 40 THEN 'Defensive Third'
                    WHEN location_x < 80 THEN 'Middle Third'
                    ELSE 'Attacking Third'
                END as zone
            FROM '{events}'
            WHERE type IN ('Tackle', 'Interception', 'Pressure', 'Block', 
                          'Clearance', 'Duel', 'Ball Recovery', 'Foul Committed')
              AND player IS NOT NULL
              {where_clause.replace('WHERE', 'AND') if where_clause else ''}
        ),
        da AS (
            SELECT 
                match_id, team, player,
                SUM(CASE WHEN type = 'Tackle' THEN 1 ELSE 0 END) as tackles,
                SUM(CASE WHEN type = 'Interception' THEN 1 ELSE 0 END) as interceptions,
                SUM(CASE WHEN type = 'Pressure' THEN 1 ELSE 0 END) as pressures,
                SUM(CASE WHEN type = 'Block' THEN 1 ELSE 0 END) as blocks,
                SUM(CASE WHEN type = 'Clearance' THEN 1 ELSE 0 END) as clearances,
                SUM(CASE WHEN type = 'Ball Recovery' THEN 1 ELSE 0 END) as ball_recoveries,
                SUM(CASE WHEN type = 'Foul Committed' THEN 1 ELSE 0 END) as fouls_committed,
                SUM(CASE WHEN type = 'Duel' AND duel_outcome = 'Won' THEN 1 ELSE 0 END) as duels_won,
                SUM(CASE WHEN type = 'Duel' THEN 1 ELSE 0 END) as total_duels,
                SUM(CASE WHEN zone = 'Defensive Third' THEN 1 ELSE 0 END) as defensive_third_actions,
                SUM(CASE WHEN zone = 'Middle Third' THEN 1 ELSE 0 END) as middle_third_actions,
                SUM(CASE WHEN zone = 'Attacking Third' THEN 1 ELSE 0 END) as attacking_third_actions,
                SUM(CASE WHEN location_x >= 72 AND type IN ('Tackle', 'Interception', 'Ball Recovery') 
                    THEN 1 ELSE 0 END) as high_turnovers,
                SUM(CASE WHEN type IN ('Tackle', 'Interception', 'Pressure', 'Block') 
                    THEN 1 ELSE 0 END) as total_defensive_actions
            FROM defensive_events
            GROUP BY match_id, team, player
            HAVING SUM(CASE WHEN type IN ('Tackle', 'Interception', 'Pressure', 'Block') 
                       THEN 1 ELSE 0 END) > 0
        )
        SELECT 
            {season_select}
            da.*,
            ROUND(da.duels_won * 100.0 / NULLIF(da.total_duels, 0), 2) as duel_win_pct
        FROM da
        {season_join}
        ORDER BY total_defensive_actions DESC
        """

        return conn.execute(query).df()

    else:
        df = events.copy()

        defensive_types = ['Tackle', 'Interception', 'Pressure', 'Block',
                          'Clearance', 'Duel', 'Ball Recovery', 'Foul Committed']
        df = df[(df['type'].isin(defensive_types)) & (df['player'].notna())]

        if match_id is not None:
            df = df[df['match_id'] == match_id]
        if player is not None:
            df = df[df['player'] == player]

        df['zone'] = pd.cut(
            df['location_x'],
            bins=[0, 40, 80, 120],
            labels=['Defensive Third', 'Middle Third', 'Attacking Third']
        )

        result = df.groupby(['match_id', 'team', 'player']).agg(
            tackles=('type', lambda x: (x == 'Tackle').sum()),
            interceptions=('type', lambda x: (x == 'Interception').sum()),
            pressures=('type', lambda x: (x == 'Pressure').sum()),
            blocks=('type', lambda x: (x == 'Block').sum()),
            clearances=('type', lambda x: (x == 'Clearance').sum()),
            ball_recoveries=('type', lambda x: (x == 'Ball Recovery').sum()),
            fouls_committed=('type', lambda x: (x == 'Foul Committed').sum()),
            duels_won=('duel_outcome', lambda x: (x == 'Won').sum()),
            total_duels=('type', lambda x: (x == 'Duel').sum()),
            defensive_third_actions=('zone', lambda x: (x == 'Defensive Third').sum()),
            middle_third_actions=('zone', lambda x: (x == 'Middle Third').sum()),
            attacking_third_actions=('zone', lambda x: (x == 'Attacking Third').sum()),
            high_turnovers=('location_x', lambda x: ((x >= 72) &
                           (df.loc[x.index, 'type'].isin(['Tackle', 'Interception', 'Ball Recovery']))).sum()),
        ).reset_index()

        result['total_defensive_actions'] = (
            result['tackles'] + result['interceptions'] +
            result['pressures'] + result['blocks']
        )
        result['duel_win_pct'] = (
            result['duels_won'] / result['total_duels'] * 100
        ).fillna(0).round(2)

        result = result[result['total_defensive_actions'] > 0]

        return result.sort_values('total_defensive_actions', ascending=False)


def calculate_defensive_profile(
    events: Union[pd.DataFrame, str],
    conn: duckdb.DuckDBPyConnection = None,
    matches: Optional[str] = None,
    match_id: Optional[int] = None,
    min_actions: int = 10,
) -> pd.DataFrame:
    """Classify players by defensive style based on action types and locations."""
    # Pass matches through to calculate_defensive_actions
    defensive = calculate_defensive_actions(events, conn, matches=matches, match_id=match_id)

    defensive = defensive[defensive['total_defensive_actions'] >= min_actions]

    if len(defensive) == 0:
        return pd.DataFrame()

    defensive['pressure_pct'] = (
        defensive['pressures'] / defensive['total_defensive_actions'] * 100
    ).round(2)

    defensive['tackle_interception_pct'] = (
        (defensive['tackles'] + defensive['interceptions']) /
        defensive['total_defensive_actions'] * 100
    ).round(2)

    defensive['high_turnover_pct'] = (
        defensive['high_turnovers'] / defensive['total_defensive_actions'] * 100
    ).round(2)

    defensive['attacking_third_pct'] = (
        defensive['attacking_third_actions'] /
        (defensive['defensive_third_actions'] + defensive['middle_third_actions'] +
         defensive['attacking_third_actions']) * 100
    ).round(2)

    def classify_defender(row):
        if row['pressure_pct'] > 40 and row['high_turnover_pct'] > 30:
            return 'Aggressive Presser'
        elif row['tackle_interception_pct'] > 50:
            return 'Ball Winner'
        elif row['blocks'] + row['clearances'] > row['total_defensive_actions'] * 0.4:
            return 'Protector'
        elif row['duels_won'] > 0 and row['duel_win_pct'] > 65:
            return 'Dominant Duelist'
        else:
            return 'Balanced Defender'

    defensive['defensive_profile'] = defensive.apply(classify_defender, axis=1)

    return defensive.sort_values('total_defensive_actions', ascending=False)