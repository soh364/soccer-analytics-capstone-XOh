"""
Ball progression metrics: Progressive Passes, Progressive Carries, and Progressive Passes Received
"""

import pandas as pd
import numpy as np
import duckdb
from typing import Union, Optional


def calculate_progressive_passes(events, conn=None, matches=None, match_id=None, player=None) -> pd.DataFrame:
    """Calculate progressive passes using simplified distance-based definition."""

    if isinstance(events, str):
        if conn is None:
            conn = duckdb.connect()

        filters = ["e.type = 'Pass'"]
        params = []

        if match_id is not None:
            filters.append("e.match_id = ?")
            params.append(match_id)
        if player is not None:
            filters.append("e.player = ?")
            params.append(player)

        where_clause = " AND ".join(filters)
        season_join = f"LEFT JOIN '{matches}' m ON r.match_id = m.match_id" if matches else ""
        season_select = "m.season_name," if matches else ""

        query = f"""
        WITH progressive_calc AS (
            SELECT 
                e.match_id, e.team, e.player,
                e.location_x, e.location_y,
                e.pass_end_location_x, e.pass_end_location_y,
                e.pass_length, e.pass_outcome,
                (e.pass_end_location_x - e.location_x) as distance_forward,
                CASE 
                    WHEN e.pass_end_location_x >= 102 AND e.pass_end_location_y BETWEEN 18 AND 62 THEN 1
                    WHEN e.location_x >= 48 AND (e.pass_end_location_x - e.location_x) >= 10 THEN 1
                    ELSE 0
                END as is_progressive_intent
            FROM '{events}' e
            WHERE {where_clause}
              AND e.play_pattern IS NOT NULL
              AND e.play_pattern IN ('Regular Play', 'From Counter')
              AND e.location_x IS NOT NULL
              AND e.pass_end_location_x IS NOT NULL
              AND e.pass_end_location_y IS NOT NULL
        ),
        r AS (
            SELECT 
                match_id, team, player,
                COUNT(*) as total_passes,
                SUM(CASE WHEN is_progressive_intent = 1 AND pass_outcome IS NULL THEN 1 ELSE 0 END) as progressive_passes,
                SUM(is_progressive_intent) as progressive_passes_attempted,
                ROUND(SUM(CASE WHEN is_progressive_intent = 1 AND pass_outcome IS NULL THEN 1 ELSE 0 END) * 100.0 /
                      NULLIF(SUM(is_progressive_intent), 0), 2) as progressive_pass_completion_pct,
                ROUND(SUM(CASE WHEN is_progressive_intent = 1 AND pass_outcome IS NULL THEN 1 ELSE 0 END) * 100.0 /
                      COUNT(*), 2) as progressive_pass_pct,
                ROUND(AVG(CASE WHEN is_progressive_intent = 1 AND pass_outcome IS NULL THEN distance_forward END), 2) as avg_progressive_distance,
                ROUND(AVG(CASE WHEN is_progressive_intent = 1 AND pass_outcome IS NULL THEN pass_length END), 2) as avg_progressive_pass_length
            FROM progressive_calc
            GROUP BY match_id, team, player
            HAVING SUM(CASE WHEN is_progressive_intent = 1 AND pass_outcome IS NULL THEN 1 ELSE 0 END) > 0
        )
        SELECT {season_select} r.*
        FROM r
        {season_join}
        ORDER BY progressive_passes DESC
        """

        return conn.execute(query, params).df()

    else:
        df = events.copy()
        OPEN_PLAY = ['Regular Play', 'From Counter']

        df = df[
            (df['type'] == 'Pass') &
            (df['location_x'].notna()) &
            (df['pass_end_location_x'].notna()) &
            (df['pass_end_location_y'].notna()) &
            (df['play_pattern'].isin(OPEN_PLAY))
        ].copy()

        if match_id is not None:
            df = df[df['match_id'] == match_id]
        if player is not None:
            df = df[df['player'] == player]

        df['pass_length'] = np.sqrt(
            (df['pass_end_location_x'] - df['location_x'])**2 +
            (df['pass_end_location_y'] - df['location_y'])**2
        )
        df['distance_forward'] = df['pass_end_location_x'] - df['location_x']

        into_box = ((df['pass_end_location_x'] >= 102) &
                    (df['pass_end_location_y'] >= 18) & (df['pass_end_location_y'] <= 62))
        forward_from_attacking_60 = ((df['location_x'] >= 48) & (df['distance_forward'] >= 10))

        df['is_progressive_intent'] = (into_box | forward_from_attacking_60).astype(int)
        df['is_progressive_completed'] = (df['is_progressive_intent'] == 1) & (df['pass_outcome'].isna())

        result = df.groupby(['match_id', 'team', 'player']).agg(
            total_passes=('type', 'count'),
            progressive_passes=('is_progressive_completed', 'sum'),
            progressive_passes_attempted=('is_progressive_intent', 'sum'),
            avg_progressive_distance=(
                'distance_forward',
                lambda x: round(x[df.loc[x.index, 'is_progressive_completed']].mean(), 2)
                if df.loc[x.index, 'is_progressive_completed'].any() else np.nan
            ),
            avg_progressive_pass_length=(
                'pass_length',
                lambda x: round(x[df.loc[x.index, 'is_progressive_completed']].mean(), 2)
                if df.loc[x.index, 'is_progressive_completed'].any() else np.nan
            )
        ).reset_index()

        result['progressive_pass_completion_pct'] = round(
            result['progressive_passes'] * 100.0 / result['progressive_passes_attempted'].replace(0, np.nan), 2)
        result['progressive_pass_pct'] = round(
            result['progressive_passes'] * 100.0 / result['total_passes'], 2)

        return result[result['progressive_passes'] > 0].sort_values('progressive_passes', ascending=False)


def calculate_progressive_carries(events, conn=None, matches=None, match_id=None, player=None) -> pd.DataFrame:
    """Calculate progressive carries using simplified distance-based definition."""

    FINAL_40_PERCENT = 72
    FINAL_THIRD = 80
    ATTACKING_60_PERCENT = 48

    if isinstance(events, str):
        if conn is None:
            conn = duckdb.connect()

        filters = ["e.type = 'Carry'"]
        params = []

        if match_id is not None:
            filters.append("e.match_id = ?")
            params.append(match_id)
        if player is not None:
            filters.append("e.player = ?")
            params.append(player)

        where_clause = " AND ".join(filters)
        season_join = f"LEFT JOIN '{matches}' m ON r.match_id = m.match_id" if matches else ""
        season_select = "m.season_name," if matches else ""

        query = f"""
        WITH progressive_calc AS (
            SELECT 
                e.match_id, e.team, e.player,
                e.location_x, e.location_y,
                e.carry_end_location_x, e.carry_end_location_y,
                (e.carry_end_location_x - e.location_x) as distance_forward,
                SQRT(POWER(e.carry_end_location_x - e.location_x, 2) +
                     POWER(e.carry_end_location_y - e.location_y, 2)) as carry_distance,
                CASE 
                    WHEN e.carry_end_location_x >= 102 AND e.carry_end_location_y BETWEEN 18 AND 62 THEN 1
                    WHEN e.location_x >= {FINAL_40_PERCENT} AND e.location_x >= {ATTACKING_60_PERCENT}
                         AND (e.carry_end_location_x - e.location_x) >= 5 THEN 1
                    WHEN e.location_x < {FINAL_40_PERCENT} AND e.location_x >= {ATTACKING_60_PERCENT}
                         AND (e.carry_end_location_x - e.location_x) >= 10 THEN 1
                    ELSE 0
                END as is_progressive,
                CASE WHEN e.carry_end_location_x >= {FINAL_THIRD} THEN 1 ELSE 0 END as into_final_third,
                CASE WHEN e.carry_end_location_x >= 102 AND e.carry_end_location_y BETWEEN 18 AND 62 THEN 1 ELSE 0 END as into_penalty_area
            FROM '{events}' e
            WHERE {where_clause}
              AND e.play_pattern IS NOT NULL
              AND e.play_pattern IN ('Regular Play', 'From Counter')
              AND e.location_x IS NOT NULL
              AND e.carry_end_location_x IS NOT NULL
              AND e.carry_end_location_y IS NOT NULL
        ),
        r AS (
            SELECT 
                match_id, team, player,
                COUNT(*) as total_carries,
                SUM(is_progressive) as progressive_carries,
                ROUND(SUM(is_progressive) * 100.0 / COUNT(*), 2) as progressive_carry_pct,
                ROUND(AVG(CASE WHEN is_progressive = 1 THEN distance_forward END), 2) as avg_progressive_distance,
                ROUND(AVG(CASE WHEN is_progressive = 1 THEN carry_distance END), 2) as avg_progressive_carry_length,
                ROUND(AVG(CASE WHEN is_progressive = 1 THEN distance_forward / NULLIF(carry_distance, 0) END) * 100, 2) as progressive_carry_directness_pct,
                SUM(CASE WHEN is_progressive = 1 AND into_final_third = 1 THEN 1 ELSE 0 END) as progressive_carries_into_final_third,
                SUM(CASE WHEN is_progressive = 1 AND into_penalty_area = 1 THEN 1 ELSE 0 END) as progressive_carries_into_penalty_area
            FROM progressive_calc
            GROUP BY match_id, team, player
            HAVING SUM(is_progressive) > 0
        )
        SELECT {season_select} r.*
        FROM r
        {season_join}
        ORDER BY progressive_carries DESC
        """

        return conn.execute(query, params).df()

    else:
        df = events.copy()
        OPEN_PLAY = ['Regular Play', 'From Counter']

        df = df[
            (df['type'] == 'Carry') &
            (df['location_x'].notna()) &
            (df['carry_end_location_x'].notna()) &
            (df['carry_end_location_y'].notna()) &
            (df['play_pattern'].isin(OPEN_PLAY))
        ].copy()

        if match_id is not None:
            df = df[df['match_id'] == match_id]
        if player is not None:
            df = df[df['player'] == player]

        df['distance_forward'] = df['carry_end_location_x'] - df['location_x']
        df['carry_distance'] = np.sqrt(
            (df['carry_end_location_x'] - df['location_x'])**2 +
            (df['carry_end_location_y'] - df['location_y'])**2
        )

        into_box = ((df['carry_end_location_x'] >= 102) &
                    (df['carry_end_location_y'] >= 18) & (df['carry_end_location_y'] <= 62))
        in_final_40 = ((df['location_x'] >= FINAL_40_PERCENT) &
                       (df['location_x'] >= ATTACKING_60_PERCENT) & (df['distance_forward'] >= 5))
        outside_final_40 = ((df['location_x'] < FINAL_40_PERCENT) &
                            (df['location_x'] >= ATTACKING_60_PERCENT) & (df['distance_forward'] >= 10))

        df['is_progressive'] = (into_box | in_final_40 | outside_final_40).astype(int)
        df['into_final_third'] = (df['carry_end_location_x'] >= FINAL_THIRD).astype(int)
        df['into_penalty_area'] = into_box.astype(int)
        df['carry_directness'] = (df['distance_forward'] / df['carry_distance'].replace(0, np.nan)) * 100

        result = df.groupby(['match_id', 'team', 'player']).agg(
            total_carries=('type', 'count'),
            progressive_carries=('is_progressive', 'sum'),
            avg_progressive_distance=(
                'distance_forward',
                lambda x: round(x[df.loc[x.index, 'is_progressive'] == 1].mean(), 2)
                if (df.loc[x.index, 'is_progressive'] == 1).any() else np.nan
            ),
            avg_progressive_carry_length=(
                'carry_distance',
                lambda x: round(x[df.loc[x.index, 'is_progressive'] == 1].mean(), 2)
                if (df.loc[x.index, 'is_progressive'] == 1).any() else np.nan
            ),
            progressive_carry_directness_pct=(
                'carry_directness',
                lambda x: round(x[df.loc[x.index, 'is_progressive'] == 1].mean(), 2)
                if (df.loc[x.index, 'is_progressive'] == 1).any() else np.nan
            ),
            progressive_carries_into_final_third=(
                'is_progressive',
                lambda x: ((df.loc[x.index, 'is_progressive'] == 1) &
                           (df.loc[x.index, 'into_final_third'] == 1)).sum()
            ),
            progressive_carries_into_penalty_area=(
                'is_progressive',
                lambda x: ((df.loc[x.index, 'is_progressive'] == 1) &
                           (df.loc[x.index, 'into_penalty_area'] == 1)).sum()
            )
        ).reset_index()

        result['progressive_carry_pct'] = round(
            result['progressive_carries'] * 100.0 / result['total_carries'], 2)

        return result[result['progressive_carries'] > 0].sort_values('progressive_carries', ascending=False)


def calculate_progressive_passes_received(events, conn=None, matches=None, match_id=None, player=None):
    """Calculate progressive passes received."""

    ZONE_14_X_MIN, ZONE_14_X_MAX = 80, 102
    ZONE_14_Y_MIN, ZONE_14_Y_MAX = 25, 55
    FINAL_THIRD_X = 80
    ATTACKING_60_PERCENT = 48

    if isinstance(events, str):
        if conn is None:
            conn = duckdb.connect()

        where_parts = ["e.type = 'Pass'", "e.pass_outcome IS NULL", "e.pass_recipient IS NOT NULL"]
        params = []

        if match_id is not None:
            where_parts.append("e.match_id = ?")
            params.append(match_id)
        if player is not None:
            where_parts.append("e.pass_recipient = ?")
            params.append(player)

        where_clause = " AND ".join(where_parts)
        season_join = f"LEFT JOIN '{matches}' m ON r.match_id = m.match_id" if matches else ""
        season_select = "m.season_name," if matches else ""

        query = f"""
        WITH progressive_calc AS (
            SELECT
                e.match_id, e.team, e.pass_recipient as player,
                e.location_x, e.pass_end_location_x, e.pass_end_location_y,
                (e.pass_end_location_x - e.location_x) AS distance_forward,
                CASE
                    WHEN e.pass_end_location_x >= 102 AND e.pass_end_location_y BETWEEN 18 AND 62 THEN 1
                    WHEN e.location_x >= {ATTACKING_60_PERCENT} AND (e.pass_end_location_x - e.location_x) >= 10 THEN 1
                    ELSE 0
                END AS is_progressive,
                CASE WHEN e.pass_end_location_x >= {FINAL_THIRD_X} THEN 1 ELSE 0 END as in_final_third,
                CASE WHEN e.pass_end_location_x BETWEEN {ZONE_14_X_MIN} AND {ZONE_14_X_MAX}
                          AND e.pass_end_location_y BETWEEN {ZONE_14_Y_MIN} AND {ZONE_14_Y_MAX} THEN 1 ELSE 0 END as in_zone_14
            FROM '{events}' e
            WHERE {where_clause}
              AND e.play_pattern IS NOT NULL
              AND e.play_pattern IN ('Regular Play', 'From Counter')
              AND e.location_x IS NOT NULL
              AND e.pass_end_location_x IS NOT NULL
              AND e.pass_end_location_y IS NOT NULL
        ),
        r AS (
            SELECT
                match_id, team, player,
                COUNT(*) AS total_passes_received,
                SUM(is_progressive) AS progressive_passes_received,
                ROUND(SUM(is_progressive) * 100.0 / COUNT(*), 2) AS progressive_passes_received_pct,
                SUM(CASE WHEN is_progressive = 1 AND in_final_third = 1 THEN 1 ELSE 0 END) as progressive_receptions_final_third,
                SUM(CASE WHEN is_progressive = 1 AND in_zone_14 = 1 THEN 1 ELSE 0 END) as progressive_receptions_zone_14,
                ROUND(AVG(CASE WHEN is_progressive = 1 THEN pass_end_location_x END), 2) AS avg_reception_x,
                ROUND(AVG(CASE WHEN is_progressive = 1 THEN pass_end_location_y END), 2) AS avg_reception_y,
                ROUND(AVG(CASE WHEN is_progressive = 1 THEN distance_forward END), 2) AS avg_progressive_distance
            FROM progressive_calc
            GROUP BY match_id, team, player
            HAVING SUM(is_progressive) > 0
        )
        SELECT {season_select} r.*
        FROM r
        {season_join}
        ORDER BY progressive_passes_received DESC
        """

        return conn.execute(query, params).df()

    else:
        df = events.copy()
        OPEN_PLAY = ['Regular Play', 'From Counter']

        df = df[
            (df["type"] == "Pass") & (df["pass_outcome"].isna()) &
            (df["location_x"].notna()) & (df["pass_end_location_x"].notna()) &
            (df["pass_end_location_y"].notna()) & (df["pass_recipient"].notna()) &
            (df['play_pattern'].isin(OPEN_PLAY))
        ].copy()

        if match_id is not None:
            df = df[df["match_id"] == match_id]
        if player is not None:
            df = df[df["pass_recipient"] == player]

        df["distance_forward"] = df["pass_end_location_x"] - df["location_x"]

        into_box = ((df["pass_end_location_x"] >= 102) &
                    (df["pass_end_location_y"] >= 18) & (df["pass_end_location_y"] <= 62))
        forward_from_attacking_60 = ((df['location_x'] >= ATTACKING_60_PERCENT) & (df['distance_forward'] >= 10))

        df["is_progressive"] = (into_box | forward_from_attacking_60).astype(int)
        df['in_final_third'] = (df['pass_end_location_x'] >= FINAL_THIRD_X).astype(int)
        df['in_zone_14'] = (
            (df['pass_end_location_x'] >= ZONE_14_X_MIN) & (df['pass_end_location_x'] <= ZONE_14_X_MAX) &
            (df['pass_end_location_y'] >= ZONE_14_Y_MIN) & (df['pass_end_location_y'] <= ZONE_14_Y_MAX)
        ).astype(int)

        grouped = (
            df.groupby(["match_id", "team", "pass_recipient"]).agg(
                total_passes_received=("type", "count"),
                progressive_passes_received=("is_progressive", "sum"),
                progressive_receptions_final_third=(
                    "is_progressive",
                    lambda x: ((df.loc[x.index, "is_progressive"] == 1) &
                               (df.loc[x.index, "in_final_third"] == 1)).sum()
                ),
                progressive_receptions_zone_14=(
                    "is_progressive",
                    lambda x: ((df.loc[x.index, "is_progressive"] == 1) &
                               (df.loc[x.index, "in_zone_14"] == 1)).sum()
                ),
            ).reset_index().rename(columns={"pass_recipient": "player"})
        )

        grouped["progressive_passes_received_pct"] = (
            grouped["progressive_passes_received"] * 100.0 / grouped["total_passes_received"]).round(2)

        prog_only = df[df["is_progressive"] == 1]
        extra = (
            prog_only.groupby(["match_id", "team", "pass_recipient"]).agg(
                avg_reception_x=("pass_end_location_x", "mean"),
                avg_reception_y=("pass_end_location_y", "mean"),
                avg_progressive_distance=("distance_forward", "mean"),
            ).round(2).reset_index().rename(columns={"pass_recipient": "player"})
        )

        result = grouped.merge(extra, on=["match_id", "team", "player"], how="left")
        return result[result["progressive_passes_received"] > 0].sort_values(
            "progressive_passes_received", ascending=False)


def calculate_progressive_actions(events, conn=None, matches=None, match_id=None, player=None) -> pd.DataFrame:
    """Combined progressive involvement = passes + carries + received."""

    prog_passes = calculate_progressive_passes(events, conn, matches=matches, match_id=match_id, player=player)
    prog_carries = calculate_progressive_carries(events, conn, matches=matches, match_id=match_id, player=player)
    prog_received = calculate_progressive_passes_received(events, conn, matches=matches, match_id=match_id, player=player)

    result = prog_passes.merge(
        prog_carries[['match_id', 'team', 'player', 'progressive_carries']],
        on=['match_id', 'team', 'player'], how='outer'
    ).merge(
        prog_received[['match_id', 'team', 'player', 'progressive_passes_received']],
        on=['match_id', 'team', 'player'], how='outer'
    ).fillna({'progressive_passes': 0, 'progressive_carries': 0, 'progressive_passes_received': 0})

    result['progressive_actions'] = (
        result['progressive_passes'] + result['progressive_carries'] + result['progressive_passes_received']
    )

    keep_cols = ['match_id', 'team', 'player', 'progressive_passes',
                 'progressive_carries', 'progressive_passes_received', 'progressive_actions']
    if 'season_name' in result.columns:
        keep_cols = ['season_name'] + keep_cols

    return result[keep_cols].sort_values('progressive_actions', ascending=False)


def calculate_progressive_actions_no_overlap(events, conn=None, matches=None, match_id=None, player=None) -> pd.DataFrame:
    """Calculate progressive actions while avoiding double-counting overlapping territory."""

    if isinstance(events, str):
        if conn is None:
            conn = duckdb.connect()

        filters = ["e.type IN ('Pass', 'Carry')"]
        params = []

        if match_id is not None:
            filters.append("e.match_id = ?")
            params.append(match_id)
        if player is not None:
            filters.append("e.player = ?")
            params.append(player)

        where_clause = " AND ".join(filters)
        season_join = f"LEFT JOIN '{matches}' m ON r.match_id = m.match_id" if matches else ""
        season_select = "m.season_name," if matches else ""

        query = f"""
        WITH player_actions AS (
            SELECT 
                e.match_id, e.team, e.player,
                CAST(e.match_id AS VARCHAR) || '_' || CAST(e.possession AS VARCHAR) as possession_id,
                e.location_x as start_x,
                COALESCE(e.pass_end_location_x, e.carry_end_location_x) as end_x,
                e.type, e.index_num,
                CASE 
                    WHEN e.type = 'Pass' AND e.pass_outcome IS NULL THEN 1
                    WHEN e.type = 'Carry' THEN 1
                    ELSE 0
                END as is_valid_action
            FROM '{events}' e
            WHERE {where_clause}
              AND e.play_pattern IN ('Regular Play', 'From Counter')
              AND e.location_x >= 48
              AND ((e.type = 'Pass' AND e.pass_end_location_x IS NOT NULL)
                OR (e.type = 'Carry' AND e.carry_end_location_x IS NOT NULL))
        ),
        possession_contributions AS (
            SELECT 
                possession_id, match_id, team, player,
                MIN(start_x) as contribution_start_x,
                MAX(end_x) as contribution_end_x,
                COUNT(*) as total_actions,
                SUM(CASE WHEN type = 'Pass' THEN 1 ELSE 0 END) as passes,
                SUM(CASE WHEN type = 'Carry' THEN 1 ELSE 0 END) as carries
            FROM player_actions
            WHERE is_valid_action = 1
            GROUP BY possession_id, match_id, team, player
        ),
        progressive_contributions AS (
            SELECT *,
                (contribution_end_x - contribution_start_x) as unique_progressive_distance,
                CASE WHEN (contribution_end_x - contribution_start_x) >= 10 THEN 1 ELSE 0 END as is_progressive_contribution
            FROM possession_contributions
        ),
        r AS (
            SELECT 
                match_id, team, player,
                COUNT(*) as possessions_contributed,
                SUM(is_progressive_contribution) as progressive_possessions,
                ROUND(SUM(unique_progressive_distance), 2) as total_unique_progressive_distance,
                ROUND(AVG(CASE WHEN is_progressive_contribution = 1 THEN unique_progressive_distance END), 2) as avg_unique_progressive_distance,
                ROUND(SUM(passes) * 1.0 / SUM(total_actions) * 100, 2) as progressive_action_pass_pct,
                ROUND(SUM(carries) * 1.0 / SUM(total_actions) * 100, 2) as progressive_action_carry_pct
            FROM progressive_contributions
            GROUP BY match_id, team, player
            HAVING SUM(is_progressive_contribution) > 0
        )
        SELECT {season_select} r.*
        FROM r
        {season_join}
        ORDER BY progressive_possessions DESC
        """

        return conn.execute(query, params).df()

    else:
        df = events.copy()
        OPEN_PLAY = ['Regular Play', 'From Counter']

        df = df[
            (df['type'].isin(['Pass', 'Carry'])) &
            (df['play_pattern'].isin(OPEN_PLAY)) &
            (df['location_x'] >= 48) &
            (((df['type'] == 'Pass') & (df['pass_end_location_x'].notna())) |
             ((df['type'] == 'Carry') & (df['carry_end_location_x'].notna())))
        ].copy()

        if match_id is not None:
            df = df[df['match_id'] == match_id]
        if player is not None:
            df = df[df['player'] == player]

        df['possession_id'] = df['match_id'].astype(str) + '_' + df['possession'].astype(str)
        df['end_x'] = df['pass_end_location_x'].fillna(df['carry_end_location_x'])

        df = df[((df['type'] == 'Pass') & (df['pass_outcome'].isna())) | (df['type'] == 'Carry')].copy()

        possession_contributions = df.groupby(['match_id', 'team', 'player', 'possession_id']).agg(
            contribution_start_x=('location_x', 'min'),
            contribution_end_x=('end_x', 'max'),
            total_actions=('type', 'count'),
            passes=('type', lambda x: (x == 'Pass').sum()),
            carries=('type', lambda x: (x == 'Carry').sum())
        ).reset_index()

        possession_contributions['unique_progressive_distance'] = (
            possession_contributions['contribution_end_x'] - possession_contributions['contribution_start_x'])
        possession_contributions['is_progressive_contribution'] = (
            possession_contributions['unique_progressive_distance'] >= 10).astype(int)

        result = possession_contributions.groupby(['match_id', 'team', 'player']).agg(
            possessions_contributed=('possession_id', 'count'),
            progressive_possessions=('is_progressive_contribution', 'sum'),
            total_unique_progressive_distance=('unique_progressive_distance', 'sum'),
            avg_unique_progressive_distance=(
                'unique_progressive_distance',
                lambda x: round(x[possession_contributions.loc[x.index, 'is_progressive_contribution'] == 1].mean(), 2)
                if (possession_contributions.loc[x.index, 'is_progressive_contribution'] == 1).any() else np.nan
            ),
            total_passes=('passes', 'sum'),
            total_carries=('carries', 'sum'),
            total_actions=('total_actions', 'sum')
        ).reset_index()

        result['progressive_action_pass_pct'] = round(result['total_passes'] * 100.0 / result['total_actions'], 2)
        result['progressive_action_carry_pct'] = round(result['total_carries'] * 100.0 / result['total_actions'], 2)

        return result[result['progressive_possessions'] > 0].sort_values('progressive_possessions', ascending=False)


def analyze_progression_profile(events, conn=None, matches=None, match_id=None, min_minutes=30):
    """Analyze player progression profiles with per-90 normalization."""

    prog_passes = calculate_progressive_passes(events, conn, matches=matches, match_id=match_id)
    prog_carries = calculate_progressive_carries(events, conn, matches=matches, match_id=match_id)
    prog_received = calculate_progressive_passes_received(events, conn, matches=matches, match_id=match_id)

    if isinstance(events, str):
        if conn is None:
            conn = duckdb.connect()
        mins_query = f"""
            WITH match_mins AS (
                SELECT match_id, team, player, (MAX(minute) - MIN(minute)) as m
                FROM '{events}' WHERE player IS NOT NULL GROUP BY 1, 2, 3
            )
            SELECT team, player, SUM(m) as total_mins FROM match_mins GROUP BY 1, 2
        """
        player_mins = conn.execute(mins_query).df()
    else:
        player_mins = events[events['player'].notna()].groupby(['match_id', 'team', 'player'])['minute'].agg(
            lambda x: x.max() - x.min()
        ).reset_index()
        player_mins = player_mins.groupby(['team', 'player'])['minute'].sum().reset_index(name='total_mins')

    p_agg = prog_passes.groupby(['team', 'player'])['progressive_passes'].sum().reset_index()
    c_agg = prog_carries.groupby(['team', 'player'])['progressive_carries'].sum().reset_index()
    r_agg = prog_received.groupby(['team', 'player'])['progressive_passes_received'].sum().reset_index()

    result = (
        player_mins.merge(p_agg, on=["team", "player"], how="left")
        .merge(c_agg, on=["team", "player"], how="left")
        .merge(r_agg, on=["team", "player"], how="left")
        .fillna(0)
    )

    result = result[result["total_mins"] >= min_minutes].copy()

    result["progressive_passes_p90"] = round((result["progressive_passes"] / result["total_mins"]) * 90, 2)
    result["progressive_carries_p90"] = round((result["progressive_carries"] / result["total_mins"]) * 90, 2)
    result["progressive_passes_received_p90"] = round((result["progressive_passes_received"] / result["total_mins"]) * 90, 2)
    result["total_progressive_actions_p90"] = (
        result["progressive_passes_p90"] + result["progressive_carries_p90"] + result["progressive_passes_received_p90"])

    pass_p75 = result["progressive_passes_p90"].quantile(0.75)
    carry_p75 = result["progressive_carries_p90"].quantile(0.75)
    recv_p75 = result["progressive_passes_received_p90"].quantile(0.75)
    pass_p50 = result["progressive_passes_p90"].quantile(0.50)
    carry_p50 = result["progressive_carries_p90"].quantile(0.50)

    def classify_progression_type(row):
        p, c, r = row["progressive_passes_p90"], row["progressive_carries_p90"], row["progressive_passes_received_p90"]
        if p >= pass_p75 and c >= carry_p75: return "Complete Progressor"
        if p >= pass_p75: return "Progressive Passer"
        if c >= carry_p75: return "Ball Carrier"
        if r >= recv_p75: return "Progression Outlet"
        if p >= pass_p50 or c >= carry_p50: return "Supporting Progressor"
        return "Limited Progression"

    result["progression_type"] = result.apply(classify_progression_type, axis=1)
    return result.sort_values("total_progressive_actions_p90", ascending=False)


def calculate_team_progression_summary(events, conn=None, matches=None, match_id=None) -> pd.DataFrame:
    """Aggregate progressive actions to TEAM level per match."""

    prog_actions = calculate_progressive_actions(events, conn, matches=matches, match_id=match_id)

    group_cols = ['match_id', 'team']
    if 'season_name' in prog_actions.columns:
        group_cols = ['season_name'] + group_cols

    team_summary = prog_actions.groupby(group_cols).agg({
        'progressive_passes': 'sum',
        'progressive_carries': 'sum',
        'progressive_passes_received': 'sum',
        'progressive_actions': 'sum'
    }).reset_index()

    team_summary['progressive_carry_pct'] = round(
        team_summary['progressive_carries'] * 100.0 / team_summary['progressive_actions'], 2)
    team_summary['progressive_pass_pct'] = round(
        team_summary['progressive_passes'] * 100.0 / team_summary['progressive_actions'], 2)

    return team_summary


def calculate_team_progression_detail(events, conn=None, matches=None, match_id=None) -> pd.DataFrame:
    """Detailed team-level progression metrics including carries and passes separately."""

    prog_passes = calculate_progressive_passes(events, conn, matches=matches, match_id=match_id)
    prog_carries = calculate_progressive_carries(events, conn, matches=matches, match_id=match_id)

    group_cols = ['match_id', 'team']
    if 'season_name' in prog_passes.columns:
        group_cols = ['season_name'] + group_cols

    team_passes = prog_passes.groupby(group_cols).agg({
        'progressive_passes': 'sum', 'total_passes': 'sum',
        'avg_progressive_distance': 'mean', 'avg_progressive_pass_length': 'mean'
    }).reset_index().rename(columns={'avg_progressive_distance': 'avg_progressive_pass_distance'})

    team_carries = prog_carries.groupby(group_cols).agg({
        'progressive_carries': 'sum', 'total_carries': 'sum',
        'avg_progressive_distance': 'mean', 'avg_progressive_carry_length': 'mean',
        'progressive_carries_into_final_third': 'sum', 'progressive_carries_into_penalty_area': 'sum'
    }).reset_index().rename(columns={'avg_progressive_distance': 'avg_progressive_carry_distance'})

    team_detail = team_passes.merge(team_carries, on=group_cols, how='outer').fillna(0)

    team_detail['progression_method_ratio'] = round(
        team_detail['progressive_carries'] /
        (team_detail['progressive_carries'] + team_detail['progressive_passes']), 3)
    team_detail['total_progressive_actions'] = (
        team_detail['progressive_carries'] + team_detail['progressive_passes'])

    return team_detail