"""
Ball progression metrics: Progressive Passes, Progressive Carries, and Progressive Passes Received
Source: https://statsultra.com/progressive-passes-carries-received-explained/
"""

import pandas as pd
import duckdb
from typing import Union, Optional


def calculate_progressive_passes(events, conn=None, match_id=None, player=None) -> pd.DataFrame:
    # Progressive pass heuristic:
    # 1. Completed pass into the penalty area (end_x >= 102 and 18 <= end_y <= 62), OR
    # 2. Moving the ball at least 10 yards closer to opponent's goal (x-gain >= 10) AND 
    # 3. Excluding passes from defending 40% of pitch (start_x >= 48) to exclude deep buildups
    # 4. Restrict to open play using play_pattern (include Regular Play + From Counter; exclude all restart/set-piece patterns)

    if isinstance(events, str):
        if conn is None:
            conn = duckdb.connect()

        filters = ["e.type = 'Pass'", "e.pass_outcome IS NULL"]
        params = []  

        if match_id is not None:
            filters.append("e.match_id = ?") 
            params.append(match_id)
        if player is not None:
            filters.append("e.player = ?")  
            params.append(player)

        where_clause = " AND ".join(filters)
        
        query = f"""
        WITH progressive_passes_calc AS (
            SELECT 
                e.match_id, e.team, e.player,
                e.location_x, e.location_y,
                e.pass_end_location_x, e.pass_end_location_y,
                (e.pass_end_location_x - e.location_x) as yards_forward,
                CASE 
                    WHEN e.pass_end_location_x >= 102 
                         AND e.pass_end_location_y BETWEEN 18 AND 62 THEN 1
                    WHEN (e.pass_end_location_x - e.location_x) >= 10 
                         AND e.location_x >= 48 THEN 1
                    ELSE 0
                END as is_progressive
            FROM '{events}' e
            WHERE {where_clause}
              AND e.play_pattern IS NOT NULL
              AND e.play_pattern IN ('Regular Play', 'From Counter')
              AND e.location_x IS NOT NULL
              AND e.pass_end_location_y IS NOT NULL
              AND e.pass_end_location_x IS NOT NULL
        )
        SELECT 
            match_id, team, player,
            COUNT(*) as total_passes,
            SUM(is_progressive) as progressive_passes,
            ROUND(SUM(is_progressive) * 100.0 / COUNT(*), 2) as progressive_pass_pct,
            ROUND(AVG(CASE WHEN is_progressive = 1 THEN yards_forward END), 2) as avg_progressive_distance
        FROM progressive_passes_calc
        GROUP BY match_id, team, player
        HAVING SUM(is_progressive) > 0
        ORDER BY progressive_passes DESC
        """
        
        return conn.execute(query, params).df()
    
    else:

        df = events.copy()

        OPEN_PLAY = ['Regular Play', 'From Counter']

        df = df[
            (df['type'] == 'Pass') &
            (df['pass_outcome'].isna()) &
            (df['location_x'].notna()) &
            (df['pass_end_location_x'].notna()) &
            (df['pass_end_location_y'].notna())
        ]

        df = df[df['play_pattern'].isin(OPEN_PLAY)].copy()  
        
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
            ((df['yards_forward'] >= 10) & (df['location_x'] >= 48))
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
    # 1. Any carry into the penalty area (end_x >= 102 and 18 <= end_y <= 62), OR
    # 2. Moving the ball at least 10 yards closer to opponent's goal (gains >= 10 in x-direction and ends past x > 60) AND
    # 3. Excluding carries starting in defending 40% of pitch (start_x >= 48) to exclude deep buildups

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
        
        query = f"""
        WITH progressive_carries_calc AS (
            SELECT 
                e.match_id, e.team, e.player,
                e.location_x, e.location_y,
                e.carry_end_location_x, e.carry_end_location_y,
                (e.carry_end_location_x - e.location_x) as yards_forward,
                CASE 
                    WHEN e.carry_end_location_x >= 102 
                         AND e.carry_end_location_y BETWEEN 18 AND 62 THEN 1
                    WHEN (e.carry_end_location_x - e.location_x) >= 10 
                         AND e.location_x >= 48 THEN 1
                    ELSE 0
                END as is_progressive
            FROM '{events}' e
            WHERE {where_clause}
              AND e.play_pattern IS NOT NULL
              AND e.play_pattern IN ('Regular Play', 'From Counter')
              AND e.location_x IS NOT NULL
              AND e.carry_end_location_y IS NOT NULL
              AND e.carry_end_location_x IS NOT NULL
        )
        SELECT 
            match_id, team, player,
            COUNT(*) as total_carries,
            SUM(is_progressive) as progressive_carries,
            ROUND(SUM(is_progressive) * 100.0 / COUNT(*), 2) as progressive_carry_pct,
            ROUND(AVG(CASE WHEN is_progressive = 1 THEN yards_forward END), 2) as avg_progressive_distance
        FROM progressive_carries_calc
        GROUP BY match_id, team, player
        HAVING SUM(is_progressive) > 0
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
            (df['carry_end_location_y'].notna())
        ]

        df = df[df['play_pattern'].isin(OPEN_PLAY)].copy()

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
            ((df['yards_forward'] >= 10) & (df['location_x'] >= 48))
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


def calculate_progressive_passes_received(events, conn=None, match_id=None, player=None):
    # Progressive passes received - count of completed passes received by a player that meet the progressive definition
    # Progressive pass rule matches calculate_progressive_passes:
    # 1. into box, OR
    # 2. x-gain >= 10 AND start_x >= 48

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

        query = f"""
        WITH progressive_passes_calc AS (
            SELECT
                e.match_id,
                e.team,
                e.pass_recipient AS player,
                e.location_x,
                e.pass_end_location_x,
                e.pass_end_location_y,
                (e.pass_end_location_x - e.location_x) AS x_gain,
                CASE
                    WHEN e.pass_end_location_x >= 102
                         AND e.pass_end_location_y BETWEEN 18 AND 62
                    THEN 1
                    WHEN (e.pass_end_location_x - e.location_x) >= 10
                         AND e.location_x >= 48
                    THEN 1
                    ELSE 0
                END AS is_progressive
            FROM '{events}' e
            WHERE {where_clause}
              AND e.play_pattern IS NOT NULL
              AND e.play_pattern IN ('Regular Play', 'From Counter')
              AND e.location_x IS NOT NULL
              AND e.pass_end_location_x IS NOT NULL
              AND e.pass_end_location_y IS NOT NULL
        )
        SELECT
            match_id,
            team,
            player,
            COUNT(*) AS total_passes_received,
            SUM(is_progressive) AS progressive_passes_received,
            ROUND(SUM(is_progressive) * 100.0 / COUNT(*), 2) AS progressive_passes_received_pct,
            ROUND(AVG(CASE WHEN is_progressive = 1 THEN pass_end_location_x END), 2) AS avg_reception_x,
            ROUND(AVG(CASE WHEN is_progressive = 1 THEN x_gain END), 2) AS avg_progressive_distance
        FROM progressive_passes_calc
        GROUP BY match_id, team, player
        HAVING SUM(is_progressive) > 0
        ORDER BY progressive_passes_received DESC
        """

        return conn.execute(query, params).df()

    else:

        df = events.copy()

        OPEN_PLAY = ['Regular Play', 'From Counter']

        df = df[
            (df["type"] == "Pass") &
            (df["pass_outcome"].isna()) &
            (df["location_x"].notna()) &
            (df["pass_end_location_x"].notna()) &
            (df["pass_end_location_y"].notna()) &
            (df["pass_recipient"].notna())
        ]

        df = df[df['play_pattern'].isin(OPEN_PLAY)].copy()

        if match_id is not None:
            df = df[df["match_id"] == match_id]
        if player is not None:
            df = df[df["pass_recipient"] == player]

        df["x_gain"] = df["pass_end_location_x"] - df["location_x"]

        into_box = (
            (df["pass_end_location_x"] >= 102)
            & (df["pass_end_location_y"] >= 18)
            & (df["pass_end_location_y"] <= 62)
        )
        forward_gain = (df["x_gain"] >= 10) & (df["location_x"] >= 48)

        df["is_progressive"] = (into_box | forward_gain).astype(int)

        grouped = (
            df.groupby(["match_id", "team", "pass_recipient"])
            .agg(
                total_passes_received=("type", "count"),
                progressive_passes_received=("is_progressive", "sum"),
            )
            .reset_index()
            .rename(columns={"pass_recipient": "player"})
        )

        grouped["progressive_passes_received_pct"] = (
            grouped["progressive_passes_received"] * 100.0 / grouped["total_passes_received"]
        ).round(2)

        prog_only = df[df["is_progressive"] == 1]

        extra = (
            prog_only.groupby(["match_id", "team", "pass_recipient"])
            .agg(
                avg_reception_x=("pass_end_location_x", "mean"),
                avg_progressive_distance=("x_gain", "mean"),
            )
            .round(2)
            .reset_index()
            .rename(columns={"pass_recipient": "player"})
        )

        result = grouped.merge(extra, on=["match_id", "team", "player"], how="left")
        result = result[result["progressive_passes_received"] > 0]

        return result.sort_values("progressive_passes_received", ascending=False)


def calculate_progressive_actions(events, conn=None, match_id=None, player=None) -> pd.DataFrame:
    # Combined progressive involvement = passes + carries + received
    
    # 1. Get all three metric components
    prog_passes = calculate_progressive_passes(events, conn, match_id, player)
    prog_carries = calculate_progressive_carries(events, conn, match_id, player)
    prog_received = calculate_progressive_passes_received(events, conn, match_id, player)
    
    # 2. Merge Passes and Carries first
    result = prog_passes.merge(
        prog_carries[['match_id', 'team', 'player', 'progressive_carries']],
        on=['match_id', 'team', 'player'],
        how='outer'
    )
    
    # 3. Merge in the Received metrics
    result = result.merge(
        prog_received[['match_id', 'team', 'player', 'progressive_passes_received']],
        on=['match_id', 'team', 'player'],
        how='outer'
    )

    # 4. Fill NaNs for players who might have one stat but not the others
    result = result.fillna({
        'progressive_passes': 0, 
        'progressive_carries': 0,
        'progressive_passes_received': 0
    })

    # 5. Calculate Total Involvement
    result['progressive_actions'] = (
        result['progressive_passes'] + 
        result['progressive_carries'] + 
        result['progressive_passes_received']
    )

    # Sort by total involvement
    result = result.sort_values('progressive_actions', ascending=False)
    
    return result[['match_id', 'team', 'player', 'progressive_passes', 
                   'progressive_carries', 'progressive_passes_received', 'progressive_actions']]


def analyze_progression_profile(events, conn=None, match_id=None, min_minutes=30):
    # Fetch raw metrics per match
    prog_passes = calculate_progressive_passes(events, conn, match_id)
    prog_carries = calculate_progressive_carries(events, conn, match_id)
    prog_received = calculate_progressive_passes_received(events, conn, match_id)

    # Season-Ready Minutes Aggregation
    if isinstance(events, str):
        if conn is None: conn = duckdb.connect()
        mins_query = f"""
            WITH match_mins AS (
                SELECT match_id, team, player, (MAX(minute) - MIN(minute)) as m
                FROM '{events}' WHERE player IS NOT NULL GROUP BY 1, 2, 3
            )
            SELECT team, player, SUM(m) as total_mins FROM match_mins GROUP BY 1, 2
        """
        player_mins = conn.execute(mins_query).df()
    else:
        # Group by match first to avoid season-span errors, then sum
        player_mins = events[events['player'].notna()].groupby(['match_id', 'team', 'player'])['minute'].agg(lambda x: x.max() - x.min()).reset_index()
        player_mins = player_mins.groupby(['team', 'player'])['minute'].sum().reset_index(name='total_mins')

    # Aggregate metrics to player/team level (strips 'match_id' for clean merge)
    p_agg = prog_passes.groupby(['team', 'player'])['progressive_passes'].sum().reset_index()
    c_agg = prog_carries.groupby(['team', 'player'])['progressive_carries'].sum().reset_index()
    r_agg = prog_received.groupby(['team', 'player'])['progressive_passes_received'].sum().reset_index()

    # Final Merge 
    result = (
        player_mins.merge(p_agg, on=["team", "player"], how="left")
        .merge(c_agg, on=["team", "player"], how="left")
        .merge(r_agg, on=["team", "player"], how="left")
        .fillna(0)
    )

    result = result[result["total_mins"] >= min_minutes].copy()

    # Normalization
    result["progressive_passes_p90"] = round((result["progressive_passes"] / result["total_mins"]) * 90, 2)
    result["progressive_carries_p90"] = round((result["progressive_carries"] / result["total_mins"]) * 90, 2)
    result["progressive_passes_received_p90"] = round((result["progressive_passes_received"] / result["total_mins"]) * 90, 2)
    result["total_progressive_actions_p90"] = result["progressive_passes_p90"] + result["progressive_carries_p90"] + result["progressive_passes_received_p90"]

    # Classification
    pass_p75 = result["progressive_passes_p90"].quantile(0.75); carry_p75 = result["progressive_carries_p90"].quantile(0.75)
    recv_p75 = result["progressive_passes_received_p90"].quantile(0.75); pass_p50 = result["progressive_passes_p90"].quantile(0.50)
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