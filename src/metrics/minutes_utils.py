"""
minutes_utils.py
----------------
Single source of truth for minutes_played computation across all metric files.

Source: StatsBomb lineups parquet file.
  - Each row is a player interval within a match (a player can have multiple
    rows if they were involved in a booking/position change mid-match).
  - from_time / to_time are "MM:SS" strings; to_time = null means played to
    end of match, filled with match duration from events.
  - minutes_played per player per match = SUM of all interval durations.

Both a SQL CTE path and a DataFrame path are provided for compatibility with
the existing metric functions.
"""

import pandas as pd
import numpy as np
from typing import Optional


# ---------------------------------------------------------------------------
# Internal helper: parse "MM:SS" -> total seconds
# ---------------------------------------------------------------------------

def _mmss_to_seconds(mmss) -> float:
    """'67:48' -> 4068.0, '00:00' -> 0.0, None -> nan"""
    if mmss is None or (isinstance(mmss, float) and np.isnan(mmss)):
        return np.nan
    parts = str(mmss).split(':')
    return int(parts[0]) * 60 + float(parts[1])


# ---------------------------------------------------------------------------
# SQL CTE path
# ---------------------------------------------------------------------------

def minutes_played_cte(lineups_path: str, events_path: str) -> str:
    # Guard: if lineups not available, return a dummy CTE
    if not lineups_path or lineups_path == 'None':
        return """
    minutes_played_cte AS (
        SELECT NULL::INTEGER as match_id, NULL::VARCHAR as team, 
               NULL::VARCHAR as player, 0 as minutes_played
        WHERE 1=0
    )"""
    """
    Return a SQL CTE string computing minutes_played per player per match
    from the lineups file.

    Parameters
    ----------
    lineups_path : path to lineups parquet
    events_path  : path to events parquet (used for match duration fallback)

    Usage — caller opens the WITH clause:
        query = f\"\"\"
        WITH {minutes_played_cte(lineups_path, events_path)},
        next_cte AS (...)
        SELECT ...
        \"\"\"
    """
    return f"""
    minutes_played_cte AS (
        WITH match_duration AS (
            SELECT match_id, MAX(minute * 60 + second) AS match_seconds
            FROM '{events_path}'
            GROUP BY match_id
        ),
        lineup_seconds AS (
            SELECT
                l.match_id,
                l.team_name AS team,
                l.player_name AS player,
                COALESCE(
                    CAST(SPLIT_PART(l.from_time, ':', 1) AS INTEGER) * 60
                    + CAST(SPLIT_PART(l.from_time, ':', 2) AS FLOAT),
                    0
                ) AS from_seconds,
                COALESCE(
                    CAST(SPLIT_PART(l.to_time, ':', 1) AS INTEGER) * 60
                    + CAST(SPLIT_PART(l.to_time, ':', 2) AS FLOAT),
                    md.match_seconds
                ) AS to_seconds
            FROM '{lineups_path}' l
            LEFT JOIN match_duration md ON l.match_id = md.match_id
            WHERE l.player_name IS NOT NULL
                AND l.from_time IS NOT NULL
                AND l.match_id IN (SELECT DISTINCT match_id FROM '{events_path}')
        )
        SELECT
            match_id,
            team,
            player,
            CAST(GREATEST(SUM(to_seconds - from_seconds), 0) / 60.0 AS INTEGER) AS minutes_played
        FROM lineup_seconds
        GROUP BY match_id, team, player
    )"""


# ---------------------------------------------------------------------------
# DataFrame path
# ---------------------------------------------------------------------------

def compute_minutes_played_df(lineups, events):
    if lineups is None:
        # Return empty frame with correct schema
        return pd.DataFrame(columns=['match_id', 'team', 'player', 'minutes_played'])
    """
    Compute minutes_played per player per match from the lineups DataFrame.

    Parameters
    ----------
    lineups : lineups DataFrame (from lineups parquet)
    events  : events DataFrame (used for match duration fallback only)

    Returns
    -------
    DataFrame with columns: [match_id, team, player, minutes_played]
    """
    # Match duration in seconds from events
    match_duration = (
        events
        .assign(event_seconds=lambda d: d['minute'] * 60 + d.get('second', 0))
        .groupby('match_id')['event_seconds']
        .max()
        .reset_index(name='match_seconds')
    )

    df = lineups.copy()
    df = df.rename(columns={'team_name': 'team', 'player_name': 'player'})
    df = df[df['player'].notna() & df['from_time'].notna()]
    df = df.merge(match_duration, on='match_id', how='left')

    df['from_seconds'] = df['from_time'].apply(_mmss_to_seconds).fillna(0)
    df['to_seconds'] = df['to_time'].apply(_mmss_to_seconds)
    df['to_seconds'] = df['to_seconds'].fillna(df['match_seconds'])

    df['interval_minutes'] = ((df['to_seconds'] - df['from_seconds']) / 60).clip(lower=0)

    result = (
        df.groupby(['match_id', 'team', 'player'])['interval_minutes']
        .sum()
        .reset_index(name='minutes_played')
    )
    result['minutes_played'] = result['minutes_played'].round(2)

    return result[['match_id', 'team', 'player', 'minutes_played']]