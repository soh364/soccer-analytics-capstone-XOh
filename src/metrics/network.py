"""Pass network centrality - measures player involvement in build-up."""

import pandas as pd
import duckdb
import numpy as np
from typing import Union, Optional
from .minutes_utils import minutes_played_cte as _minutes_played_cte, compute_minutes_played_df as _compute_minutes_played_df_shared


def _get_minutes_played_query(events_path: str, match_id_filter: str = "", team_filter: str = "") -> str:
    """Build a CTE that computes minutes_played per player per match."""
    extra_filters = ""
    if match_id_filter:
        extra_filters += f" AND match_id {match_id_filter}"
    if team_filter:
        extra_filters += f" AND team {team_filter}"

    return f"""
    minutes_played_cte AS (
        -- Each player's starting minute (0 if in Starting XI, else minute of sub-on)
        WITH lineups AS (
            SELECT match_id, team, player,
                CASE WHEN type = 'Starting XI' THEN 0
                     WHEN type = 'Substitution' AND substitution_replacement_name = player THEN minute
                     ELSE NULL
                END AS minute_on
            FROM '{events_path}'
            WHERE type IN ('Starting XI', 'Substitution'){extra_filters}
        ),
        starters AS (
            SELECT DISTINCT match_id, team, player, 0 AS minute_on
            FROM '{events_path}'
            WHERE type = 'Starting XI'{extra_filters}
              AND player IS NOT NULL
        ),
        -- Players who came on as substitutes
        sub_on AS (
            SELECT match_id, team, substitution_replacement_name AS player, minute AS minute_on
            FROM '{events_path}'
            WHERE type = 'Substitution'{extra_filters}
              AND substitution_replacement_name IS NOT NULL
        ),
        all_players AS (
            SELECT match_id, team, player, minute_on FROM starters
            UNION ALL
            SELECT match_id, team, player, minute_on FROM sub_on
        ),
        -- Players who were substituted off or sent off
        minute_off AS (
            -- Substituted off
            SELECT match_id, team, player, minute AS minute_out
            FROM '{events_path}'
            WHERE type = 'Substitution'{extra_filters}
              AND player IS NOT NULL
            UNION ALL
            -- Red card (straight red or second yellow)
            SELECT match_id, team, player, minute AS minute_out
            FROM '{events_path}'
            WHERE type = 'Bad Behaviour'
              AND bad_behaviour_card IN ('Red Card', 'Second Yellow'){extra_filters}
              AND player IS NOT NULL
            UNION ALL
            SELECT match_id, team, player, minute AS minute_out
            FROM '{events_path}'
            WHERE type = 'Foul Committed'
              AND foul_committed_card IN ('Red Card', 'Second Yellow'){extra_filters}
              AND player IS NOT NULL
        ),
        -- Match duration: use max minute from events
        match_duration AS (
            SELECT match_id, MAX(minute) AS total_minutes
            FROM '{events_path}'
            WHERE True{extra_filters}
            GROUP BY match_id
        )
        SELECT 
            ap.match_id,
            ap.team,
            ap.player,
            ap.minute_on,
            COALESCE(mo.minute_out, md.total_minutes) AS minute_out,
            COALESCE(mo.minute_out, md.total_minutes) - ap.minute_on AS minutes_played
        FROM all_players ap
        LEFT JOIN (
            SELECT match_id, team, player, MIN(minute_out) AS minute_out
            FROM minute_off
            GROUP BY match_id, team, player
        ) mo ON ap.match_id = mo.match_id AND ap.team = mo.team AND ap.player = mo.player
        LEFT JOIN match_duration md ON ap.match_id = md.match_id
    )
    """


def calculate_pass_network_centrality(events, conn=None, matches=None, lineups=None, match_id=None, team=None):

    """
    Calculate pass network centrality metrics for players.
    """
    if isinstance(events, str):
        if conn is None:
            conn = duckdb.connect()

        filters = ["e.type = 'Pass'", "e.pass_outcome IS NULL"]

        if match_id is not None:
            filters.append(f"e.match_id = {match_id}")
        if team is not None:
            filters.append(f"e.team = '{team}'")

        where_clause = " AND ".join(filters)
        season_join = f"LEFT JOIN '{matches}' m ON COALESCE(ps.match_id, rs.match_id) = m.match_id" if matches else ""
        season_select = "m.season_name," if matches else ""

        minutes_cte = _minutes_played_cte(lineups, events)

        query = f"""
        WITH pass_network AS (
            SELECT 
                e.match_id,
                e.team,
                e.player as passer,
                e.pass_recipient as receiver,
                COUNT(*) as pass_count
            FROM '{events}' e
            WHERE {where_clause}
              AND e.player IS NOT NULL
              AND e.pass_recipient IS NOT NULL
            GROUP BY e.match_id, e.team, e.player, e.pass_recipient
        ),
        player_stats AS (
            SELECT 
                match_id, team, passer as player,
                SUM(pass_count) as passes_made,
                COUNT(DISTINCT receiver) as unique_receivers
            FROM pass_network
            GROUP BY match_id, team, passer
        ),
        received_stats AS (
            SELECT 
                match_id, team, receiver as player,
                SUM(pass_count) as passes_received,
                COUNT(DISTINCT passer) as unique_passers
            FROM pass_network
            GROUP BY match_id, team, receiver
        ),
        team_totals AS (
            SELECT match_id, team, SUM(pass_count) as team_total_passes
            FROM pass_network
            GROUP BY match_id, team
        ),
        {minutes_cte}
        SELECT 
            {season_select}
            COALESCE(ps.match_id, rs.match_id) as match_id,
            COALESCE(ps.team, rs.team) as team,
            COALESCE(ps.player, rs.player) as player,
            COALESCE(ps.passes_made, 0) as passes_made,
            COALESCE(rs.passes_received, 0) as passes_received,
            COALESCE(ps.passes_made, 0) + COALESCE(rs.passes_received, 0) as total_pass_involvement,
            COALESCE(ps.unique_receivers, 0) as unique_receivers,
            COALESCE(rs.unique_passers, 0) as unique_passers,
            COALESCE(ps.unique_receivers, 0) + COALESCE(rs.unique_passers, 0) as degree_centrality,
            ROUND(
                (COALESCE(ps.passes_made, 0) + COALESCE(rs.passes_received, 0)) * 100.0 /
                tt.team_total_passes, 2
            ) as network_involvement_pct,
            COALESCE(mp.minutes_played, 0) as minutes_played
        FROM player_stats ps
        FULL OUTER JOIN received_stats rs
            ON ps.match_id = rs.match_id
            AND ps.team = rs.team
            AND ps.player = rs.player
        LEFT JOIN team_totals tt
            ON COALESCE(ps.match_id, rs.match_id) = tt.match_id
            AND COALESCE(ps.team, rs.team) = tt.team
        LEFT JOIN minutes_played_cte mp
            ON COALESCE(ps.match_id, rs.match_id) = mp.match_id
            AND COALESCE(ps.team, rs.team) = mp.team
            AND COALESCE(ps.player, rs.player) = mp.player
        {season_join}
        WHERE COALESCE(ps.passes_made, 0) + COALESCE(rs.passes_received, 0) > 0
        ORDER BY total_pass_involvement DESC
        """

        return conn.execute(query).df()

    else:
        df = events.copy()

        # --- Compute minutes_played from the DataFrame ---
        minutes_df = _compute_minutes_played_df_shared(lineups, events)

        df = df[
            (df['type'] == 'Pass') &
            (df['pass_outcome'].isna()) &
            (df['player'].notna()) &
            (df['pass_recipient'].notna())
        ]

        if match_id is not None:
            df = df[df['match_id'] == match_id]
        if team is not None:
            df = df[df['team'] == team]

        pass_network = df.groupby(['match_id', 'team', 'player', 'pass_recipient']).size().reset_index(name='pass_count')

        passes_made = pass_network.groupby(['match_id', 'team', 'player']).agg(
            passes_made=('pass_count', 'sum'),
            unique_receivers=('pass_recipient', 'nunique')
        ).reset_index()

        passes_received = pass_network.groupby(['match_id', 'team', 'pass_recipient']).agg(
            passes_received=('pass_count', 'sum'),
            unique_passers=('player', 'nunique')
        ).reset_index()
        passes_received.rename(columns={'pass_recipient': 'player'}, inplace=True)

        team_totals = pass_network.groupby(['match_id', 'team']).agg(
            team_total_passes=('pass_count', 'sum')
        ).reset_index()

        result = passes_made.merge(passes_received, on=['match_id', 'team', 'player'], how='outer').fillna(0)
        result = result.merge(team_totals, on=['match_id', 'team'])
        result = result.merge(minutes_df, on=['match_id', 'team', 'player'], how='left')

        result['total_pass_involvement'] = result['passes_made'] + result['passes_received']
        result['degree_centrality'] = result['unique_receivers'] + result['unique_passers']
        result['network_involvement_pct'] = (
            result['total_pass_involvement'] / result['team_total_passes'] * 100
        ).round(2)
        result['minutes_played'] = result['minutes_played'].fillna(0).astype(int)

        result = result[result['total_pass_involvement'] > 0]

        return result.sort_values('total_pass_involvement', ascending=False)


def _compute_minutes_played_df(
    events: pd.DataFrame,
    match_id: Optional[int] = None,
    team: Optional[str] = None,
) -> pd.DataFrame:
    """Compute minutes_played per player per match from a DataFrame of events."""
    df = events.copy()

    if match_id is not None:
        df = df[df['match_id'] == match_id]
    if team is not None:
        df = df[df['team'] == team]

    # Match duration per match
    match_duration = df.groupby('match_id')['minute'].max().reset_index()
    match_duration.columns = ['match_id', 'total_minutes']

    # Starters: minute_on = 0
    starters = df[df['type'] == 'Starting XI'][['match_id', 'team', 'player']].dropna(subset=['player']).copy()
    starters = starters.drop_duplicates()
    starters['minute_on'] = 0

    # Substitutes coming on
    subs_on = df[df['type'] == 'Substitution'][
        ['match_id', 'team', 'substitution_replacement_name', 'minute']
    ].dropna(subset=['substitution_replacement_name']).copy()
    subs_on = subs_on.rename(columns={'substitution_replacement_name': 'player', 'minute': 'minute_on'})

    all_players = pd.concat([starters, subs_on[['match_id', 'team', 'player', 'minute_on']]], ignore_index=True)

    # Players substituted off
    subs_off = df[df['type'] == 'Substitution'][['match_id', 'team', 'player', 'minute']].dropna(subset=['player']).copy()
    subs_off = subs_off.rename(columns={'minute': 'minute_out'})

    # Red cards (bad behaviour or foul committed)
    red_cards = pd.DataFrame()
    if 'bad_behaviour_card' in df.columns:
        rc1 = df[df['bad_behaviour_card'].isin(['Red Card', 'Second Yellow'])][
            ['match_id', 'team', 'player', 'minute']
        ].dropna(subset=['player']).copy()
        rc1 = rc1.rename(columns={'minute': 'minute_out'})
        red_cards = pd.concat([red_cards, rc1], ignore_index=True)

    if 'foul_committed_card' in df.columns:
        rc2 = df[df['foul_committed_card'].isin(['Red Card', 'Second Yellow'])][
            ['match_id', 'team', 'player', 'minute']
        ].dropna(subset=['player']).copy()
        rc2 = rc2.rename(columns={'minute': 'minute_out'})
        red_cards = pd.concat([red_cards, rc2], ignore_index=True)

    minute_off = pd.concat([subs_off, red_cards], ignore_index=True)
    # Take the earliest minute_out per player per match (in case of duplicates)
    minute_off = minute_off.groupby(['match_id', 'team', 'player'])['minute_out'].min().reset_index()

    result = all_players.merge(minute_off, on=['match_id', 'team', 'player'], how='left')
    result = result.merge(match_duration, on='match_id', how='left')

    result['minute_out'] = result['minute_out'].fillna(result['total_minutes'])
    result['minutes_played'] = (result['minute_out'] - result['minute_on']).clip(lower=0).astype(int)

    return result[['match_id', 'team', 'player', 'minutes_played']]


def calculate_pass_network_positions(
    events: Union[pd.DataFrame, str],
    conn: duckdb.DuckDBPyConnection = None,
    matches: Optional[str] = None,
    match_id: Optional[int] = None,
    team: Optional[str] = None,
) -> pd.DataFrame:
    """
    Calculate average positions for pass network visualization.
    """
    if isinstance(events, str):
        if conn is None:
            conn = duckdb.connect()

        filters = ["e.type = 'Pass'", "e.pass_outcome IS NULL"]

        if match_id is not None:
            filters.append(f"e.match_id = {match_id}")
        if team is not None:
            filters.append(f"e.team = '{team}'")

        where_clause = " AND ".join(filters)
        season_join = f"LEFT JOIN '{matches}' m ON e.match_id = m.match_id" if matches else ""
        season_select = "m.season_name," if matches else ""
        season_group = "m.season_name," if matches else ""

        query = f"""
        SELECT 
            {season_select}
            e.match_id, e.team, e.player,
            ROUND(AVG(e.location_x), 2) as avg_x,
            ROUND(AVG(e.location_y), 2) as avg_y,
            COUNT(*) as pass_count
        FROM '{events}' e
        {season_join}
        WHERE {where_clause}
          AND e.player IS NOT NULL
          AND e.location_x IS NOT NULL
          AND e.location_y IS NOT NULL
        GROUP BY {season_group} e.match_id, e.team, e.player
        ORDER BY pass_count DESC
        """

        return conn.execute(query).df()

    else:
        df = events.copy()

        df = df[
            (df['type'] == 'Pass') &
            (df['pass_outcome'].isna()) &
            (df['player'].notna()) &
            (df['location_x'].notna()) &
            (df['location_y'].notna())
        ]

        if match_id is not None:
            df = df[df['match_id'] == match_id]
        if team is not None:
            df = df[df['team'] == team]

        result = df.groupby(['match_id', 'team', 'player']).agg(
            avg_x=('location_x', 'mean'),
            avg_y=('location_y', 'mean'),
            pass_count=('type', 'count')
        ).reset_index()

        result['avg_x'] = result['avg_x'].round(2)
        result['avg_y'] = result['avg_y'].round(2)

        return result.sort_values('pass_count', ascending=False)


def classify_network_role(
    events: Union[pd.DataFrame, str],
    conn: duckdb.DuckDBPyConnection = None,
    matches: Optional[str] = None,
    match_id: Optional[int] = None,
    min_involvement: float = 5.0,
) -> pd.DataFrame:
    """
    Classify players by their role in the passing network.
    """
    centrality = calculate_pass_network_centrality(events, conn, matches=matches, match_id=match_id)

    centrality = centrality[centrality['network_involvement_pct'] >= min_involvement]

    if len(centrality) == 0:
        return pd.DataFrame()

    centrality['pass_balance'] = (
        centrality['passes_made'] / (centrality['passes_received'] + 1)
    ).round(2)

    def classify_role(row):
        involvement = row['network_involvement_pct']
        degree = row['degree_centrality']
        balance = row['pass_balance']

        if involvement > 15:
            return 'Hub/Orchestrator'
        elif degree > 10:
            return 'Connector'
        elif balance > 1.5:
            return 'Distributor'
        elif balance < 0.7:
            return 'Target'
        else:
            return 'Balanced'

    centrality['network_role'] = centrality.apply(classify_role, axis=1)

    return centrality.sort_values('network_involvement_pct', ascending=False)