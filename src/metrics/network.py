"""Pass network centrality - measures player involvement in build-up."""

import pandas as pd
import duckdb
import numpy as np
from typing import Union, Optional


def calculate_pass_network_centrality(
    events: Union[pd.DataFrame, str],
    conn: duckdb.DuckDBPyConnection = None,
    matches: Optional[str] = None,
    match_id: Optional[int] = None,
    team: Optional[str] = None,
) -> pd.DataFrame:
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
        )
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
            ) as network_involvement_pct
        FROM player_stats ps
        FULL OUTER JOIN received_stats rs
            ON ps.match_id = rs.match_id
            AND ps.team = rs.team
            AND ps.player = rs.player
        LEFT JOIN team_totals tt
            ON COALESCE(ps.match_id, rs.match_id) = tt.match_id
            AND COALESCE(ps.team, rs.team) = tt.team
        {season_join}
        WHERE COALESCE(ps.passes_made, 0) + COALESCE(rs.passes_received, 0) > 0
        ORDER BY total_pass_involvement DESC
        """

        return conn.execute(query).df()

    else:
        df = events.copy()

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

        result['total_pass_involvement'] = result['passes_made'] + result['passes_received']
        result['degree_centrality'] = result['unique_receivers'] + result['unique_passers']
        result['network_involvement_pct'] = (
            result['total_pass_involvement'] / result['team_total_passes'] * 100
        ).round(2)

        result = result[result['total_pass_involvement'] > 0]

        return result.sort_values('total_pass_involvement', ascending=False)


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