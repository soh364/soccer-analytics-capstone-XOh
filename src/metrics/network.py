"""Pass network centrality - measures player involvement in build-up."""

import pandas as pd
import duckdb
import numpy as np
from typing import Union, Optional


def calculate_pass_network_centrality(
    events: Union[pd.DataFrame, str],
    conn: duckdb.DuckDBPyConnection = None,
    match_id: Optional[int] = None,
    team: Optional[str] = None,
) -> pd.DataFrame:
    """
    Calculate pass network centrality metrics for players.
    
    Measures how central/connected each player is in the team's passing network.
    High centrality = key hub in build-up play (even without assists/goals).
    
    Metrics calculated:
    - Degree centrality: How many different teammates player passes to/from
    - Pass volume: Total passes given + received
    - Betweenness (approx): How often player is "between" other players
    - Network involvement: % of team's total passes touched
    
    Args:
        events: DataFrame of events or path to parquet file
        conn: DuckDB connection (required if events is a file path)
        match_id: Optional match ID to filter by
        team: Optional team name to filter by
        
    Returns:
        DataFrame with centrality metrics per player
    """
    if isinstance(events, str):
        if conn is None:
            conn = duckdb.connect()
        
        filters = ["e.type = 'Pass'", "e.pass_outcome IS NULL"]  # Completed passes
        
        if match_id is not None:
            filters.append(f"e.match_id = {match_id}")
        if team is not None:
            filters.append(f"e.team = '{team}'")
        
        where_clause = " AND ".join(filters)
        
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
            -- Passes given
            SELECT 
                match_id,
                team,
                passer as player,
                SUM(pass_count) as passes_made,
                COUNT(DISTINCT receiver) as unique_receivers
            FROM pass_network
            GROUP BY match_id, team, passer
        ),
        received_stats AS (
            -- Passes received
            SELECT 
                match_id,
                team,
                receiver as player,
                SUM(pass_count) as passes_received,
                COUNT(DISTINCT passer) as unique_passers
            FROM pass_network
            GROUP BY match_id, team, receiver
        ),
        team_totals AS (
            SELECT 
                match_id,
                team,
                SUM(pass_count) as team_total_passes
            FROM pass_network
            GROUP BY match_id, team
        )
        SELECT 
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
        WHERE COALESCE(ps.passes_made, 0) + COALESCE(rs.passes_received, 0) > 0
        ORDER BY total_pass_involvement DESC
        """
        
        return conn.execute(query).df()
    
    else:
        # Pandas implementation
        df = events.copy()
        
        # Filter to completed passes with recipient
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
        
        # Build pass network
        pass_network = df.groupby(['match_id', 'team', 'player', 'pass_recipient']).size().reset_index(name='pass_count')
        
        # Passes made
        passes_made = pass_network.groupby(['match_id', 'team', 'player']).agg(
            passes_made=('pass_count', 'sum'),
            unique_receivers=('pass_recipient', 'nunique')
        ).reset_index()
        
        # Passes received
        passes_received = pass_network.groupby(['match_id', 'team', 'pass_recipient']).agg(
            passes_received=('pass_count', 'sum'),
            unique_passers=('player', 'nunique')
        ).reset_index()
        passes_received.rename(columns={'pass_recipient': 'player'}, inplace=True)
        
        # Team totals
        team_totals = pass_network.groupby(['match_id', 'team']).agg(
            team_total_passes=('pass_count', 'sum')
        ).reset_index()
        
        # Merge all
        result = passes_made.merge(
            passes_received,
            on=['match_id', 'team', 'player'],
            how='outer'
        ).fillna(0)
        
        result = result.merge(team_totals, on=['match_id', 'team'])
        
        # Calculate metrics
        result['total_pass_involvement'] = result['passes_made'] + result['passes_received']
        result['degree_centrality'] = result['unique_receivers'] + result['unique_passers']
        result['network_involvement_pct'] = (
            result['total_pass_involvement'] / result['team_total_passes'] * 100
        ).round(2)
        
        # Filter and sort
        result = result[result['total_pass_involvement'] > 0]
        
        return result.sort_values('total_pass_involvement', ascending=False)


def calculate_pass_network_positions(
    events: Union[pd.DataFrame, str],
    conn: duckdb.DuckDBPyConnection = None,
    match_id: Optional[int] = None,
    team: Optional[str] = None,
) -> pd.DataFrame:
    """
    Calculate average positions for pass network visualization.
    
    Returns each player's average x,y position when making/receiving passes.
    Used for creating pass network diagrams.
    
    Args:
        events: Events data
        conn: DuckDB connection
        match_id: Optional match filter
        team: Optional team filter
        
    Returns:
        DataFrame with player average positions
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
        
        query = f"""
        SELECT 
            match_id,
            team,
            player,
            ROUND(AVG(location_x), 2) as avg_x,
            ROUND(AVG(location_y), 2) as avg_y,
            COUNT(*) as pass_count
        FROM '{events}' e
        WHERE {where_clause}
          AND player IS NOT NULL
          AND location_x IS NOT NULL
          AND location_y IS NOT NULL
        GROUP BY match_id, team, player
        ORDER BY pass_count DESC
        """
        
        return conn.execute(query).df()
    
    else:
        # Pandas implementation
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
    match_id: Optional[int] = None,
    min_involvement: float = 5.0,
) -> pd.DataFrame:
    """
    Classify players by their role in the passing network.
    
    Network roles:
    - Hub/Orchestrator: Very high centrality (>15% involvement)
    - Connector: High degree centrality, links players
    - Distributor: High passes made, many receivers
    - Target: High passes received, fewer made
    - Peripheral: Low involvement
    
    Args:
        events: Events data
        conn: DuckDB connection
        match_id: Optional match filter
        min_involvement: Minimum network involvement % to include
        
    Returns:
        DataFrame with network role classification
    """
    # Get centrality metrics
    centrality = calculate_pass_network_centrality(events, conn, match_id)
    
    # Filter by minimum involvement
    centrality = centrality[centrality['network_involvement_pct'] >= min_involvement]
    
    if len(centrality) == 0:
        return pd.DataFrame()
    
    # Calculate ratios
    centrality['pass_balance'] = (
        centrality['passes_made'] / (centrality['passes_received'] + 1)
    ).round(2)
    
    # Classify role
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