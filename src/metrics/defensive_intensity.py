import pandas as pd
import numpy as np
import duckdb

def calculate_pressure_metrics(events, conn=None, match_id=None, player=None) -> pd.DataFrame:
    """
    Calculate Pressure and Counter-pressing metrics from StatsBomb data.
    
    A 'Counterpress' is a StatsBomb-specific flag for pressures within 5s of a turnover.
    A 'Regain' is defined as the team winning the ball within 5 seconds of the pressure.
    """
    
    if isinstance(events, str):
        # --- DUCKDB IMPLEMENTATION (High Performance) ---
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

        query = f"""
        WITH pressure_base AS (
            SELECT 
                e.match_id, e.team, e.player, e.id, e.index_num, e.location_x, e.minute,
                COALESCE(e.counterpress, False) as is_counterpress,
                -- Look ahead 5 events to see if team regained possession
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
        GROUP BY 1, 2, 3, pm.mins
        ORDER BY total_pressures DESC
        """
        return conn.execute(query, params).df()

    else:
        # --- PANDAS IMPLEMENTATION ---
        df = events.copy()
        
        if match_id:
            df = df[df['match_id'] == match_id]
        if player:
            df = df[df['player'] == player]

        # Identify Regains (team wins ball within 5 events after pressure)
        # We sort by index to ensure chronological order
        df = df.sort_values(['match_id', 'index_num'])
        
        regain_types = ['Interception', 'Tackle', 'Ball Recovery', 'Duel', 'Clearance']
        df['is_regain_event'] = df['type'].isin(regain_types).astype(int)
        
        # Check next 5 rows for a regain by the same team
        # Note: This is a simplified rolling check
        df['regain_within_5'] = False
        for i in range(1, 6):
            df['regain_within_5'] = df['regain_within_5'] | (
                (df['is_regain_event'].shift(-i) == 1) & 
                (df['team'].shift(-i) == df['team']) &
                (df['match_id'].shift(-i) == df['match_id'])
            )

        pressures = df[df['type'] == 'Pressure'].copy()
        
        # Calculate Minutes
        mins = df.groupby(['match_id', 'team', 'player'])['minute'].agg(lambda x: x.max() - x.min()).reset_index(name='mins')

        # Aggregation
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