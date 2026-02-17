"""Load and manage StatsBomb parquet data using DuckDB"""

from pathlib import Path
import duckdb
import pandas as pd
from typing import Optional, Union


class DataLoader:
    #Handle loading StatsBomb parquet files with DuckDB
    
    def __init__(self, data_dir=None, verbose=True):
        self.verbose = verbose

        # Default directory layout
        if data_dir is None:
            data_dir = Path("..") / "data" / "Statsbomb"
        
        self.data_dir = Path(data_dir)
        
        if not self.data_dir.exists():
            raise ValueError(f"Data directory not found: {self.data_dir}")
        
        # Initialize DuckDB connection
        self.conn = duckdb.connect()
        
        # Expected parquet files
        self.files = {
            'matches': self.data_dir / 'matches.parquet',
            'events': self.data_dir / 'events.parquet',
            'lineups': self.data_dir / 'lineups.parquet',
            'three_sixty': self.data_dir / 'three_sixty.parquet',
            'reference': self.data_dir / 'reference.parquet',
        }
        
        # Verify files exist
        self._verify_files()
        
    def _verify_files(self):
        # Check which parquet files are available

        self.available_files = {}
        for name, path in self.files.items():
            if path.exists():
                self.available_files[name] = path
                print(f"✓ Found {name}.parquet")
            else:
                print(f"✗ Missing {name}.parquet")
    
    def load_matches(self, as_dataframe=True) -> Union[pd.DataFrame, duckdb.DuckDBPyRelation]:
        # Load matches data 

        if 'matches' not in self.available_files:
            raise ValueError("matches.parquet not found")
        
        query = f"SELECT * FROM '{self.available_files['matches']}'"
        result = self.conn.execute(query)
        
        return result.df() if as_dataframe else result

    def load_events(self, as_dataframe=True) -> Union[pd.DataFrame, duckdb.DuckDBPyRelation]:
        # Load events data

        if 'events' not in self.available_files:
            raise ValueError("events.parquet not found")
        
        query = f"SELECT * FROM '{self.available_files['events']}'"
        result = self.conn.execute(query)
        
        return result.df() if as_dataframe else result
    
    def load_lineups(self, as_dataframe=True) -> Union[pd.DataFrame, duckdb.DuckDBPyRelation]:
        # Load lineups data 

        if 'lineups' not in self.available_files:
            raise ValueError("lineups.parquet not found")
        
        query = f"SELECT * FROM '{self.available_files['lineups']}'"
        result = self.conn.execute(query)
        
        return result.df() if as_dataframe else result
    
    def load_three_sixty(self, as_dataframe=True) -> Union[pd.DataFrame, duckdb.DuckDBPyRelation]:
        # Load 360 tracking data.
        
        if 'three_sixty' not in self.available_files:
            raise ValueError("three_sixty.parquet not found")
        
        query = f"SELECT * FROM '{self.available_files['three_sixty']}'"
        result = self.conn.execute(query)
        
        return result.df() if as_dataframe else result
    
    def query(self, sql: str, as_dataframe=True) -> Union[pd.DataFrame, duckdb.DuckDBPyRelation]:
        # Since we registered views, sql can reference: matches, events, lineups, three_sixty, reference

        query = sql
        for name, path in self.available_files.items():
            query = query.replace(f"FROM {name}", f"FROM '{path}'")
            query = query.replace(f"JOIN {name}", f"JOIN '{path}'")
        
        result = self.conn.execute(query)
        return result.df() if as_dataframe else result
    
    def get_match_events(self, match_id, as_dataframe=True) -> Union[pd.DataFrame, duckdb.DuckDBPyRelation]:
        # Parameterized query to avoid string formatting issues

        query = f"""
        SELECT * 
        FROM '{self.available_files['events']}'
        WHERE match_id = {match_id}
        ORDER BY period, minute, second
        """
        result = self.conn.execute(query)
        return result.df() if as_dataframe else result
    
    def get_player_events(self, player_name, as_dataframe=True) -> Union[pd.DataFrame, duckdb.DuckDBPyRelation]:
        # Parameterized query for safety and quoting

        query = f"""
        SELECT * 
        FROM '{self.available_files['events']}'
        WHERE player = '{player_name}'
        """
        result = self.conn.execute(query)
        return result.df() if as_dataframe else result
    
    def get_data_summary(self) -> dict:
        # Quick counts / sanity checks for what got loaded

        summary = {}
        
        if 'matches' in self.available_files:
            matches_count = self.conn.execute(
                f"SELECT COUNT(*) FROM '{self.available_files['matches']}'"
            ).fetchone()[0]
            summary['total_matches'] = matches_count
            
            competitions = self.conn.execute(
                f"SELECT COUNT(DISTINCT competition_name) FROM '{self.available_files['matches']}'"
            ).fetchone()[0]
            summary['competitions'] = competitions
        
        if 'events' in self.available_files:
            events_count = self.conn.execute(
                f"SELECT COUNT(*) FROM '{self.available_files['events']}'"
            ).fetchone()[0]
            summary['total_events'] = events_count
            
            event_types = self.conn.execute(f"""
                SELECT type, COUNT(*) as count 
                FROM '{self.available_files['events']}'
                GROUP BY type
                ORDER BY count DESC
                LIMIT 10
            """).df()
            summary['top_event_types'] = event_types.to_dict('records')
        
        if 'three_sixty' in self.available_files:
            frames_count = self.conn.execute(
                f"SELECT COUNT(DISTINCT event_uuid) FROM '{self.available_files['three_sixty']}'"
            ).fetchone()[0]
            summary['360_frames'] = frames_count
        
        return summary
    
    def close(self):
        # Close DuckDB connection
        self.conn.close()
    
    def __enter__(self):
        # Allows: with DataLoader(...) as loader:
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# Convenience function for quick loading
def load_data(data_dir=None) -> DataLoader:
    # Small helper so we can just do: loader = load_data()

    return DataLoader(data_dir)