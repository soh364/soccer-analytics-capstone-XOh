'''Test module to validate that all metric functions can be imported and run without errors on a standard StatsBomb-style dataframe'''
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Set project root
sys.path.insert(0, str(Path(__file__).parent))

def get_test_data():
    # Generates a standard StatsBomb-style dataframe for metric validation

    np.random.seed(42)
    n = 200
    
    df = pd.DataFrame({
        'id': [f"event_{i}" for i in range(n)],
        'match_id': [101] * n,
        'team': np.random.choice(['Home Team', 'Away Team'], n),
        'player': [f'Player_{i % 11}' for i in range(n)],
        'type': np.random.choice(['Pass', 'Carry', 'Pressure', 'Ball Recovery'], n),
        'play_pattern': ['Regular Play'] * n,
        'location_x': np.random.uniform(0, 100, n),
        'location_y': np.random.uniform(0, 80, n),
        'minute': np.random.randint(0, 90, n),
        'pass_outcome': [None] * n,
        'pass_recipient': [f'Player_{(i+1) % 11}' for i in range(n)],
        'possession': (np.arange(n) // 10) + 1
    })

    # Coordinate mapping for progression metrics
    df['pass_end_location_x'] = df['location_x'] + 20
    df['pass_end_location_y'] = df['location_y']
    df['carry_end_location_x'] = df['location_x'] + 10
    df['carry_end_location_y'] = df['location_y']

    # Sample shot data for xG Chain/Buildup validation
    df.loc[n-10:, 'type'] = 'Shot'
    df.loc[n-10:, 'shot_statsbomb_xg'] = 0.15
    df.loc[n-10:, 'shot_outcome'] = 'Goal'
    
    df['shot_statsbomb_xg'] = df['shot_statsbomb_xg'].fillna(0)
    df['shot_outcome'] = df['shot_outcome'].fillna('N/A')

    return df

def run_checks():
    df = get_test_data()
    status = {}

    # File System
    files = [
        'src/metrics/possession.py', 'src/metrics/progression.py',
        'src/metrics/aggregations.py', 'src/metrics/xg_chain.py'
    ]
    status['Filesystem'] = all(Path(f).exists() for f in files)

    # Imports
    try:
        from src.metrics import (
            calculate_progressive_actions, 
            calculate_field_tilt, 
            calculate_xg_chain
        )
        status['Imports'] = True
    except ImportError:
        status['Imports'] = False

    # Progression
    try:
        status['Progression'] = not calculate_progressive_actions(df).empty
    except Exception:
        status['Progression'] = False

    # Possession
    try:
        status['Possession'] = not calculate_field_tilt(df).empty
    except Exception:
        status['Possession'] = False

    # xG Chain
    try:
        status['xG_Chain'] = not calculate_xg_chain(df).empty
    except Exception:
        status['xG_Chain'] = False

    return status

if __name__ == "__main__":
    results = run_checks()
    
    print("-" * 30)
    print("TEST SUMMARY")
    print("-" * 30)
    for key, val in results.items():
        print(f"{key.ljust(15)}: {'[OK]' if val else '[ERR]'}")
    print("-" * 30)

    if not all(results.values()):
        sys.exit(1)