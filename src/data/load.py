import pandas as pd
from pathlib import Path


def load_raw_data(data_dir):
    """Load raw StatsBomb parquet files and return a dict of DataFrames."""
    data_path = Path(data_dir)

    matches = pd.read_parquet(data_path / "matches.parquet")
    events = pd.read_parquet(data_path / "events.parquet")
    lineups = pd.read_parquet(data_path / "lineups.parquet")
    reference = pd.read_parquet(data_path / "reference.parquet")

    print(f"Loaded: matches={len(matches):,}, events={len(events):,}, lineups={len(lineups):,}, reference={len(reference):,}")

    return {
        'matches': matches,
        'events': events,
        'lineups': lineups,
        'reference': reference
    }


def load_processed_data(processed_dir, split='train'):
    processed_path = Path(processed_dir)

    if split == 'all':
        matches = pd.read_parquet(processed_path / "matches_normalized.parquet")
        events = pd.read_parquet(processed_path / "events_normalized.parquet")
        lineups = pd.read_parquet(processed_path / "lineups_normalized.parquet")
    else:
        matches = pd.read_parquet(processed_path / f"matches_{split}.parquet")
        events = pd.read_parquet(processed_path / f"events_{split}.parquet")
        lineups = pd.read_parquet(processed_path / f"lineups_{split}.parquet")

    return {
        'matches': matches,
        'events': events,
        'lineups': lineups
    }