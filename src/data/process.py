"""
Main data processing pipeline.
Orchestrates loading, cleaning, normalizing, and splitting data.
"""
import pandas as pd
import json
from pathlib import Path

from .load import load_raw_data
from .normalize import normalize_ids
from .split import create_splits
from ..utils.config import (
    DATA_DIR, 
    PROCESSED_DIR, 
    TRAIN_RATIO, 
    VAL_RATIO, 
    TEST_RATIO,
    LOW_EVENT_PERCENTILE
)


def filter_low_quality_matches(matches, events, lineups, percentile=LOW_EVENT_PERCENTILE):
    # remove matches with unusually low event counts
    events_per_match = events.groupby('match_id').size()
    threshold = events_per_match.quantile(percentile)
    low_match_ids = events_per_match[events_per_match < threshold].index

    matches_clean = matches[~matches['match_id'].isin(low_match_ids)].copy()
    events_clean = events[~events['match_id'].isin(low_match_ids)].copy()
    lineups_clean = lineups[~lineups['match_id'].isin(low_match_ids)].copy()

    print(f"Filtered low-quality matches: removed {len(low_match_ids)}")
    return {'matches': matches_clean, 'events': events_clean, 'lineups': lineups_clean}


def validate_splits(splits):
    train_ids = set(splits['metadata']['train_match_ids'])
    val_ids = set(splits['metadata']['val_match_ids'])
    test_ids = set(splits['metadata']['test_match_ids'])

    ov_train_val = len(train_ids & val_ids)
    ov_train_test = len(train_ids & test_ids)
    ov_val_test = len(val_ids & test_ids)

    print(f"Split overlaps - train/val={ov_train_val}, train/test={ov_train_test}, val/test={ov_val_test}")
    return (ov_train_val == 0 and ov_train_test == 0 and ov_val_test == 0)


def save_processed_data(normalized_data, splits, output_dir):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    normalized_data['matches'].to_parquet(output_path / "matches_normalized.parquet", index=False)
    normalized_data['events'].to_parquet(output_path / "events_normalized.parquet", index=False)
    normalized_data['lineups'].to_parquet(output_path / "lineups_normalized.parquet", index=False)

    for split_name in ['train', 'val', 'test']:
        splits[split_name]['matches'].to_parquet(output_path / f"matches_{split_name}.parquet", index=False)
        splits[split_name]['events'].to_parquet(output_path / f"events_{split_name}.parquet", index=False)
        splits[split_name]['lineups'].to_parquet(output_path / f"lineups_{split_name}.parquet", index=False)

    with open(output_path / "split_info.json", 'w') as f:
        json.dump(splits['metadata'], f)

    print("Saved processed data")


def run_pipeline(data_dir=DATA_DIR, output_dir=PROCESSED_DIR):
    """Run data pipeline and return brief results."""
    print("Starting pipeline...")

    raw_data = load_raw_data(data_dir)

    clean_data = filter_low_quality_matches(raw_data['matches'], raw_data['events'], raw_data['lineups'])

    normalized_data = normalize_ids(clean_data['matches'], clean_data['events'], clean_data['lineups'], raw_data['reference'])

    splits = create_splits(normalized_data['matches'], normalized_data['events'], normalized_data['lineups'], TRAIN_RATIO, VAL_RATIO, TEST_RATIO)

    validation_passed = validate_splits(splits)

    save_processed_data(normalized_data, splits, output_dir)

    print(f"Done: validation={'PASSED' if validation_passed else 'FAILED'} | matches={len(normalized_data['matches']):,} events={len(normalized_data['events']):,} lineups={len(normalized_data['lineups']):,}")

    return {
        'validation_passed': validation_passed,
        'n_matches': len(normalized_data['matches']),
        'n_events': len(normalized_data['events']),
        'n_lineups': len(normalized_data['lineups']),
        'splits': splits['metadata']
    }