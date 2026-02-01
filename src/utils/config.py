"""
Configuration file for data processing pipeline.
Contains paths, constants, and settings.
"""
from pathlib import Path

# Project structure
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "Statsbomb"
PROCESSED_DIR = DATA_DIR / "processed"

# Ensure processed directory exists
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Data quality thresholds
LOW_EVENT_PERCENTILE = 0.05  # Filter bottom 5% of matches by event count

# Reference table names
REFERENCE_TABLES = {
    'team': 'team',
    'player': 'player',
    'position': 'position',
    'event_type': 'event_type',
    'play_pattern': 'play_pattern',
    'country': 'country'
}