"""
Player position archetype lookup built from StatsBomb events data.
"""

import pandas as pd
from pathlib import Path

POSITION_MAP = {
    "Goalkeeper":                "GK",
    "Left Back":                 "FB",
    "Right Back":                "FB",
    "Left Wing Back":            "FB",
    "Right Wing Back":           "FB",
    "Left Center Back":          "CB",
    "Right Center Back":         "CB",
    "Center Back":               "CB",
    "Left Defensive Midfield":   "DM",
    "Right Defensive Midfield":  "DM",
    "Center Defensive Midfield": "DM",
    "Left Midfield":             "CM",
    "Right Midfield":            "CM",
    "Left Center Midfield":      "CM",
    "Right Center Midfield":     "CM",
    "Center Midfield":           "CM",
    "Left Attacking Midfield":   "AM",
    "Right Attacking Midfield":  "AM",
    "Center Attacking Midfield": "AM",
    "Left Wing":                 "W",
    "Right Wing":                "W",
    "Left Center Forward":       "FW",
    "Right Center Forward":      "FW",
    "Center Forward":            "FW",
    "Secondary Striker":         "FW",
}

# Cache — populated on first call
_PLAYER_POSITION_MAP: dict | None = None
_EVENTS_PATH: str = "../data/Statsbomb/events.parquet"


def set_events_path(path: str) -> None:
    """Override default events path before first use."""
    global _EVENTS_PATH
    _EVENTS_PATH = path


def get_player_position_map(verbose: bool = False) -> dict:
    """
    Return player -> archetype lookup, building it on first call.
    Subsequent calls return cached result instantly.
    """
    global _PLAYER_POSITION_MAP

    if _PLAYER_POSITION_MAP is not None:
        return _PLAYER_POSITION_MAP

    try:
        if verbose:
            print("  Building position map from events...")
        events = pd.read_parquet(_EVENTS_PATH)
        events = events.dropna(subset=["player", "position"])
        events["archetype"] = events["position"].map(POSITION_MAP)
        events = events.dropna(subset=["archetype"])

        lookup = (
            events.groupby("player")["archetype"]
            .agg(lambda x: x.mode()[0])
            .reset_index()
        )

        _PLAYER_POSITION_MAP = dict(zip(lookup["player"], lookup["archetype"]))

        if verbose:
            print(f"  Position map: {len(_PLAYER_POSITION_MAP):,} players")

    except Exception as e:
        print(f"  Warning: could not build position map: {e}")
        _PLAYER_POSITION_MAP = {}

    return _PLAYER_POSITION_MAP