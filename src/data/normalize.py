"""
Normalize IDs by adding readable name columns.
"""
import pandas as pd


def create_lookup_dict(reference, table_name):
    table_data = reference[reference['table_name'] == table_name]
    return table_data.set_index('id')['name'].to_dict()


def normalize_matches(matches, reference):
    matches_norm = matches.copy()
    team_lookup = create_lookup_dict(reference, 'team')
    if 'home_team_id' in matches_norm.columns:
        matches_norm['home_team_name'] = matches_norm['home_team_id'].map(team_lookup)
    if 'away_team_id' in matches_norm.columns:
        matches_norm['away_team_name'] = matches_norm['away_team_id'].map(team_lookup)
    return matches_norm


def normalize_events(events, reference):
    events_norm = events.copy()
    team_lookup = create_lookup_dict(reference, 'team')
    player_lookup = create_lookup_dict(reference, 'player')
    position_lookup = create_lookup_dict(reference, 'position')
    event_type_lookup = create_lookup_dict(reference, 'event_type')
    play_pattern_lookup = create_lookup_dict(reference, 'play_pattern')

    if 'team_id' in events_norm.columns:
        events_norm['team_name'] = events_norm['team_id'].map(team_lookup)
    if 'player_id' in events_norm.columns:
        events_norm['player_name'] = events_norm['player_id'].map(player_lookup)
    if 'position_id' in events_norm.columns:
        events_norm['position_name'] = events_norm['position_id'].map(position_lookup)
    if 'type_id' in events_norm.columns:
        events_norm['type_name'] = events_norm['type_id'].map(event_type_lookup)
    if 'play_pattern_id' in events_norm.columns:
        events_norm['play_pattern_name'] = events_norm['play_pattern_id'].map(play_pattern_lookup)
    return events_norm


def normalize_lineups(lineups, reference):
    lineups_norm = lineups.copy()
    team_lookup = create_lookup_dict(reference, 'team')
    player_lookup = create_lookup_dict(reference, 'player')
    position_lookup = create_lookup_dict(reference, 'position')

    if 'team_id' in lineups_norm.columns:
        lineups_norm['team_name'] = lineups_norm['team_id'].map(team_lookup)
    if 'player_id' in lineups_norm.columns:
        lineups_norm['player_name'] = lineups_norm['player_id'].map(player_lookup)
    if 'position_id' in lineups_norm.columns:
        lineups_norm['position_name'] = lineups_norm['position_id'].map(position_lookup)
    return lineups_norm


def normalize_ids(matches, events, lineups, reference):
    """Normalize datasets by adding readable name columns and return normalized dict."""
    print("Normalizing IDs...")

    matches_norm = normalize_matches(matches, reference)
    events_norm = normalize_events(events, reference)
    lineups_norm = normalize_lineups(lineups, reference)

    print(f"Done: matches={len(matches_norm):,}, events={len(events_norm):,}, lineups={len(lineups_norm):,}")

    return {
        'matches': matches_norm,
        'events': events_norm,
        'lineups': lineups_norm
    }