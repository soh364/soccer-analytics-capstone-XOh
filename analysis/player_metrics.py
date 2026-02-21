"""
Define the 10 player metrics for quality scoring.
"""

# The 10 dimensions for player quality
PLAYER_METRICS = {
    # Creativity (2 metrics)
    'xg_assisted': {
        'file': 'xg_player_totals.csv',
        'column': 'xa',  # xG assisted
        'per_90': True,
        'positions': ['MF', 'FW'],
        'description': 'Expected assists per 90'
    },
    'key_passes': {
        'file': 'progression_player_profiles.csv',
        'column': 'key_passes',
        'per_90': True,
        'positions': ['MF', 'FW'],
        'description': 'Passes leading to shots'
    },
    
    # Progression (3 metrics)
    'progressive_carries': {
        'file': 'progression_carries.csv',
        'column': 'progressive_carries',
        'per_90': True,
        'positions': ['DF', 'MF', 'FW'],
        'description': 'Ball-carrying advancement'
    },
    'progressive_passes': {
        'file': 'progression_passes.csv',
        'column': 'progressive_passes',
        'per_90': True,
        'positions': ['DF', 'MF'],
        'description': 'Forward passing'
    },
    'packing': {
        'file': 'advanced_packing_stats.csv',
        'column': 'packing_value',
        'per_90': True,
        'positions': ['MF', 'FW'],
        'description': 'Line-breaking passes'
    },
    
    # Efficiency (2 metrics)
    'finishing': {
        'file': 'xg_player_totals.csv',
        'column': 'npxg',  # Non-penalty xG
        'per_90': True,
        'positions': ['FW'],
        'description': 'Shooting efficiency'
    },
    'xg_chain': {
        'file': 'advanced_xg_chain_raw.csv',
        'column': 'xgchain_per90',
        'per_90': False,  # Already per 90
        'positions': ['MF', 'FW'],
        'description': 'Contribution to possessions'
    },
    
    # Defensive (3 metrics)
    'tackles_interceptions': {
        'file': 'defensive_actions_by_zone.csv',
        'column': 'total_actions',
        'per_90': True,
        'positions': ['DF', 'MF'],
        'description': 'Defensive actions'
    },
    'pressure_regains': {
        'file': 'defensive_pressures_player.csv',
        'column': 'pressure_regains',
        'per_90': True,
        'positions': ['DF', 'MF', 'FW'],
        'description': 'Successful pressing'
    },
    'defensive_positioning': {
        'file': 'defensive_line_height_team.csv',  # May need player-level file
        'column': 'defensive_line_height',
        'per_90': False,
        'positions': ['DF'],
        'description': 'Average defensive height'
    }
}

