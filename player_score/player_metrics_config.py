"""
Player quality metrics configuration.
Uses 8 files with 13 dimensions across 4 trait categories.
"""

PLAYER_METRICS = {

    "finishing_quality": {
        "file": "xg__player__totals.csv",
        "column": "goals_per_xg",  # ← was goals_minus_xg
        "trait_category": "Final_Third_Output",
        "description": "Goals per xG (finishing efficiency)",
        "min_minutes": 270,
        "higher_is_better": True,
        "filter_column": "matches",
        "filter_threshold": 3
    },
    'xg_volume': {
        'file': 'xg__player__totals.csv',
        'column': 'xg',
        'trait_category': 'Final_Third_Output',
        'description': 'Total xG (shooting volume)',
        'min_minutes': 270,
        'higher_is_better': True,
        'filter_column': 'matches',
        'filter_threshold': 3
    },
    
    'progressive_passes': {
        'file': 'progression__player__profile.csv',
        'column': 'progressive_passes_p90',
        'trait_category': 'Progression',
        'description': 'Forward passing per 90',
        'min_minutes': 270,
        'higher_is_better': True
    },
    'progressive_carries': {
        'file': 'progression__player__profile.csv',
        'column': 'progressive_carries_p90',
        'trait_category': 'Progression',
        'description': 'Ball-carrying per 90',
        'min_minutes': 270,
        'higher_is_better': True
    },
    'packing': {
        'file': 'advanced__player__packing.csv',
        'column': 'avg_packing_per_pass',
        'trait_category': 'Progression',
        'description': 'Avg packing per pass',
        'min_minutes': 270,
        'higher_is_better': True,
        'filter_column': 'total_passes',
        'filter_threshold': 30,
    },
    
    'xg_chain': {
        'file': 'advanced__player__xg_chain.csv',
        'column': 'xg_chain_per90',
        'trait_category': 'Final_Third_Output',
        'description': 'xG chain per 90 (possession involvement)',
        'min_minutes': 270,
        'higher_is_better': True
    },
    'team_involvement': {
        'file': 'advanced__player__xg_chain.csv',
        'column': 'team_involvement_pct',
        'trait_category': 'Final_Third_Output',
        'description': '% of team xG chains involved in',
        'min_minutes': 270,
        'higher_is_better': True
    },
    
    'xg_buildup': {
        'file': 'advanced__player__xg_buildup.csv',
        'column': 'xg_buildup_per90',
        'trait_category': 'Final_Third_Output',
        'description': 'Build-up xG per 90',
        'min_minutes': 270,
        'higher_is_better': True
    },
    'network_centrality': {
        'file': 'advanced__player__network_centrality.csv',
        'column': 'network_involvement_pct',
        'trait_category': 'Control',
        'description': 'Network centrality (passing hub)',
        'min_minutes': 270,
        'higher_is_better': True
    },
    
    'defensive_actions': {
        'file': 'defensive__player__profile.csv',
        'column': 'total_defensive_actions',
        'trait_category': 'Mobility_Intensity',
        'description': 'Total defensive actions',
        'min_minutes': 270,
        'higher_is_better': True
    },
    'high_turnovers': {
        'file': 'defensive__player__profile.csv',
        'column': 'high_turnovers',
        'trait_category': 'Mobility_Intensity',
        'description': 'High turnovers (pressing success)',
        'min_minutes': 270,
        'higher_is_better': True
    },

    'pressure_volume': {
        'file': 'defensive__player__pressures.csv',
        'column': 'pressures_per_90',
        'trait_category': 'Mobility_Intensity',
        'description': 'Pressing volume per 90',
        'min_minutes': 270,
        'higher_is_better': True
    },

    'pressure_success': {
        'file': 'defensive__player__pressures.csv',
        'column': 'pressure_success_pct',
        'trait_category': 'Mobility_Intensity',
        'description': 'Pressure success rate',
        'min_minutes': 270,
        'higher_is_better': True
    },
    
}

# Trait category mapping
TRAIT_CATEGORIES = {
    "Mobility_Intensity": [
        "defensive_actions",
        "high_turnovers",
        "pressure_volume",
        "pressure_success",
    ],
    "Progression": [
        "progressive_passes",
        "progressive_carries",
        "packing",
        "xg_chain",        # how often involved in shot-leading sequences
        "team_involvement", # % of team chains involved in
    ],
    "Control": [
        "network_centrality",  # passing hub / ball retention
    ],
    "Final_Third_Output": [
        "finishing_quality",  # goals/xg ratio
        "xg_volume",          # total xG generated
        "xg_buildup",         # buildup contribution per 90
    ],
}