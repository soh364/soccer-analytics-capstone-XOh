"""
Player quality metrics configuration.
Uses 8 files with 13 dimensions across 4 trait categories.
"""

PLAYER_METRICS = {
    # ========================================================================
    # FINISHING & EFFICIENCY 
    # ========================================================================
    'finishing_quality': {
        'file': 'xg__player__totals.csv',
        'column': 'goals_minus_xg',
        'trait_category': 'Final_Third_Output',
        'description': 'Goals vs xG (finishing quality)',
        'min_minutes': 450,  # ~5 matches minimum
        'higher_is_better': True
    },
    'xg_volume': {
        'file': 'xg__player__totals.csv',
        'column': 'xg',
        'trait_category': 'Final_Third_Output',
        'description': 'Total xG (shooting volume)',
        'min_minutes': 450,
        'higher_is_better': True
    },
    
    # ========================================================================
    # PROGRESSION 
    # ========================================================================
    'progressive_passes': {
        'file': 'progression__player__profile.csv',
        'column': 'progressive_passes_p90',
        'trait_category': 'Progression',
        'description': 'Forward passing per 90',
        'min_minutes': 450,
        'higher_is_better': True
    },
    'progressive_carries': {
        'file': 'progression__player__profile.csv',
        'column': 'progressive_carries_p90',
        'trait_category': 'Progression',
        'description': 'Ball-carrying per 90',
        'min_minutes': 450,
        'higher_is_better': True
    },
    
    # ========================================================================
    # ATTACKING CONTRIBUTION 
    # ========================================================================
    'xg_chain': {
        'file': 'advanced__player__xg_chain.csv',
        'column': 'xg_chain_per90',
        'trait_category': 'Final_Third_Output',
        'description': 'xG chain per 90 (possession involvement)',
        'min_minutes': 450,
        'higher_is_better': True
    },
    'team_involvement': {
        'file': 'advanced__player__xg_chain.csv',
        'column': 'team_involvement_pct',
        'trait_category': 'Control',
        'description': '% of team xG chains involved in',
        'min_minutes': 450,
        'higher_is_better': True
    },
    
    # ========================================================================
    # BUILD-UP QUALITY 
    # ========================================================================
    'xg_buildup': {
        'file': 'advanced__player__xg_buildup.csv',
        'column': 'xg_buildup_per90',
        'trait_category': 'Final_Third_Output',
        'description': 'Build-up xG per 90',
        'min_minutes': 450,
        'higher_is_better': True
    },
    'network_centrality': {
        'file': 'advanced__player__network_centrality.csv',
        'column': 'network_involvement_pct',
        'trait_category': 'Control',
        'description': 'Network centrality (passing hub)',
        'min_minutes': 450,
        'higher_is_better': True
    },
    
    # ========================================================================
    # DEFENSIVE CONTRIBUTION 
    # ========================================================================
    'defensive_actions': {
        'file': 'defensive__player__profile.csv',
        'column': 'total_defensive_actions',
        'trait_category': 'Mobility_Intensity',
        'description': 'Total defensive actions',
        'min_minutes': 450,
        'higher_is_better': True
    },
    'high_turnovers': {
        'file': 'defensive__player__profile.csv',
        'column': 'high_turnovers',
        'trait_category': 'Mobility_Intensity',
        'description': 'High turnovers (pressing success)',
        'min_minutes': 450,
        'higher_is_better': True
    },
    
    # ========================================================================
    # PRESSING QUALITY 
    # ========================================================================
    'pressure_volume': {
        'file': 'defensive__player__pressures.csv',
        'column': 'pressures_per_90',
        'trait_category': 'Mobility_Intensity',
        'description': 'Pressing volume per 90',
        'min_minutes': 450,
        'higher_is_better': True
    },
    'pressure_success': {
        'file': 'defensive__player__pressures.csv',
        'column': 'pressure_success_pct',
        'trait_category': 'Mobility_Intensity',
        'description': 'Pressure success rate',
        'min_minutes': 450,
        'higher_is_better': True
    },
    
}

# Trait category mapping
TRAIT_CATEGORIES = {
    'Mobility_Intensity': ['defensive_actions', 'high_turnovers', 'pressure_volume', 'pressure_success'],
    'Progression': ['progressive_passes', 'progressive_carries', 'packing'],
    'Control': ['team_involvement', 'network_centrality'],
    'Final_Third_Output': ['finishing_quality', 'xg_volume', 'xg_chain', 'xg_buildup']
}