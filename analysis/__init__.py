"""
Tactical analysis modules for tournament compression study.
"""

# Use absolute imports instead of relative
from data_loader import TacticalDataLoader
from profile_builder import build_team_profile
from clustering_analysis import TacticalClustering
from tournament_compression import (
    assign_to_archetypes,
    calculate_cmi,
    analyze_archetype_distribution_shift
)

from rosters_2026 import rosters_2026
from tournament_progression import get_progression_df, wc_2022, euro_2024, copa_2024

__version__ = "0.1.0"

__all__ = [
    'TacticalDataLoader',
    'build_team_profile',
    'TacticalClustering',
    'assign_to_archetypes',
    'calculate_cmi',
    'analyze_archetype_distribution_shift',
    'rosters_2026',
    'get_progression_df',
    'wc_2022',
    'euro_2024',
    'copa_2024'
]