"""
Tournament progression scores for 2022-24 men's tournaments.
Score = furthest round reached in ANY tournament (best performance).

Scoring system:
    5 = Winner
    4 = Runner-up
    3 = Semi-final
    2 = Quarter-final
    1 = Round of 16 / Group stage (better performers)
    0 = Group stage exit
"""

import pandas as pd

tournament_progression = {}

# 2022 FIFA World Cup
wc_2022 = {
    'Argentina': 5,      # Winner
    'France': 4,         # Runner-up
    'Croatia': 3,        # 3rd place
    'Morocco': 3,        # 4th place
    'Netherlands': 2,    # Quarter-final
    'Brazil': 2,
    'England': 2,
    'Portugal': 2,
    'Spain': 1,          # Round of 16
    'Japan': 1, 'Switzerland': 1, 'South Korea': 1,
    'Senegal': 1, 'Australia': 1, 'Poland': 1, 'United States': 1,
    'Germany': 0, 'Belgium': 0, 'Uruguay': 0, 'Denmark': 0,
    'Mexico': 0, 'Wales': 0, 'Ecuador': 0, 'Ghana': 0,
    'Cameroon': 0, 'Serbia': 0, 'Tunisia': 0, 'Costa Rica': 0,
    'Canada': 0, 'Iran': 0, 'Saudi Arabia': 0, 'Qatar': 0,
}

# 2024 UEFA Euro
euro_2024 = {
    'Spain': 5,          # Winner
    'England': 4,        # Runner-up  
    'Netherlands': 3,    # Semi-final
    'France': 3,         # Semi-final
    'Portugal': 2,       # Quarter-final
    'Germany': 2, 'Switzerland': 2, 'Turkey': 2,
    'Austria': 1,        # Round of 16
    'Belgium': 1, 'Italy': 1, 'Denmark': 1,
    'Romania': 1, 'Slovakia': 1, 'Slovenia': 1, 'Georgia': 1,
    'Croatia': 0, 'Czech Republic': 0, 'Hungary': 0, 'Poland': 0,
    'Scotland': 0, 'Serbia': 0, 'Albania': 0, 'Ukraine': 0,
}

# 2024 Copa America
copa_2024 = {
    'Argentina': 5,      # Winner
    'Colombia': 4,       # Runner-up
    'Uruguay': 3,        # Semi-final (3rd place)
    'Canada': 3,         # Semi-final (4th place)
    'Venezuela': 2,      # Quarter-final
    'Ecuador': 2, 'Panama': 2, 'Brazil': 2,
    'Mexico': 1,         # Group stage (4 pts - better performer)
    'United States': 1,  # Group stage (3 pts)
    'Chile': 1,          # Group stage (2 pts)
    'Paraguay': 0,       # Group stage (0 pts)
    'Peru': 0, 'Costa Rica': 0, 'Jamaica': 0, 'Bolivia': 0,
}

def get_progression_scores():

    # Merge all tournaments taking best performance per team.

    progression = {}

    for tournament in [wc_2022, euro_2024, copa_2024]:
        for team, score in tournament.items():
            if team in progression:
                progression[team] = max(progression[team], score)
            else:
                progression[team] = score
    
    return progression


def get_progression_df():

    # Return progression scores as a pandas DataFrame.
    
    import pandas as pd
    
    progression = get_progression_scores()
    
    return pd.DataFrame([
        {'team': team, 'progression_score': score}
        for team, score in progression.items()
    ])

