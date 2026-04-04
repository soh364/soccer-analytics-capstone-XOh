# composite_score/external_factors.py

# ── FIFA Rankings (January 2026, for qualified teams) ─────────────────────
# Verified Official Rankings as of March 29, 2026
FIFA_RANKINGS_SORTED = {
    "France": 1,             # Reclaimed Top Spot
    "Spain": 2,              # Down 1
    "Argentina": 3,          # Down 1
    "England": 4,
    "Portugal": 5,           # Swapped with Brazil
    "Brazil": 6,
    "Netherlands": 7,
    "Morocco": 8,            # Historic High maintained
    "Belgium": 9,
    "Germany": 10,
    "Croatia": 11,
    "Italy": 12,             # Highest-ranked team NOT in the WC
    "Colombia": 13,
    "Senegal": 14,
    "Mexico": 15,            # Host #1
    "United States": 16,     # Host #2
    "Uruguay": 17,
    "Japan": 18,
    "Switzerland": 19,
    "Denmark": 20,           # Did not qualify
    "Iran": 21,
    "Turkey": 22,            # Playoff Winner boost
    "Ecuador": 23,
    "Austria": 24,
    "South Korea": 25,
    "Nigeria": 26,           # Did not qualify
    "Australia": 27,
    "Algeria": 28,
    "Egypt": 29,             # Rose after draw vs Spain
    "Canada": 30,            # Host #3
    "Norway": 31,
    "Ukraine": 32,
    "Panama": 33,
    "Côte d'Ivoire": 34,
    "Poland": 35,
    "Sweden": 38,            # Playoff Winner boost
    "Serbia": 39,
    "Paraguay": 40,
    "Czech Republic": 41,    # Playoff Winner boost
    "Hungary": 42,
    "Scotland": 43,
    "Tunisia": 44,
    "Cameroon": 45,
    "DR Congo": 46,
    "Uzbekistan": 50,        # Historic High (Debutants)
    "Mali": 52,
    "Qatar": 55,
    "Iraq": 57,              # Playoff Winner boost
    "South Africa": 60,
    "Saudi Arabia": 61,
    "Jordan": 63,
    "Bosnia and Herzegovina": 65, # Huge rise after qualifying
    "Cape Verde": 69,
    "Ghana": 74,
    "Curaçao": 82,           # Historic High (Debutants)
    "Haiti": 83,
    "New Zealand": 85
}

def normalize_fifa_rank(rank: int, worst_rank: int = 90) -> float:
    """Rank 1 → 100, Rank 90 → 20. Log scale."""
    import math
    return round(100 - (math.log(rank) / math.log(worst_rank)) * 80, 1)

# ── Coach Tenure (years in charge as of June 2026) ────────────────────────
COACH_TENURE = {
    # --- Group A ---
    "Mexico": {"coach": "Javier Aguirre", "years": 1.7},
    "South Africa": {"coach": "Hugo Broos", "years": 4.9},
    "South Korea": {"coach": "Hong Myung-bo", "years": 1.7},
    "Czech Republic": {"coach": "Miroslav Koubek", "years": 0.3},

    # --- Group B ---
    "Canada": {"coach": "Jesse Marsch", "years": 1.8},
    "Bosnia and Herzegovina": {"coach": "Sergej Barbarez", "years": 2.0},
    "Qatar": {"coach": "Tintín Márquez", "years": 2.3},
    "Switzerland": {"coach": "Murat Yakin", "years": 4.6},

    # --- Group C ---
    "Brazil": {"coach": "Carlo Ancelotti", "years": 0.8},
    "Morocco": {"coach": "Mohamed Ouahbi", "years": 0.1},
    "Haiti": {"coach": "Sébastien Migné", "years": 1.0},
    "Scotland": {"coach": "Steve Clarke", "years": 6.9},

    # --- Group D ---
    "United States": {"coach": "Mauricio Pochettino", "years": 1.5},
    "Paraguay": {"coach": "Gustavo Alfaro", "years": 1.6},
    "Australia": {"coach": "Tony Popovic", "years": 1.5},
    "Turkey": {"coach": "Vincenzo Montella", "years": 2.5},

    # --- Group E ---
    "Germany": {"coach": "Julian Nagelsmann", "years": 2.5},
    "Curaçao": {"coach": "Fred Rutten", "years": 0.1},
    "Côte d'Ivoire": {"coach": "Emerse Faé", "years": 2.2},
    "Ecuador": {"coach": "Sebastián Beccacece", "years": 1.6},

    # --- Group F ---
    "Netherlands": {"coach": "Ronald Koeman", "years": 3.2},
    "Japan": {"coach": "Hajime Moriyasu", "years": 7.7},
    "Sweden": {"coach": "Jon Dahl Tomasson", "years": 2.1},
    "Tunisia": {"coach": "Sabri Lamouchi", "years": 0.2},

    # --- Group G ---
    "Belgium": {"coach": "Rudi Garcia", "years": 1.2},
    "Egypt": {"coach": "Hossam Hassan", "years": 2.1},
    "Iran": {"coach": "Amir Ghalenoei", "years": 3.0},
    "New Zealand": {"coach": "Darren Bazeley", "years": 2.8},

    # --- Group H ---
    "Spain": {"coach": "Luis de la Fuente", "years": 3.3},
    "Cape Verde": {"coach": "Bubista", "years": 6.2},
    "Saudi Arabia": {"coach": "Hervé Renard", "years": 1.4},
    "Uruguay": {"coach": "Marcelo Bielsa", "years": 2.9},

    # --- Group I ---
    "France": {"coach": "Didier Deschamps", "years": 13.7},
    "Senegal": {"coach": "Pape Thiaw", "years": 1.3},
    "Iraq": {"coach": "Jesús Casas", "years": 3.4},
    "Norway": {"coach": "Ståle Solbakken", "years": 5.3},

    # --- Group J ---
    "Argentina": {"coach": "Lionel Scaloni", "years": 7.6},
    "Algeria": {"coach": "Vladimir Petković", "years": 2.0},
    "Austria": {"coach": "Ralf Rangnick", "years": 3.9},
    "Jordan": {"coach": "Jamal Sellami", "years": 1.7},

    # --- Group K ---
    "Portugal": {"coach": "Roberto Martínez", "years": 3.2},
    "DR Congo": {"coach": "Sébastien Desabre", "years": 3.6},
    "Uzbekistan": {"coach": "Fabio Cannavaro", "years": 0.5},
    "Colombia": {"coach": "Néstor Lorenzo", "years": 3.7},

    # --- Group L ---
    "England": {"coach": "Thomas Tuchel", "years": 1.2},
    "Croatia": {"coach": "Zlatko Dalić", "years": 8.5},
    "Ghana": {"coach": "Otto Addo", "years": 2.0},
    "Panama": {"coach": "Thomas Christiansen", "years": 5.7},
}

def normalize_tenure(years: float) -> float:
    if years < 1:
        return 50.0  # up from 30 — new coach of a top team still has quality
    elif years <= 3:
        return 50 + (years - 1) * 15  # 50 → 80
    elif years <= 7:
        return 80 + (years - 3) * 2.5  # 80 → 90
    elif years <= 10:
        return 90 - (years - 7) * 5   # 90 → 75
    else:
        return max(75 - (years - 10) * 3, 50)

# ── Tournament Experience (World Cup appearances) ─────────────────────────
WC_APPEARANCES = {
    # --- HOSTS (3) ---
    "Canada": 3,
    "Mexico": 18,
    "United States": 12,

    # --- AFC: Asia (9) ---
    "Australia": 7,
    "Iran": 7,
    "Iraq": 2,
    "Japan": 8,
    "Jordan": 1,         # Debut
    "Qatar": 2,
    "Saudi Arabia": 7,
    "South Korea": 12,
    "Uzbekistan": 1,      # Debut

    # --- CAF: Africa (10) ---
    "Algeria": 5,
    "Cape Verde": 1,      # Debut
    "Côte d'Ivoire": 4,
    "DR Congo": 2,
    "Egypt": 4,
    "Ghana": 5,
    "Morocco": 7,
    "Senegal": 4,
    "South Africa": 4,
    "Tunisia": 7,

    # --- CONCACAF: N. America & Caribbean (3 additional) ---
    "Curaçao": 1,         # Debut
    "Haiti": 2,
    "Panama": 2,

    # --- CONMEBOL: S. America (6) ---
    "Argentina": 19,
    "Brazil": 23,
    "Colombia": 7,
    "Ecuador": 5,
    "Paraguay": 9,
    "Uruguay": 15,

    # --- OFC: Oceania (1) ---
    "New Zealand": 3,

    # --- UEFA: Europe (16) ---
    "Austria": 8,
    "Belgium": 15,
    "Bosnia and Herzegovina": 2,
    "Croatia": 8,
    "Czech Republic": 10,
    "England": 17,
    "France": 17,
    "Germany": 21,
    "Netherlands": 12,
    "Norway": 4,
    "Portugal": 10,
    "Scotland": 9,
    "Spain": 17,
    "Sweden": 13,
    "Switzerland": 13,
    "Turkey": 3
}

def normalize_wc_experience(appearances: int, max_appearances: int = 22) -> float:
    """Log scale — first appearance = 40, Brazil (22) = 100."""
    import math
    if appearances == 0:
        return 20.0
    return round(40 + (math.log(appearances + 1) / math.log(max_appearances + 1)) * 60, 1)


def get_all_external_factors(country: str) -> dict:
    """Get all external factors for a country as a single dict."""
    fifa_rank = FIFA_RANKINGS.get(country, 80)
    tenure_years = COACH_TENURE.get(country, {}).get("years", 1.0)
    wc_apps = WC_APPEARANCES.get(country, 0)

    return {
        "country": country,
        "fifa_rank": fifa_rank,
        "fifa_score": normalize_fifa_rank(fifa_rank),
        "coach": COACH_TENURE.get(country, {}).get("coach", "Unknown"),
        "coach_tenure_years": tenure_years,
        "coach_tenure_score": round(normalize_tenure(tenure_years), 1),
        "wc_appearances": wc_apps,
        "wc_experience_score": normalize_wc_experience(wc_apps),
    }

def compute_squad_age_score(country: str) -> float:
    """
    Peak age 26-29 = 100.
    Penalty for very young or very old squads.
    """
    from rosters_2026 import rosters_2026
    
    players = rosters_2026.get(country, {})
    if not players:
        return 70.0
    
    ages = [info.get('age', 27) for info in players.values() 
            if info.get('age') is not None]
    if not ages:
        return 70.0
    
    avg_age = sum(ages) / len(ages)
    
    if 26 <= avg_age <= 29:
        return 100.0
    elif avg_age < 26:
        return round(100 - (26 - avg_age) * 8, 1)
    else:
        return round(max(100 - (avg_age - 29) * 8, 30), 1)


if __name__ == "__main__":
    import pandas as pd
    rows = [get_all_external_factors(c) for c in FIFA_RANKINGS]
    df = pd.DataFrame(rows).sort_values("fifa_score", ascending=False)
    print(df.to_string(index=False))