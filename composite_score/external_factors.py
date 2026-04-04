# composite_score/external_factors.py

# ── FIFA Rankings (January 2026, for qualified teams) ─────────────────────
# Verified Official Rankings as of March 29, 2026
FIFA_RANKINGS_SORTED = {
    "Spain": 1,
    "Argentina": 2,
    "France": 3,
    "England": 4,
    "Brazil": 5,
    "Portugal": 6,
    "Netherlands": 7,
    "Morocco": 8,            # Historic High
    "Belgium": 9,
    "Germany": 10,
    "Croatia": 11,
    "Senegal": 12,           # AFCON Winners
    "Italy": 13,
    "Colombia": 14,
    "United States": 15,     # Host #1
    "Mexico": 16,            # Host #2
    "Uruguay": 17,
    "Switzerland": 18,
    "Japan": 19,
    "Iran": 20,
    "Denmark": 21,
    "South Korea": 22,
    "Ecuador": 23,
    "Austria": 24,
    "Turkey": 25,
    "Nigeria": 26,
    "Australia": 27,
    "Algeria": 28,
    "Canada": 29,            # Host #3
    "Ukraine": 30,
    "Egypt": 31,
    "Norway": 32,
    "Panama": 33,
    "Poland": 34,
    "Côte d'Ivoire": 37,
    "Scotland": 38,
    "Serbia": 39,
    "Paraguay": 40,
    "Hungary": 41,
    "Czech Republic": 43,
    "Cameroon": 45,
    "Tunisia": 47,
    "DR Congo": 48,
    "Uzbekistan": 52,        # Qualified for 1st WC
    "Mali": 54,
    "Qatar": 56,
    "South Africa": 60,
    "Saudi Arabia": 61,
    "Jordan": 64,
    "Cape Verde": 67,
    "Ghana": 72,
    "Curaçao": 81,           # Historic Qualifier
    "Haiti": 83,
    "New Zealand": 85
}

def normalize_fifa_rank(rank: int, worst_rank: int = 90) -> float:
    """Rank 1 → 100, Rank 90 → 20. Log scale."""
    import math
    return round(100 - (math.log(rank) / math.log(worst_rank)) * 80, 1)

# ── Coach Tenure (years in charge as of June 2026) ────────────────────────
COACH_TENURE = {
    "Spain": {"coach": "Luis de la Fuente", "years": 3.3},
    "Argentina": {"coach": "Lionel Scaloni", "years": 7.6},
    "France": {"coach": "Didier Deschamps", "years": 13.7},
    "England": {"coach": "Thomas Tuchel", "years": 1.2},
    "Brazil": {"coach": "Carlo Ancelotti", "years": 0.8},
    "Portugal": {"coach": "Roberto Martínez", "years": 3.2},
    "Netherlands": {"coach": "Ronald Koeman", "years": 3.2},
    "Morocco": {"coach": "Mohamed Ouahbi", "years": 0.1},      # Appointed March 2026
    "Belgium": {"coach": "Rudi Garcia", "years": 1.2},
    "Germany": {"coach": "Julian Nagelsmann", "years": 2.5},
    "Croatia": {"coach": "Zlatko Dalić", "years": 8.5},
    "Senegal": {"coach": "Pape Thiaw", "years": 1.3},
    "Italy": {"coach": "Luciano Spalletti", "years": 2.6},
    "Colombia": {"coach": "Néstor Lorenzo", "years": 3.7},
    "United States": {"coach": "Mauricio Pochettino", "years": 1.5},
    "Mexico": {"coach": "Javier Aguirre", "years": 1.7},
    "Uruguay": {"coach": "Marcelo Bielsa", "years": 2.9},
    "Switzerland": {"coach": "Murat Yakin", "years": 4.6},
    "Japan": {"coach": "Hajime Moriyasu", "years": 7.7},
    "Iran": {"coach": "Amir Ghalenoei", "years": 3.0},
    "Denmark": {"coach": "Brian Riemer", "years": 1.4},
    "South Korea": {"coach": "Hong Myung-bo", "years": 1.7},
    "Ecuador": {"coach": "Sebastián Beccacece", "years": 1.6},
    "Austria": {"coach": "Ralf Rangnick", "years": 3.9},
    "Turkey": {"coach": "Vincenzo Montella", "years": 2.5},
    "Nigeria": {"coach": "Eric Chelle", "years": 1.2},
    "Australia": {"coach": "Tony Popovic", "years": 1.5},
    "Algeria": {"coach": "Vladimir Petković", "years": 2.0},
    "Canada": {"coach": "Jesse Marsch", "years": 1.8},
    "Ukraine": {"coach": "Serhiy Rebrov", "years": 2.8},
    "Egypt": {"coach": "Hossam Hassan", "years": 2.1},
    "Norway": {"coach": "Ståle Solbakken", "years": 5.3},
    "Panama": {"coach": "Thomas Christiansen", "years": 5.7},
    "Poland": {"coach": "Michał Probierz", "years": 2.5},
    "Côte d'Ivoire": {"coach": "Emerse Faé", "years": 2.2},
    "Scotland": {"coach": "Steve Clarke", "years": 6.9},
    "Serbia": {"coach": "Dragan Stojković", "years": 5.1},
    "Paraguay": {"coach": "Gustavo Alfaro", "years": 1.6},
    "Hungary": {"coach": "Marco Rossi", "years": 7.8},
    "Czech Republic": {"coach": "Miroslav Koubek", "years": 0.3},
    "Cameroon": {"coach": "David Pagou", "years": 0.3},
    "Tunisia": {"coach": "Sabri Lamouchi", "years": 0.2},
    "DR Congo": {"coach": "Sébastien Desabre", "years": 3.6},
    "Uzbekistan": {"coach": "Fabio Cannavaro", "years": 0.5},
    "Mali": {"coach": "Tom Saintfiet", "years": 1.6},
    "Qatar": {"coach": "Tintín Márquez", "years": 2.3},
    "South Africa": {"coach": "Hugo Broos", "years": 4.9},
    "Saudi Arabia": {"coach": "Hervé Renard", "years": 1.4},
    "Jordan": {"coach": "Jamal Sellami", "years": 1.7},
    "Cape Verde": {"coach": "Bubista", "years": 6.2},
    "Ghana": {"coach": "Otto Addo", "years": 2.0},
    "Curaçao": {"coach": "Fred Rutten", "years": 0.1},
    "Haiti": {"coach": "Sébastien Migné", "years": 1.0},
    "New Zealand": {"coach": "Darren Bazeley", "years": 2.8},
    "Italy": {"coach": "Gennaro Gattuso", "years": 0.79},
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
    "Algeria": 4, "Argentina": 18, "Australia": 6, "Austria": 7, "Belgium": 14,
    "Brazil": 22, "Cameroon": 8, "Canada": 3, "Cape Verde": 1, "Colombia": 6,
    "Croatia": 7, "Curaçao": 1, "Czech Republic": 9, "Côte d'Ivoire": 4,
    "DR Congo": 2, "Denmark": 6, "Ecuador": 5, "Egypt": 3, "England": 16,
    "France": 16, "Germany": 20, "Ghana": 4, "Haiti": 1, "Hungary": 9,
    "Iran": 6, "Italy": 18, "Japan": 7, "Jordan": 1, "Mali": 1, "Mexico": 17,
    "Morocco": 6, "Netherlands": 11, "New Zealand": 2, "Nigeria": 7, "Norway": 3,
    "Panama": 2, "Paraguay": 9, "Poland": 9, "Portugal": 9, "Qatar": 2,
    "Saudi Arabia": 6, "Scotland": 8, "Senegal": 4, "Serbia": 13, "South Africa": 3,
    "South Korea": 11, "Spain": 16, "Switzerland": 12, "Tunisia": 6, "Turkey": 3,
    "Ukraine": 3, "United States": 11, "Uruguay": 14, "Uzbekistan": 1, "Italy": 18,
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