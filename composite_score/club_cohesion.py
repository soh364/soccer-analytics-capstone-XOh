# You can compute this right now
import sys
import pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "player_score"))
from rosters_2026 import rosters_2026

def compute_club_cohesion(country: str) -> dict:
    players = rosters_2026.get(country, {})
    if not players:
        return {"top_club": None, "cohesion_count": 0, "cohesion_score": 0.0}
    
    clubs = [info["club"] for info in players.values()]
    club_counts = pd.Series(clubs).value_counts()
    top_club = club_counts.index[0]
    top_count = club_counts.iloc[0]
    total = len(clubs)
    
    return {
        "top_club": top_club,
        "cohesion_count": top_count,
        "cohesion_pct": top_count / total,
        "cohesion_score": min(top_count / total * 100, 100)
    }

for country in rosters_2026:
    result = compute_club_cohesion(country)
    if result["cohesion_count"] >= 3:
        print(f"{country:20} {result['top_club']:25} {result['cohesion_count']} players ({result['cohesion_pct']:.0%})")