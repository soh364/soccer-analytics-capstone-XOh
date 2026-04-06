# ⚽ 2026 World Cup Readiness Framework

**Team XOH — Soomi Oh, Yoo Mi Oh**  
**GT OMSA Capstone · April 2026**

---

## Overview

A data-driven framework for quantifying national team readiness ahead of the 2026 FIFA World Cup, built on StatsBomb open event data. The project was delivered in two phases: an exploratory analysis at midterm (February 2026) and a full three-layer predictive framework as the final report (April 2026).

The central argument is not that we can predict the winner. It is that readiness is a multi-layered construct that requires distinct measurement tools for each layer — and that each of those tools has a failure mode worth understanding.

**Three-Layer Framework:**

| Layer | What it measures | Data source |
|---|---|---|
| **Tactical Identity** | *How* teams play — collective patterns under pressure | StatsBomb tournament data 2022–2024 |
| **Player Quality** | *Who* plays — individual performance above position baseline | StatsBomb club data 2021/22–2023/24 |
| **Readiness Score** | *How ready* — composite of quality, style, context, uncertainty | All of the above + FIFA rankings, Guardian 100 |

---

## Notebooks

| Notebook | Phase | Purpose |
|---|---|---|
| `EDA.ipynb` | Midterm | Full exploratory analysis — data audit, metric engineering, tactical scatter, player distributions |
| `EDA_Executive.ipynb` | Midterm | Executive summary — key findings and visualisations only |
| `wc2026_analysis.ipynb` | **Final** | Complete framework — tactical clustering, player scoring, composite model, Monte Carlo simulation |

> The final notebook (`wc2026_analysis.ipynb`) picks up at Section III. Sections I–II (data exploration and metric engineering) are covered in the midterm EDA notebooks.

---

## Midterm EDA — What Was Covered (Sections I–II)

The midterm established the data foundations and validated the measurement approach:

- **Data Landscape:** 12.2M events across 3,464 matches; temporal scope analysis; the case for restricting to post-2021 data
- **Metric Engineering:** 8 team tactical dimensions (PPDA, EPR, field tilt, defensive line height, npxG, progression, buildup xG); 12 player quality metrics across 4 categories
- **Tactical Foundations:** Correlation analysis, possession efficiency paradox, CONMEBOL rigidity cluster, Winner Zone identification
- **Player Foundations:** 270-minute threshold rationale, archetype-specific evaluation model, coverage distribution analysis
- **Early Clustering:** k=4 prototype (superseded by k=6 in the final report)

To regenerate the midterm metric outputs:

```bash
python run_metrics.py men_tournament_2022_24 recent_club_players
```

---

## Final Report — What Was Built (Sections III–VII)

| Section | Content |
|---|---|
| **III — Tactical Identity** | KMeans clustering of 71 nations into 6 archetypes, GMM validation (ARI 0.455), outcome validation against WC 2022, four-layer archetype score derivation |
| **IV — Player Quality** | 8-step scoring pipeline, 13 metrics, Guardian blend, country-level aggregation, coverage gap analysis (median confidence 0.18) |
| **V — Composite Readiness** | 8-component model, full 48-nation rankings, volatility index, signal divergence, FIFA vs readiness delta |
| **VI — Tournament Simulation** | 10,000 Monte Carlo simulations, champion probabilities, survival curves, radar/heatmap/cohesion visualisations |
| **VII — Synthesis** | Two-layer finding, measurement limits quantified, framework philosophy, 7 directions for extension |

To reproduce all final outputs:

```bash
python run_pipeline.py
```

---

## Project Structure

```
soccer-analytics-capstone-template/
│
├── wc2026_analysis.ipynb            ← Final report notebook
├── run_pipeline.py                  ← Single entry point for final pipeline
├── run_metrics.py                   ← Midterm metric generation
├── requirements.txt
│
├── player_score/                    ← Player scoring pipeline
│   ├── player_score_pipeline.py     ← 8-step pipeline entry point
│   ├── player_aggregator.py         ← Country-level aggregation + FIFA fallback
│   ├── aggregation.py               ← Match → player × season aggregation
│   ├── loader.py                    ← Data loading from parquet
│   ├── guardians_2025.py            ← Guardian 100 external list (2025 edition)
│   ├── rosters_2026.py              ← 2026 squad rosters (48 nations)
│   ├── player_metrics_config.py     ← 13-metric configuration by position
│   ├── player_position_map.py       ← Events-based position lookup (lazy-loaded)
│   ├── club_mapping_2026.py         ← Club name normalisation
│   ├── steps/
│   │   ├── filter.py                ← 270-min hard floor, 180-min shrinkage zone
│   │   ├── decay.py                 ← Temporal decay (1.0 / 0.90 / 0.80)
│   │   ├── normalization.py         ← Per-season: log / log1p / rank / z-score
│   │   ├── shrinkage.py             ← Bayesian shrinkage toward positional mean
│   │   ├── segmentation.py          ← GK removal, positional archetype labelling
│   │   └── scoring.py               ← Percentile → composite → Guardian blend
│   └── outputs/
│       ├── player_quality_2026.csv  ← Country-level player scores
│       └── player_details_2026.csv  ← Individual player scores
│
├── tactical_clustering/             ← Tactical clustering pipeline
│   ├── tc_pipeline.py               ← Single-function entry point
│   ├── tc_data.py                   ← Load, merge, aggregate tournament metrics
│   ├── tc_preprocessing.py          ← 95th-percentile capping + StandardScaler
│   ├── tc_k_selection.py            ← Silhouette, DB index, ARI, GMM sweep
│   ├── tc_clustering.py             ← KMeans (n_init=20) + GMM validation
│   ├── tc_validation.py             ← Bootstrap, ANOVA, LOO validation suite
│   ├── tc_visualisation.py          ← PCA scatter, archetype radars, outcome charts
│   ├── tc_outcome_validation.py     ← WC 2022 result mapping
│   ├── figures/                     ← Generated PNG visualisations
│   └── outputs/
│       └── team_archetypes.csv      ← Archetype assignments + GMM confidence scores
│
├── composite_score/                 ← Composite scoring + simulation
│   ├── composite_scorer.py          ← Main scoring function (8 components)
│   ├── external_factors.py          ← FIFA rankings, coach tenure, WC appearances
│   ├── club_cohesion.py             ← Squad club concentration (log-scaled)
│   ├── monte_carlo.py               ← 10,000-simulation bracket tournament
│   └── outputs/
│       ├── team_readiness_2026.csv  ← Composite readiness scores (all components)
│       └── monte_carlo_2026.csv     ← Champion/SF/QF/R16/R32 probabilities
│
├── eda/                             ← Midterm EDA (reference)
│   ├── EDA_draft_final_xoh.ipynb
│   ├── EDA_executive_xoh.ipynb
│   ├── analysis/                    ← EDA helper functions and data loaders
│   ├── figures/                     ← EDA visualisations
│   └── processed/                   ← Aggregated EDA outputs
│
└── outputs/
    └── raw_metrics/
        ├── men_tourn_2022_24/       ← Team tactical metrics by season
        └── recent_club_players/     ← Player quality metrics by season
            ├── 2021_2022/
            ├── 2022_2023/
            └── 2023_2024/
```

---

## Setup & Requirements

### Installation

```bash
pip install -r requirements.txt
```

### Core Dependencies

```
polars==1.3.0          # pinned — later versions break list column handling
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
scipy>=1.11
rapidfuzz>=3.0         # fuzzy name matching for player reconciliation
matplotlib>=3.7
seaborn>=0.12
pyarrow>=12.0
fastparquet>=2023.0
```

> **Note:** `polars==1.3.0` is pinned. The `seasons_present` list column in the player pipeline breaks under later versions.

### Data Prerequisites

StatsBomb data must be present at `data/Statsbomb/` with:

```
data/Statsbomb/
├── matches.parquet
├── events.parquet
├── lineups.parquet
└── outputs/raw_metrics/
    ├── men_tourn_2022_24/     ← 8 team metric CSVs
    └── recent_club_players/
        ├── 2021_2022/
        ├── 2022_2023/
        └── 2023_2024/
```

Download via:

```bash
python data/download_data.py
```

---

## Reproducing the Final Report

### Step 1. Generate Metric Outputs

The pipeline modules read from `outputs/raw_metrics/`, which is generated by the midterm metric pipeline. If these files are not present, run this first:
```bash
python run_metrics.py men_tournament_2022_24 recent_club_players
```

This produces the 8 team tactical CSVs in `outputs/raw_metrics/men_tourn_2022_24/` and the player quality metrics in `outputs/raw_metrics/recent_club_players/` across three seasons.

### Step 2. Run the Final Pipeline
```bash
# Full pipeline — all outputs + 10,000 MC simulations (~8 min)
python run_pipeline.py

# Faster testing — 1,000 simulations
python run_pipeline.py --mc-sims 1000

# Skip Monte Carlo entirely
python run_pipeline.py --skip-mc
```

This produces `team_archetypes.csv`, `player_quality_2026.csv`, `team_readiness_2026.csv`, and `monte_carlo_2026.csv` in their respective `outputs/` folders.

### Step 3. Open the Notebook

Open `wc2026_analysis.ipynb` and run all cells in order. The setup cell (`cell 2`) must be run first — it sets `PROJECT_ROOT`, adds all package paths to `sys.path`, and defines shared constants (`FIGURES_DIR`, `PALETTE`). Running cells out of order or after a kernel restart without re-running cell 2 is the most common source of `NameError` and `ModuleNotFoundError`.

---

## Key Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Clustering k | 6 | Highest GMM ARI among k≥4 (0.455); k=2 wins statistically but collapses analytically distinct teams |
| Outlier exclusion | Georgia, Slovenia removed | PPDA >4σ; neither qualifies for WC 2026 |
| Cap threshold | 95th percentile | 99th = single extreme value at n=69; 95th corrects without distorting Japan |
| Minutes threshold | 270 min hard, 180 min floor | Three full-match equivalents; Bayesian shrinkage between floor and threshold |
| Decay weights | 1.0 / 0.90 / 0.80 | Geometric; 2023/24 carries ~2.5× the weight of 2021/22 |
| Position scoring | Intra-archetype percentiles | Prevents generalist bias; DM vs FW not directly comparable |
| External validation | Guardian 100 (2025) | Supplements StatsBomb gaps for high-profile players outside coverage window |
| Coverage fallback | FIFA ranking × 0.75 | Discounted proxy; minimum 40% weight on actual StatsBomb data |
| GMM boundary | gmm_confidence < 0.5 → blended score | 23 of 39 archetype nations are boundary cases; hard assignment would misrepresent uncertainty |
| MC scale parameter | σ = 15 | 10-pt gap ≈ 60% win probability; 25-pt gap ≈ 75% |
| Confederation bonus | ×1.05 (hosts), ×1.01–1.03 (CONMEBOL) | Conservative travel/recovery advantage estimate |

---

## Key Results

**Composite Readiness Top 5:** France (71.18), Argentina (69.81), Spain (69.21), Germany (65.69), Brazil (65.32)

**Monte Carlo champion Probabilities:** France 8.5%, Spain 7.8%, Argentina 7.4%, Germany 7.3%, Brazil 6.6%

**Player Scoring:** 523 players; coverage median confidence 0.18; 11 nations at zero coverage

**Clustering:** k=6, 69 teams (Georgia/Slovenia excluded); High Press / High Output 14 teams, Possession Dominant 11, Mid-Block Reactive 14, Moderate Possession 16, Compact Transition 6, Low Intensity 8

---

## Limitations

- StatsBomb open data does not cover all competitions equally. Approximately one-third of the 48 qualified nations have player coverage confidence <0.3, meaning their readiness scores rely substantially on FIFA ranking as a proxy.
- Tactical clustering uses 2022–2024 tournament data only. 9 of 48 qualified nations have no archetype assignment (Bosnia, Curaçao, Haiti, Iraq, Jordan, New Zealand, Norway, Panama, Uzbekistan).
- The Monte Carlo simulation treats each match as independent. It does not model within-tournament dynamics — injuries, tactical adjustments, momentum, or referee decisions.
- The Guardian 100 list introduces a human-curated external signal. Its subjectivity is disclosed but not eliminated.
- `polars==1.3.0` is required. The player pipeline uses list column operations that break under later versions.

---

## Data Licensing

### StatsBomb Data
- **License:** [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)
- **Usage:** Non-commercial use only, attribution required
- **Citation:** "StatsBomb Open Data"

### External Sources
- **FIFA World Rankings** — April 2026 official rankings (public)
- **The Guardian's 100 Best Footballers 2025** — used as external quality benchmark
- **2026 World Cup Rosters** — manually curated from official confederation announcements

> The code in this repository is MIT-licensed. The data sources are not covered by the MIT license and have their own licensing terms. Users must comply with each provider's terms when using data for their own projects.

---

*For the full analytical narrative, methodology, and visualisations — open `wc2026_analysis.ipynb`.*