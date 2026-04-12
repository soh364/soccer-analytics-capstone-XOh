# ⚽ 2026 World Cup Readiness Framework

**Team XOH — Soomi Oh, Yoo Mi Oh**  
**GT OMSA Capstone · April 2026**

---

## Overview

This is a data-driven framework for quantifying national team readiness ahead of the 2026 FIFA World Cup, built on StatsBomb open event data. The project was delivered in two phases: an exploratory analysis at midterm (February 2026) and a full three-layer predictive framework as the final report (April 2026).

The central argument is not that we can predict the winner. It is that readiness is a multi-layered construct that requires distinct measurement tools for each layer — and that each of those tools has a failure mode worth understanding.

**Three-Layer Framework:**

| Layer | What it measures | Data source |
|---|---|---|
| **Tactical Identity** | *How* teams play: collective patterns under pressure | StatsBomb tournament data 2022–2024 |
| **Player Quality** | *Who* plays: individual performance above position baseline | StatsBomb club data 2021/22–2023/24 |
| **Readiness Score** | *How ready*: composite of quality, style, context, uncertainty | All of the above + FIFA rankings, Guardian 100 |

---

## Dashboard

An interactive dashboard accompanies the final report, built with Plotly Dash. It visualises all framework outputs — tactical archetypes, player quality scores, composite readiness rankings, and Monte Carlo simulation results — in a navigable interface designed for both technical and non-technical audiences.

### Running the Dashboard

**Prerequisites:** The final pipeline must have been run at least once so the output CSVs exist (see [Reproducing the Final Report](#reproducing-the-final-report)).
```bash
# Install dashboard dependencies (if not already installed)
pip install dash plotly pandas numpy

# Run the dashboard
python wc2026_dashboard.py
```

Then open your browser to: **http://127.0.0.1:8050**

The dashboard loads all data at startup from the pipeline output CSVs. If you rerun the pipeline with updated data, restart the dashboard server to reflect the changes.

### Dashboard Structure

| Tab | What it shows |
|---|---|
| **⚔️ War Room** | Global overview — tactical vs player quality scatter, top contenders, signal disagreement panel |
| **🎯 Archetypes** | Tactical fingerprints — radar charts per archetype, bootstrap stability findings, team lists |
| **🔍 Team Deep Dive** | Per-nation view — score decomposition, player trait heatmap, comparison panel, story callout |
| **🎲 Monte Carlo** | Simulation results — outcome probability bars, group-by-group champion and exit probabilities |
| **📋 About & Limits** | Framework explainer, data limitations, methodological choices |

### Data Sources (Dashboard)

The dashboard reads directly from these pipeline output files:

player_score/outputs/player_quality_2026.csv
player_score/outputs/player_details_2026.csv
tactical_clustering/outputs/team_archetypes.csv
composite_score/outputs/monte_carlo_2026.csv

No internet connection required after initial setup. All data is local.

---

## Notebooks

| Notebook | Phase | Purpose |
|---|---|---|
| `EDA.ipynb` | Midterm | Full exploratory analysis: data audit, metric engineering, tactical scatter, player distributions |
| `EDA_Executive.ipynb` | Midterm | Executive summary: key findings and visualisations only |
| `wc2026_analysis.ipynb` | **Final** | Complete framework: tactical clustering, player scoring, composite model, Monte Carlo simulation |

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
| **III. Tactical Identity** | KMeans clustering of 71 nations into 6 archetypes, GMM validation (ARI 0.455), outcome validation against WC 2022, four-layer archetype score derivation |
| **IV. Player Quality** | 8-step scoring pipeline, 13 metrics, Guardian blend, country-level aggregation, coverage gap analysis (median confidence 0.18) |
| **V. Composite Readiness** | 8-component model, full 48-nation rankings, volatility index, signal divergence, FIFA vs readiness delta |
| **VI. Tournament Simulation** | 10,000 Monte Carlo simulations, champion probabilities, survival curves, radar/heatmap/cohesion visualisations |
| **VII. Synthesis** | Two-layer finding, measurement limits quantified, framework philosophy, 7 directions for extension |

To reproduce all final outputs:

```bash
python run_pipeline.py
```

---

## Reproducing the Final Report

### Step 1 — Clone the Repository
```bash
git clone https://github.com/soh364/soccer-analytics-capstone-XOh.git
cd soccer-analytics-capstone-XOh
```

### Step 2 — Download the Data
```bash
python data/download_data.py
```
Note: This will download both StatsBomb (required) and Polymarket (optional) data.

### Step 3 — Install Dependencies
```bash
pip install -r requirements.txt
```
> ⚠️ `polars==1.3.0` must be pinned exactly. Later versions silently break the player scoring pipeline.

### Step 4 — Generate Raw Metric Outputs (if not already present)

The pipeline reads from `outputs/raw_metrics/`, which is generated by the midterm metric pipeline. If these files are not present:
```bash
python run_metrics.py men_tournament_2022_24 recent_club_players
```

This produces the 8 team tactical CSVs in `outputs/raw_metrics/men_tourn_2022_24/` and the player quality metrics in `outputs/raw_metrics/recent_club_players/` across three seasons. If you are reproducing from a cloned repository where these files are already committed, skip this step.

### Step 5 — Run the Final Pipeline
```bash
# Full run — all outputs + 10,000 Monte Carlo simulations (~8 min)
python run_pipeline.py

# Suppress detailed output tables
python run_pipeline.py --quiet

# Faster testing — 1,000 simulations instead of 10,000
python run_pipeline.py --mc-sims 1000

# Skip Monte Carlo entirely (~30s faster)
python run_pipeline.py --skip-mc
```

Outputs written to:
- `player_score/outputs/player_quality_2026.csv`
- `player_score/outputs/player_details_2026.csv`
- `tactical_clustering/outputs/team_archetypes.csv`
- `composite_score/outputs/team_readiness_2026.csv`
- `composite_score/outputs/monte_carlo_2026.csv`

### Step 6 — Open the Notebook & Launch the Dashboard

Open `wc2026_analysis.ipynb` and run all cells in order. The setup cell (`cell 2`) must be run first — it sets `PROJECT_ROOT`, adds all package paths to `sys.path`, and defines shared constants (`FIGURES_DIR`, `PALETTE`). Running cells out of order or after a kernel restart without re-running cell 2 is the most common source of `NameError` and `ModuleNotFoundError`.

To explore all 48 nations interactively, run the dashboard after completing the pipeline:
```bash
wc2026_dashboard app.py
```

Then open `http://127.0.0.1:8050` in your browser. The dashboard requires all pipeline outputs to be present (`composite_score/outputs/`, `player_score/outputs/`, `tactical_clustering/outputs/`).
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

## Readiness Score: Weighting Logic

The composite readiness score combines eight components. Weights reflect the relative predictive importance of each signal for a 54-game, 48-team tournament, informed by the outcome validation in Section III. 

| Component | Weight | Rationale |
|---|---|---|
| Player quality | 35% | Primary discriminator within archetypes: EDA confirmed individual quality separates outcomes more than any other signal |
| Tactical archetype | 20% | Sets the floor: no Low Intensity or Moderate Possession team has reached a WC quarter-final in the validation sample |
| FIFA ranking | 15% | External validity anchor: most widely accepted independent quality signal |
| Club cohesion | 10% | Proxy for tactical familiarity: log-scaled squad concentration |
| Squad age | 5% | Physical prime window: peak 26–29, penalties outside |
| Coach tenure | 5% | Tactical stability: sweet spot 3–7 years, staleness penalty beyond 10 |
| Tournament experience | 5% | Knowhow under pressure: log-scaled WC appearances |
| Confederation bonus | 5% | Host advantage: travel, recovery, crowd (×1.05 for US/CAN/MEX) |

When archetype data is unavailable (9 of 48 nations), the 20% tactical weight is redistributed proportionally across the remaining components. Full derivation in `composite_score/composite_scorer.py` and Section 5.1 of `wc2026_analysis.ipynb`.

---

## Key Results

**Composite Readiness Top 5:** France (71.18), Argentina (69.81), Spain (69.21), Germany (65.69), Brazil (65.32)

**Monte Carlo Champion Probabilities:** France 8.5%, Spain 7.8%, Argentina 7.4%, Germany 7.3%, Brazil 6.6%

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

## Project Structure

```
soccer-analytics-capstone-template/
│
├── wc2026_analysis.ipynb            ← Final report notebook
├── wc2026_dashboard.ipynb           ← Interactive dashboard 
├── run_pipeline.py                  ← Final pipeline entry point
├── run_metrics.py                   ← Midterm metric generation entry point
├── requirements.txt
│
├── data/                            ← Raw data (not versioned)
│   ├── Statsbomb/                   ← StatsBomb open event data (parquet)
│   │   ├── matches.parquet
│   │   ├── events.parquet
│   │   └── lineups.parquet
│   └── Polymarket/                  ← Historical prediction market data (optional)
│
├── outputs/                         ← Generated metric files — critical pipeline input
│   └── raw_metrics/                 ← Produced by run_metrics.py; read by run_pipeline.py
│       ├── men_tourn_2022_24/       ← 8 team tactical metric CSVs (PPDA, EPR, etc.)
│       └── recent_club_players/     ← Player quality metrics by season
│           ├── 2021_2022/
│           ├── 2022_2023/
│           └── 2023_2024/
│
├── src/                             ← Midterm metric calculation modules
│   └── metrics/                     ← Called by run_metrics.py to build outputs/raw_metrics/
│
├── player_score/                    ← Player scoring pipeline
│   ├── player_score_pipeline.py
│   ├── player_aggregator.py
│   ├── aggregation.py
│   ├── loader.py
│   ├── guardians_2025.py
│   ├── rosters_2026.py              ← 2026 squad rosters (48 nations)
│   ├── player_metrics_config.py
│   ├── player_position_map.py
│   ├── club_mapping_2026.py
│   ├── steps/
│   │   ├── filter.py
│   │   ├── decay.py
│   │   ├── normalization.py
│   │   ├── shrinkage.py
│   │   ├── segmentation.py
│   │   └── scoring.py
│   └── outputs/
│       ├── player_quality_2026.csv
│       └── player_details_2026.csv
│
├── tactical_clustering/             ← Tactical clustering pipeline
│   ├── tc_pipeline.py
│   ├── tc_data.py
│   ├── tc_preprocessing.py
│   ├── tc_k_selection.py
│   ├── tc_clustering.py
│   ├── tc_validation.py
│   ├── tc_visualisation.py
│   ├── tc_outcome_validation.py
│   ├── figures/
│   └── outputs/
│       └── team_archetypes.csv
│
├── composite_score/                 ← Composite scoring + simulation
│   ├── composite_scorer.py
│   ├── external_factors.py
│   ├── club_cohesion.py
│   ├── monte_carlo.py
│   └── outputs/
│       ├── team_readiness_2026.csv
│       └── monte_carlo_2026.csv
│
├── eda/                             ← Midterm EDA notebooks and helpers
│   ├── EDA.ipynb
│   ├── EDA_Executive.ipynb
│   ├── analysis/
│   ├── figures/
│   └── processed/
│
├── notebook/                        ← Notebook figures output directory
│   └── figures/
│
├── scripts/                         ← Utility scripts
├── template/                        ← Dashboard template (from capstone template)
└── tests/                           ← Test suite
```

**Data Flow:**
```
data/Statsbomb/
│
▼ run_metrics.py (via src/metrics/)
outputs/raw_metrics/
│
▼ run_pipeline.py
player_score/outputs/     tactical_clustering/outputs/     composite_score/outputs/
│                           │                                │
└───────────────────────────┴────────────────────────────────┘
│
▼
wc2026_analysis.ipynb & wc2026_dashboard.py

```

> ⚠️ `outputs/raw_metrics/` is the critical handoff between the midterm pipeline (`run_metrics.py`) and the final pipeline (`run_pipeline.py`). If these files are absent, the final pipeline will fail at Stage 1.

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
├── reference.parquet
└── three_sixty.parquet
```

Download via:

```bash
python data/download_data.py
```
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