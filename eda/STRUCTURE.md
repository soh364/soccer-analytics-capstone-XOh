# EDA Project Structure

All project components are now consolidated within the `eda/` folder for a self-contained analysis environment.

## Directory Layout

```
eda/
├── EDA_Soomi.ipynb          # Main analysis notebook
├── analysis/                 # All analysis modules
│   ├── visualization.py
│   ├── clustering_analysis.py
│   ├── player_quality_scorer.py
│   ├── data_loader.py
│   ├── trait_mapper.py
│   ├── system_fit_engine.py
│   ├── rosters_2026.py
│   ├── success_analyzer.py
│   ├── profile_builder.py
│   ├── player_metrics.py
│   ├── player_metrics_config.py
│   ├── tournament_progression.py
│   └── club_mapping_2026.py
├── figures/                 # All output visualizations
│   ├── cluster_optimization.png
│   ├── archetype_radars.png
│   ├── tactical_pca.png
│   └── player_comparison_pizzas.png
├── processed/               # All output data files
│   ├── tournament_profiles_8d_2022_24.csv
│   ├── archetype_trait_requirements.csv
│   ├── tournament_success_with_archetypes.csv
│   ├── archetype_success_rates.csv
│   └── player_quality_scores_final.csv
└── templates/              # Dashboard templates
```

## Path Configuration

In the notebook setup cell, all paths are defined relative to the `eda/` folder:

```python
sys.path.insert(0, str(Path.cwd() / 'analysis'))

DATA_DIR = Path("..") / "data" / "Statsbomb"  # Points to project root data
PROCESSED_DIR = Path("processed")              # Local to eda/
FIGURES_DIR = Path("figures")                  # Local to eda/
```

## Key Changes

- **Single Notebook**: `EDA_Soomi.ipynb` orchestrates all analysis
- **Local Imports**: All 13+ analysis modules are in `analysis/` subfolder
- **Contained Outputs**: All figures and processed data stay in `eda/` folder
- **Clean Separation**: Raw data remains at project root (`../data/`)
