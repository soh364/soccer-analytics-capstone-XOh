"""
Load pre-filtered tactical metrics for tournament prediction analysis.
"""

import polars as pl
from pathlib import Path
from typing import Dict

class TacticalDataLoader:
    # Load pre-filtered tactical metrics using Polars.
    
    def __init__(self, base_dir: str = None):
        if base_dir is None:
            try:
                eda_dir = Path(__file__).parent
                project_root = eda_dir.parent
                base_dir = project_root / "outputs" / "raw_metrics"
            except:
                base_dir = Path("../outputs/raw_metrics")
        
        self.base_dir = Path(base_dir)
        
        if not self.base_dir.exists():
            raise FileNotFoundError(
                f"Metrics directory not found: {self.base_dir.absolute()}\n"
                f"Run 'python run_metrics.py <scope_name>' to generate metrics first."
            )
    
    def load_scope(self, scope_name: str, verbose: bool = True) -> Dict[str, pl.DataFrame]:
        scope_folders = {
            'men_club_2015': 'men_club_2015',
            'men_tournaments_2022_24': 'men_tourn_2022_24',
            'women_club_2018_21': 'women_club_2018_21',
            'women_tournaments_2022_25': 'women_tourn_2022_25',
            'recent_club_validation': 'recent_club_val'
        }
        
        if scope_name not in scope_folders:
            raise ValueError(
                f"Unknown scope: {scope_name}\n"
                f"Available: {list(scope_folders.keys())}"
            )
        
        scope_dir = self.base_dir / scope_folders[scope_name]
        
        if not scope_dir.exists():
            raise FileNotFoundError(
                f"Scope directory not found: {scope_dir}\n"
                f"Run: python run_metrics.py {scope_name}"
            )
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"LOADING: {scope_name.upper()}")
            print(f"{'='*60}")
            print(f"Directory: {scope_dir}\n")
        
        metric_files = {
            'qual': 'possession_quality_analysis.csv',
            'eff': 'possession_efficiency_epr.csv',
            'seq': 'possession_sequence_style.csv',
            'prog_summary': 'progression_team_summary.csv',
            'prog_detail': 'progression_team_detail.csv',
            'xg_buildup': 'advanced_xg_buildup_team.csv',
            'xg_total': 'xg_team_totals.csv',
            'ppda': 'defensive_ppda.csv',
            'def_line': 'defensive_line_height_team.csv',
            'turnover': 'defensive_high_turnovers.csv',
            'counter': 'defensive_counter_speed.csv',
            'pressure': 'defensive_pressures_team.csv',
        }
        
        metrics = {}
        errors = []
        
        for key, filename in metric_files.items():
            filepath = scope_dir / filename
            
            try:
                df = pl.read_csv(filepath)
                metrics[key] = df
                
                if verbose:
                    print(f"  {filename}")
                    print(f"     Rows: {len(df):,}")
                    if 'match_id' in df.columns:
                        print(f"     Matches: {df['match_id'].n_unique():,}")
                    if 'team' in df.columns:
                        print(f"     Teams: {df['team'].n_unique():,}")
                    print()
                    
            except Exception as e:
                errors.append(f"{filename}: {str(e)}")
                if verbose:
                    print(f"  ERROR - {filename}: {str(e)}\n")
        
        if errors:
            if verbose:
                print(f"ERRORS: {len(errors)}/{len(metric_files)} files failed")
        else:
            if verbose:
                print(f"SUCCESS: {len(metrics)}/{len(metric_files)} files loaded")
                total_memory = sum(df.estimated_size('mb') for df in metrics.values())
                print(f"Total memory: {total_memory:.2f} MB")
        
        return metrics