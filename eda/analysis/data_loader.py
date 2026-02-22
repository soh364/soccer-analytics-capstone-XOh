from pathlib import Path
import polars as pl
from player_metrics_config import PLAYER_METRICS


def parse_season_year(season_name: str) -> int:
    """
    Extract the latest year from a season_name string for time decay weighting.
    '2023/2024' -> 2024, '2022' -> 2022
    """
    parts = str(season_name).replace('-', '/').split('/')
    return max(int(p) for p in parts if p.isdigit())


def _season_folder_to_name(folder_name: str) -> str:
    """Convert folder name back to season_name. '2021_2022' -> '2021/2022'"""
    return folder_name.replace('_', '/', 1)


class TournamentDataLoader8D:
    """Load 8-dimensional team metrics for tournament analysis."""

    def __init__(self, base_dir='../outputs/raw_metrics'):
        self.base_dir = Path(base_dir)
        self.metric_files = {
            'ppda': {
                'file': 'possession__team__ppda.csv',
                'columns': ['match_id', 'team', 'ppda'],
                'dimension': 'D1: Pressing Intensity'
            },
            'field_tilt': {
                'file': 'possession__team__field_tilt.csv',
                'columns': ['match_id', 'team', 'field_tilt_pct'],
                'dimension': 'D2: Territorial Dominance'
            },
            'possession_pct': {
                'file': 'possession__team__percentage.csv',
                'columns': ['match_id', 'team', 'possession_pct'],
                'dimension': 'D3: Ball Control'
            },
            'epr': {
                'file': 'possession__team__value_epr.csv',
                'columns': ['match_id', 'team', 'epr'],
                'dimension': 'D4: Possession Efficiency'
            },
            'line_height': {
                'file': 'defensive__team__line_height.csv',
                'columns': ['match_id', 'team', 'defensive_line_height'],
                'dimension': 'D5: Defensive Positioning'
            },
            'xg': {
                'file': 'xg__team__totals.csv',
                'columns': ['match_id', 'team', 'total_xg'],
                'dimension': 'D6: Attacking Threat'
            },
            'progression': {
                'file': 'progression__team__summary.csv',
                'columns': ['match_id', 'team', 'progressive_passes', 'progressive_carries'],
                'dimension': 'D7: Progression Style'
            },
            'buildup': {
                'file': 'advanced__team__xg_buildup.csv',
                'columns': ['match_id', 'team', 'avg_xg_per_buildup_possession'],
                'dimension': 'D8: Build-up Quality'
            }
        }

    def load_scope(self, scope_name, verbose=True):
        scope_dir = self.base_dir / scope_name
        if not scope_dir.exists():
            raise FileNotFoundError(
                f"Scope directory not found: {scope_dir.absolute()}\n"
                f"Run 'python run_metrics.py {scope_name}' first."
            )

        if verbose:
            print(f"\n{'='*70}")
            print(f"LOADING: {scope_name.upper()}")
            print(f"{'='*70}")

        metrics = {}
        for metric_key, config in self.metric_files.items():
            filepath = scope_dir / config['file']
            if not filepath.exists():
                if verbose:
                    print(f"  ⚠️  {config['file']}: NOT FOUND")
                continue

            df = pl.read_csv(filepath)
            missing_cols = [col for col in config['columns'] if col not in df.columns]
            if missing_cols:
                if verbose:
                    print(f"  ⚠️  {config['file']}: Missing columns {missing_cols}")
                continue

            df = df.select(config['columns'])
            metrics[metric_key] = df

            if verbose:
                print(f"  ✓ {config['file']} — {len(df):,} rows, {df['team'].n_unique()} teams")

        if verbose:
            print(f"\nSUCCESS: {len(metrics)}/{len(self.metric_files)} files loaded")

        return metrics


def load_tournament_data_8d(scope_name, verbose=True):
    loader = TournamentDataLoader8D()
    return loader.load_scope(scope_name, verbose=verbose)


class PlayerDataLoader:
    """
    Load player metrics from per-season subfolders with time-decay weighting.
    
    Expected structure:
        base_dir/
          recent_club_players/
            2021_2022/   <- one folder per season
              advanced__player__xg_chain.csv
              ...
            2022_2023/
            2023_2024/
    """

    def __init__(self, base_dir='../outputs/raw_metrics'):
        self.base_dir = Path(base_dir)
        self.metrics_config = PLAYER_METRICS

    def _discover_season_folders(self, scope_dir: Path) -> list[tuple[str, Path]]:
        """
        Return (season_name, folder_path) pairs for all season subfolders,
        sorted chronologically.
        """
        seasons = []
        for folder in sorted(scope_dir.iterdir()):
            if folder.is_dir():
                season_name = _season_folder_to_name(folder.name)
                seasons.append((season_name, folder))
        return seasons

    def load_season(self, season_folder: Path, season_name: str, verbose=True) -> dict:
        """
        Load all metric files for a single season folder.
        Adds season_name and season_year columns to every DataFrame.
        """
        season_year = parse_season_year(season_name)

        # Get unique files needed
        files_to_load = {}
        for metric_name, config in self.metrics_config.items():
            filename = config['file']
            if filename not in files_to_load:
                files_to_load[filename] = []
            files_to_load[filename].append((metric_name, config['column']))

        loaded = {}
        for filename in files_to_load:
            filepath = season_folder / filename
            if not filepath.exists():
                if verbose:
                    print(f"    ⚠️  {filename}: NOT FOUND")
                continue

            df = pl.read_csv(filepath)

            # Ensure season_name column exists (progression profile is missing it)
            if 'season_name' not in df.columns:
                df = df.with_columns(pl.lit(season_name).alias('season_name'))

            # Always add/overwrite season_year for time decay
            df = df.with_columns(pl.lit(season_year).cast(pl.Int32).alias('season_year'))

            loaded[filename] = df

            if verbose:
                n_players = df['player'].n_unique() if 'player' in df.columns else 'N/A'
                print(f"    ✓ {filename} — {len(df):,} rows, {n_players} players")

        return loaded

    def load_scope(self, scope_name, verbose=True) -> dict:
        """
        Load all seasons within a scope, combining into one DataFrame per file.
        
        Returns:
            dict[filename -> polars DataFrame] with all seasons stacked,
            season_name and season_year columns present on every DataFrame.
        """
        scope_dir = self.base_dir / scope_name
        if not scope_dir.exists():
            raise FileNotFoundError(f"Scope not found: {scope_dir}")

        season_folders = self._discover_season_folders(scope_dir)
        if not season_folders:
            raise FileNotFoundError(f"No season subfolders found in {scope_dir}")

        if verbose:
            print(f"\n{'='*70}")
            print(f"Loading: {scope_name}")
            print(f"Seasons: {[s for s, _ in season_folders]}")
            print(f"{'='*70}")

        all_data: dict[str, list[pl.DataFrame]] = {}

        for season_name, season_folder in season_folders:
            if verbose:
                print(f"\n  ── {season_name} ──")
            season_data = self.load_season(season_folder, season_name, verbose=verbose)
            for filename, df in season_data.items():
                all_data.setdefault(filename, []).append(df)

        # Stack seasons
        combined = {}
        for filename, dfs in all_data.items():
            combined[filename] = pl.concat(dfs, how='diagonal')

        if verbose:
            print(f"\n{'='*70}")
            print(f"COMBINED SUMMARY")
            print(f"{'='*70}")
            for filename, df in combined.items():
                n_players = df['player'].n_unique() if 'player' in df.columns else 'N/A'
                seasons = sorted(df['season_name'].unique().to_list())
                print(f"  {filename}")
                print(f"    {len(df):,} rows | {n_players} players | {seasons}")

        return combined

    def load_multiple_scopes(self, scope_names, verbose=True) -> dict:
        """
        Load and combine multiple scopes (e.g. club + tournament data).
        Handles legacy (scope_name, year) tuple format gracefully.
        """
        all_data: dict[str, list[pl.DataFrame]] = {}

        for scope_name in scope_names:
            if isinstance(scope_name, tuple):
                scope_name = scope_name[0]
            scope_data = self.load_scope(scope_name, verbose=verbose)
            for filename, df in scope_data.items():
                all_data.setdefault(filename, []).append(df)

        combined = {f: pl.concat(dfs, how='diagonal') for f, dfs in all_data.items()}
        return combined


def load_player_data_for_scoring(scopes_to_load, verbose=True) -> dict:
    """
    Convenience function. Accepts scope name strings or legacy (scope, year) tuples.
    
    Usage:
        player_data = load_player_data_for_scoring(['recent_club_players'])
    """
    loader = PlayerDataLoader()
    if isinstance(scopes_to_load, str):
        return loader.load_scope(scopes_to_load, verbose=verbose)
    if len(scopes_to_load) == 1:
        return loader.load_scope(scopes_to_load[0], verbose=verbose)
    return loader.load_multiple_scopes(scopes_to_load, verbose=verbose)