"""Calculate 0-100 quality scores for players using 13 metrics across 4 trait categories, time-decay weighting, position-based weights, and league multipliers."""

import polars as pl
import pandas as pd
import numpy as np
from player_metrics_config import PLAYER_METRICS, TRAIT_CATEGORIES
from player_positions import POSITION_OVERRIDES


class PlayerQualityScorer:
    """Calculate quality scores for players."""

    def __init__(self, player_data, current_year=2026, decay_lambda=0.6, latest_team_map=None):
        self.player_data = player_data
        self.current_year = current_year
        self.decay_lambda = decay_lambda
        self.min_minutes = 900
        self.latest_team_map = latest_team_map

    def calculate_time_decay_weight(self, season_year):
        """Calculate time decay weight for a season."""
        years_ago = self.current_year - season_year
        return np.exp(-self.decay_lambda * years_ago)

    def aggregate_player_metrics(self, verbose=True):
        """Aggregate all metrics per player with time-decay weighting."""
        if verbose:
            print("Aggregating player metrics...")

        player_scores = []
        MIN_MATCHES = 5
        MIN_MINUTES = 400

        for metric_name, config in PLAYER_METRICS.items():
            filename = config['file']
            column = config['column']

            if filename not in self.player_data:
                continue

            df = self.player_data[filename]

            if column not in df.columns or 'player' not in df.columns:
                continue

            volume_col = next((c for c in ['matches', 'minutes_played', 'total_mins']
                               if c in df.columns), None)

            df = df.with_columns(
                pl.col('season_year').map_elements(
                    self.calculate_time_decay_weight,
                    return_dtype=pl.Float64
                ).alias('time_weight')
            )

            # Aggregate with weighting
            agg_df = (
                df.group_by('player')
                .agg([
                    (pl.col(column) * pl.col('time_weight')).sum().alias('w_sum'),
                    pl.col('time_weight').sum().alias('w_total'),
                    (pl.col(volume_col).sum() if volume_col else pl.len()).alias('volume')
                ])
            )

            if volume_col:
                default_threshold = MIN_MATCHES if volume_col == 'matches' else MIN_MINUTES
                metric_threshold = config.get('min_volume', default_threshold)
                agg_df = agg_df.filter(pl.col('volume') >= metric_threshold)

            agg_df = agg_df.with_columns(
                (pl.col('w_sum') / pl.col('w_total')).alias(metric_name)
            ).select(['player', metric_name, 'volume'])

            player_scores.append(agg_df)

            if verbose:
                print(f"  {metric_name}: {len(agg_df)} players")

        if not player_scores:
            raise ValueError("No metrics aggregated")

        combined = player_scores[0]
        for next_df in player_scores[1:]:
            combined = combined.join(
                next_df.drop('volume'),
                on='player', 
                how='left'
            )

        if 'finishing_quality' in combined.columns:
            global_mean = combined['finishing_quality'].mean()
            k = 10
            combined = combined.with_columns(
                ((pl.col('finishing_quality') * pl.col('volume') + global_mean * k)
                / (pl.col('volume') + k))
                .alias('finishing_quality')
            )

        # Drop volume before staleness penalty
        combined = combined.drop('volume')

        most_recent_year = max(
            df['season_year'].max()
            for df in self.player_data.values()
            if 'season_year' in df.columns
        )

        players_current = set()
        players_one_season_ago = set()

        for df in self.player_data.values():
            if 'player' in df.columns and 'season_year' in df.columns:
                current = df.filter(
                    pl.col('season_year') == most_recent_year
                )['player'].to_list()
                players_current.update(current)
                
                one_ago = df.filter(
                    pl.col('season_year') == most_recent_year - 1
                )['player'].to_list()
                players_one_season_ago.update(one_ago)

        players_one_season_ago = players_one_season_ago - players_current

        combined = combined.with_columns(
            pl.when(pl.col('player').is_in(list(players_current)))
            .then(pl.lit(1.0))
            .when(pl.col('player').is_in(list(players_one_season_ago)))
            .then(pl.lit(0.85))
            .otherwise(pl.lit(0.70))
            .alias('recency_factor')
        )

        if verbose:
            n_current = len(combined.filter(pl.col('recency_factor') == 1.0))
            n_one_ago = len(combined.filter(pl.col('recency_factor') == 0.85))
            n_stale = len(combined.filter(pl.col('recency_factor') == 0.70))
            print(f"Recency: {n_current} current, {n_one_ago} one season ago, {n_stale} stale")
            print(f"Combined: {len(combined)} unique players")

        return combined

    def normalize_to_percentile(self, df, metric_cols):
        """Normalize metrics to 0-100 percentile scale."""
        for col in metric_cols:
            if col not in df.columns:
                continue
            df = df.with_columns(
                pl.col(col).rank(method='average')
                  .truediv(pl.col(col).count())
                  .mul(99)
                  .alias(f'{col}_percentile')
            )
        return df

    def calculate_trait_scores(self, player_metrics_df, verbose=True):
        """Calculate trait scores from percentile metrics."""
        for trait_name, metric_list in TRAIT_CATEGORIES.items():
            percentile_cols = [f'{m}_percentile' for m in metric_list
                            if f'{m}_percentile' in player_metrics_df.columns]
            if percentile_cols:
                filled = [pl.col(c).fill_null(0) for c in percentile_cols]
                player_metrics_df = player_metrics_df.with_columns(
                    trait_avg=pl.mean_horizontal(filled),
                    trait_max=pl.max_horizontal(filled)
                ).with_columns(
                    (pl.col("trait_max") * 0.7 + pl.col("trait_avg") * 0.3).alias(trait_name)
                ).drop(["trait_avg", "trait_max"])
        return player_metrics_df
    
    @staticmethod
    def get_league_multiplier(team_name):
        """Return league quality multiplier based on club tier."""
        CLUB_TIERS = {
            "Tier_1": [
                "Liverpool", "Manchester City", "Arsenal", "Chelsea",
                "Borussia Dortmund", "Manchester United", "Barcelona", "Real Madrid",
                "Paris Saint-Germain", "Inter Milan", "AC Milan", "Juventus",
                "Bayern Munich", "Atlético Madrid",
            ],
            "Tier_2": [
                "VfB Stuttgart", "Tottenham Hotspur", "RB Leipzig", "Bayer Leverkusen",
                "Real Sociedad", "Athletic Club", "Lazio", "Roma", "Atalanta",
                "Napoli", "Fiorentina", "Aston Villa", "Newcastle United",
                "West Ham United", "Brighton & Hove Albion", "Eintracht Frankfurt",
                "Benfica", "Porto", "Sporting CP", "Ajax", "PSV Eindhoven",
                "Feyenoord", "Celtic", "Rangers", "Galatasaray", "Fenerbahçe",
                "Lens", "Stade de Reims", "Guingamp",
            ],
            "Tier_3": [
                "Wolfsburg", "Werder Bremen", "Freiburg", "Augsburg", "Mainz",
                "Hoffenheim", "Getafe", "Villarreal", "Celta Vigo", "Sevilla",
                "Real Betis", "Nice", "Lille", "Monaco", "Rennes", "Lyon",
                "Marseille", "Everton", "Crystal Palace", "Fulham",
                "Nantes", "Angers", "Strasbourg", "Montpellier", "Bordeaux",
                "Köln", "Schalke 04", "Hamburger SV", "Hertha Berlin",
                "Hannover 96", "Darmstadt 98",
            ],
            "Tier_4": [
                "Bochum", "Heidenheim", "Southampton",
                "Al Ahli", "Al Hilal", "Al Nassr", "Al Ittihad",
                "Inter Miami", "LA Galaxy", "New York City",
                "Santos", "Flamengo", "Boca Juniors",
            ],
            "Tier_5": [
                "Mumbai City", "Hyderabad", "Kerala Blasters", "Bengaluru",
                "ATK Mohun Bagan", "Chennaiyin", "Jamshedpur", "Odisha",
                "Gazélec Ajaccio", "Angers SCO", "Al-Arabi Qatar"
            ]}
        if team_name in CLUB_TIERS["Tier_1"]: return 1.3
        if team_name in CLUB_TIERS["Tier_2"]: return 1.15
        if team_name in CLUB_TIERS["Tier_3"]: return 1.05
        if team_name in CLUB_TIERS["Tier_4"]: return 0.85
        if team_name in CLUB_TIERS["Tier_5"]: return 0.65
        return 1.0

    def _get_position_weights(self, position):
        """Return trait weights based on position."""
        if pd.isna(position) or position is None:
            return {'Mobility_Intensity': 1.0, 'Progression': 1.0,
                    'Control': 1.0, 'Final_Third_Output': 1.0}

        pos = str(position).lower()

        if 'goalkeeper' in pos:
            return {'Mobility_Intensity': 0.1, 'Progression': 0.3,
                    'Control': 0.2, 'Final_Third_Output': 0.0}
        elif 'center back' in pos or 'centre back' in pos:
            return {'Mobility_Intensity': 0.45, 'Progression': 0.55,
                    'Control': 0.55, 'Final_Third_Output': 0.15}
        elif any(x in pos for x in ['left back', 'right back', 'wing back']):
            return {'Mobility_Intensity': 0.85, 'Progression': 1.0,
                    'Control': 0.65, 'Final_Third_Output': 0.55}
        elif 'defensive midfield' in pos:
            return {'Mobility_Intensity': 0.75, 'Progression': 0.75,
                    'Control': 0.85, 'Final_Third_Output': 0.4}
        elif 'center midfield' in pos or 'centre midfield' in pos:
            return {'Mobility_Intensity': 0.80, 'Progression': 0.80,
                    'Control': 0.80, 'Final_Third_Output': 0.65}
        elif 'attacking midfield' in pos:
            return {'Mobility_Intensity': 0.65, 'Progression': 0.85,
                    'Control': 0.95, 'Final_Third_Output': 1.0}
        elif 'wing' in pos and 'back' not in pos:
            return {'Mobility_Intensity': 1.0, 'Progression': 0.80,
                    'Control': 0.75, 'Final_Third_Output': 1.0}
        elif any(x in pos for x in ['forward', 'striker']):
            return {'Mobility_Intensity': 0.50, 'Progression': 0.55,
                    'Control': 0.75, 'Final_Third_Output': 1.0}
        else:
            return {'Mobility_Intensity': 0.75, 'Progression': 0.85,
                    'Control': 0.85, 'Final_Third_Output': 0.75}

    def get_player_position(self, player_name, current_pos):
        """Resolve position: use current if available, else fall back to static map."""
        if current_pos is not None and not pd.isna(current_pos) and current_pos != "":
            return current_pos
        return PLAYER_POSITION_MAP.get(player_name, None)

    def apply_position_weights(self, player_metrics, verbose=True):
        """Apply position-based weights to trait scores."""
        df = player_metrics.to_pandas()
        trait_cols = ['Mobility_Intensity', 'Progression', 'Control', 'Final_Third_Output']
        position_counts = {}

        for idx, row in df.iterrows():
            position = row.get('position', None)
            weights = self._get_position_weights(position)
            pos_key = str(position) if pd.notna(position) else 'Unknown'
            position_counts[pos_key] = position_counts.get(pos_key, 0) + 1
            for trait in trait_cols:
                if trait in df.columns and pd.notna(row[trait]):
                    df.at[idx, trait] = row[trait] * weights[trait]

        if verbose:
            print("Position-weighted traits:")
            for pos, count in sorted(position_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
                print(f"  {pos}: {count} players")

        return pl.from_pandas(df)

    def calculate_overall_quality(self, player_metrics_df):
        """Calculate overall quality score from trait scores using weighted mean."""
        df = player_metrics_df.to_pandas()
        traits = ['Mobility_Intensity', 'Progression', 'Control', 'Final_Third_Output']

        for idx, row in df.iterrows():
            pos = row.get('position', None)
            weights = self._get_position_weights(pos)

            weighted_sum = 0
            weight_total = 0
            for t in traits:
                if t in df.columns:
                    if pd.notna(row[t]):
                        weighted_sum += row[t] * weights[t]
                    # Always add to denominator whether null or not
                    weight_total += weights[t]

            df.at[idx, 'overall_quality'] = weighted_sum / weight_total if weight_total > 0 else 0

        return pl.from_pandas(df)

    def score_players(self, verbose=True):
        """Calculate final quality scores for all players."""
        player_metrics = self.aggregate_player_metrics(verbose=verbose)

        if self.latest_team_map is not None:
            player_metrics = player_metrics.join(
                self.latest_team_map.select(['player', 'latest_club']),
                on='player', how='left'
            )
        
        metric_cols = [m for m in PLAYER_METRICS.keys() if m in player_metrics.columns]
        player_metrics = player_metrics.with_columns(
            pl.col('latest_club')
            .map_elements(self.get_league_multiplier, return_dtype=pl.Float64)
            .alias('league_mult')
        )
        for col in metric_cols:
            player_metrics = player_metrics.with_columns(
                (pl.col(col) * pl.col('league_mult')).alias(col)
            )

        player_metrics = self.normalize_to_percentile(player_metrics, metric_cols)

        if self.latest_team_map is not None:
            cols_to_drop = [c for c in ['latest_club', 'position'] if c in player_metrics.columns]
            if cols_to_drop:
                player_metrics = player_metrics.drop(cols_to_drop)
            player_metrics = player_metrics.join(self.latest_team_map, on='player', how='left')

        player_metrics = player_metrics.with_columns(
            pl.struct(["player", "position"]).map_elements(
                lambda x: self.get_player_position(x["player"], x["position"]),
                return_dtype=pl.String
            ).alias("position")
        )

        player_metrics = self.calculate_trait_scores(player_metrics, verbose=verbose)

        if 'position' in player_metrics.columns:
            player_metrics = self.apply_position_weights(player_metrics, verbose=verbose)

        player_metrics = self.calculate_overall_quality(player_metrics)

        player_metrics = player_metrics.with_columns(
            (pl.col('overall_quality') * pl.col('recency_factor')).clip(0, 100).alias('overall_quality')
        )

        if verbose:
            print("Quality scoring complete")
            print(f"Players scored: {len(player_metrics)}")
            print(f"Avg quality: {player_metrics['overall_quality'].mean():.1f}")

        return player_metrics