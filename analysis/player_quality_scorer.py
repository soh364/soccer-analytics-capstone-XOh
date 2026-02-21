"""
Calculate 0-100 quality scores for players using:
- 13 metrics across 4 trait categories
- Time-decay weighting (recent performance weighted higher)
- Position-based trait weighting
- League quality multipliers
"""

import polars as pl
import pandas as pd
import numpy as np
from player_metrics_config import PLAYER_METRICS, TRAIT_CATEGORIES
from player_positions import PLAYER_POSITION_MAP


class PlayerQualityScorer:
    """Calculate quality scores for players."""

    def __init__(self, player_data, current_year=2026, decay_lambda=0.6, latest_team_map=None):
        self.player_data = player_data
        self.current_year = current_year
        self.decay_lambda = decay_lambda
        self.min_minutes = 450
        self.latest_team_map = latest_team_map

    def calculate_time_decay_weight(self, season_year):
        """Calculate time decay weight for a season."""
        years_ago = self.current_year - season_year
        return np.exp(-self.decay_lambda * years_ago)

    def aggregate_player_metrics(self, verbose=True):
        """Aggregate all metrics per player with time-decay weighting."""
        if verbose:
            print("\n" + "="*70)
            print("AGGREGATING PLAYER METRICS")
            print("="*70)

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

            # Find volume column
            volume_col = next((c for c in ['matches', 'minutes_played', 'total_mins']
                               if c in df.columns), None)

            # Add time decay weight
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

            # Apply volume filter
            if volume_col:
                threshold = MIN_MATCHES if volume_col == 'matches' else MIN_MINUTES
                agg_df = agg_df.filter(pl.col('volume') >= threshold)

            # Calculate weighted average metric
            agg_df = agg_df.with_columns(
                (pl.col('w_sum') / pl.col('w_total')).alias(metric_name)
            ).select(['player', metric_name])

            player_scores.append(agg_df)

            if verbose:
                print(f"  ✓ {metric_name}: {len(agg_df)} players")

        if not player_scores:
            raise ValueError("No metrics aggregated")

        # Join all metrics
        combined = player_scores[0]
        for next_df in player_scores[1:]:
            combined = combined.join(next_df, on='player', how='left')

        if verbose:
            print(f"\n✓ Combined: {len(combined)} unique players")

        return combined

    def normalize_to_percentile(self, df, metric_cols):
        """Normalize metrics to 0-100 percentile scale."""
        for col in metric_cols:
            if col not in df.columns:
                continue
            df = df.with_columns(
                pl.col(col).rank(method='average')
                  .truediv(pl.col(col).count())
                  .mul(100)
                  .alias(f'{col}_percentile')
            )
        return df

    def calculate_trait_scores(self, player_metrics_df, verbose=True):
        """Calculate 4 trait scores from percentile columns."""
        for trait_name, metric_list in TRAIT_CATEGORIES.items():
            percentile_cols = [f'{m}_percentile' for m in metric_list
                               if f'{m}_percentile' in player_metrics_df.columns]
            if percentile_cols:
                player_metrics_df = player_metrics_df.with_columns(
                    trait_avg=pl.mean_horizontal(percentile_cols),
                    trait_max=pl.max_horizontal(percentile_cols)
                ).with_columns(
                    # 70/30: rewards elite peak while acknowledging rounded contribution
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
                "Gazélec Ajaccio", "Angers SCO",
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
            return {'Mobility_Intensity': 0.9, 'Progression': 0.6,
                    'Control': 0.5, 'Final_Third_Output': 0.2}
        elif any(x in pos for x in ['left back', 'right back', 'wing back']):
            return {'Mobility_Intensity': 0.7, 'Progression': 1.0,
                    'Control': 0.7, 'Final_Third_Output': 0.5}
        elif 'defensive midfield' in pos:
            return {'Mobility_Intensity': 1.0, 'Progression': 0.8,
                    'Control': 0.9, 'Final_Third_Output': 0.3}
        elif 'center midfield' in pos or 'centre midfield' in pos:
            return {'Mobility_Intensity': 0.8, 'Progression': 1.0,
                    'Control': 1.0, 'Final_Third_Output': 0.7}
        elif 'attacking midfield' in pos:
            return {'Mobility_Intensity': 0.4, 'Progression': 0.9,
                    'Control': 1.0, 'Final_Third_Output': 1.0}
        elif 'wing' in pos and 'back' not in pos:
            return {'Mobility_Intensity': 0.9, 'Progression': 0.8,
                    'Control': 0.8, 'Final_Third_Output': 1.0}
        elif any(x in pos for x in ['forward', 'striker']):
            return {'Mobility_Intensity': 0.3, 'Progression': 0.5,
                    'Control': 0.6, 'Final_Third_Output': 1.0}
        else:
            return {'Mobility_Intensity': 1.0, 'Progression': 1.0,
                    'Control': 1.0, 'Final_Third_Output': 1.0}

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
            print("\n" + "="*70)
            print("POSITION-WEIGHTED TRAITS")
            print("="*70)
            print("Position distribution (top 10):")
            for pos, count in sorted(position_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
                print(f"  {pos}: {count} players")

        return pl.from_pandas(df)

    def calculate_overall_quality(self, player_metrics_df):
        """Calculate overall quality score from trait scores."""
        trait_cols = list(TRAIT_CATEGORIES.keys())
        available_traits = [t for t in trait_cols if t in player_metrics_df.columns]

        if not available_traits:
            raise ValueError("No trait scores available")

        player_metrics_df = player_metrics_df.with_columns(
            pl.concat_list(available_traits).list.mean().alias('overall_quality')
        )
        return player_metrics_df

    def score_players(self, verbose=True):
        """Full scoring pipeline: aggregate → normalize → traits → position → quality."""

        # Step 1: Aggregate metrics with time decay
        player_metrics = self.aggregate_player_metrics(verbose=verbose)

        # Step 2: Normalize to percentiles
        metric_cols = [m for m in PLAYER_METRICS.keys() if m in player_metrics.columns]
        player_metrics = self.normalize_to_percentile(player_metrics, metric_cols)

        # Step 3: Calculate trait scores
        player_metrics = self.calculate_trait_scores(player_metrics, verbose=verbose)

        # Step 4: Join latest_team_map (club + position)
        if self.latest_team_map is not None:
            cols_to_drop = [c for c in ['latest_club', 'position'] if c in player_metrics.columns]
            if cols_to_drop:
                player_metrics = player_metrics.drop(cols_to_drop)
            player_metrics = player_metrics.join(self.latest_team_map, on='player', how='left')

        # Step 5: Resolve missing positions via static map
        player_metrics = player_metrics.with_columns(
            pl.struct(["player", "position"]).map_elements(
                lambda x: self.get_player_position(x["player"], x["position"]),
                return_dtype=pl.String
            ).alias("position")
        )

        # Step 6: Apply position-based trait weights
        if 'position' in player_metrics.columns:
            player_metrics = self.apply_position_weights(player_metrics, verbose=verbose)

        # Step 7: Calculate overall quality
        player_metrics = self.calculate_overall_quality(player_metrics)

        # Step 8: Apply league multiplier
        player_metrics = player_metrics.with_columns(
            pl.col('latest_club')
            .map_elements(self.get_league_multiplier, return_dtype=pl.Float64)
            .alias('league_mult')
        ).with_columns(
            (pl.col('overall_quality') * pl.col('league_mult'))
            .clip(0, 100)
            .alias('overall_quality')
        )

        if verbose:
            print("\n" + "="*70)
            print("QUALITY SCORING COMPLETE")
            print("="*70)
            print(f"Players scored: {len(player_metrics)}")
            print(f"Avg overall quality: {player_metrics['overall_quality'].mean():.1f}")

        return player_metrics