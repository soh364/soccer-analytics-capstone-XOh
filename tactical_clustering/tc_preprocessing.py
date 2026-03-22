"""
tc_preprocessing.py
───────────────────
Outlier capping and feature scaling for the tactical clustering pipeline.

Design decisions documented here:
- Cap PPDA and EPR at 95th percentile (not 99th) — see docstrings
- Scale AFTER capping — see cap_and_scale() docstring
"""

import numpy as np
import pandas as pd
import polars as pl
from sklearn.preprocessing import StandardScaler

from tc_data import CORE_METRICS, CLUSTER_FEATURES


# ── Capping ───────────────────────────────────────────────────────────────────
def cap_outliers(team_metrics: pl.DataFrame,
                 ppda_pct: float = 0.95,
                 epr_pct:  float = 0.95) -> tuple[pl.DataFrame, dict]:
    """
    Cap PPDA and EPR at the specified percentile thresholds.

    Why PPDA and EPR specifically?
    - PPDA: Georgia (31.53) and Slovenia (23.08) sat 4+ standard deviations
      from the median. In initial runs this caused them to form their own
      2-team cluster — a taxonomically meaningless result.
    - EPR: A 99th percentile cap still left a cluster centroid at EPR=226,
      indicating residual outlier pull. The 95th percentile corrects this.

    Why 95th and not 99th?
    - The 99th percentile preserves too much of the extreme tail in a 71-team
      dataset. At n=71, the 99th percentile is effectively the single most
      extreme value. The 95th pulls in the top ~3-4 teams per feature while
      preserving the directional signal (Georgia is still the most passive team).

    Parameters
    ----------
    team_metrics : pl.DataFrame
    ppda_pct     : float  percentile for PPDA cap (default 0.95)
    epr_pct      : float  percentile for EPR cap  (default 0.95)

    Returns
    -------
    (capped_df, cap_info) : pl.DataFrame, dict
    """
    ppda_cap = team_metrics['ppda'].quantile(ppda_pct)
    epr_cap  = team_metrics['epr'].quantile(epr_pct)

    n_ppda = int((team_metrics['ppda'] > ppda_cap).sum())
    n_epr  = int((team_metrics['epr']  > epr_cap).sum())

    capped = team_metrics.with_columns([
        pl.col('ppda').clip(upper_bound=ppda_cap).alias('ppda'),
        pl.col('epr').clip(upper_bound=epr_cap).alias('epr'),
    ])

    cap_info = {
        'ppda_cap': ppda_cap, 'ppda_teams_affected': n_ppda,
        'epr_cap' : epr_cap,  'epr_teams_affected' : n_epr,
    }

    print(f'PPDA cap ({ppda_pct:.0%}): {ppda_cap:.2f}  →  {n_ppda} team(s) affected')
    print(f'EPR  cap ({epr_pct:.0%}): {epr_cap:.2f}  →  {n_epr} team(s) affected')

    return capped, cap_info


def cap_and_scale(team_metrics: pl.DataFrame,
                  ppda_pct: float = 0.95,
                  epr_pct:  float = 0.95) -> tuple[np.ndarray, list, StandardScaler, dict]:
    """
    Full preprocessing pipeline: cap outliers → build feature matrix → scale.

    Order matters — cap FIRST, then scale:
        If we scaled first, the outlier would inflate the std used by
        StandardScaler, compressing all other teams closer together and
        making real tactical differences harder to detect.

    Parameters
    ----------
    team_metrics : pl.DataFrame  (raw, uncapped)
    ppda_pct     : float
    epr_pct      : float

    Returns
    -------
    X       : np.ndarray  shape (n_teams, 10) — scaled feature matrix
    teams   : list[str]   team names in same row order as X
    scaler  : StandardScaler  fitted scaler (needed for centroid back-transform)
    cap_info: dict
    """
    # 1. Cap
    capped, cap_info = cap_outliers(team_metrics, ppda_pct, epr_pct)

    # 2. Build matrix
    X_raw = capped.select(CLUSTER_FEATURES).to_numpy()
    teams = capped['team'].to_list()

    # 3. Scale
    scaler = StandardScaler()
    X      = scaler.fit_transform(X_raw)

    # 4. Verify
    scaled_df = pd.DataFrame(X, columns=CLUSTER_FEATURES)
    print(f'\nFeature matrix shape : {X.shape}')
    print(f'Any NaNs             : {np.isnan(X).any()}')
    print(f'\nPost-scaling means (should be ~0):')
    print(scaled_df.mean().round(3).to_string())
    print(f'\nPost-scaling stds (should be ~1):')
    print(scaled_df.std().round(3).to_string())

    return X, teams, scaler, cap_info
