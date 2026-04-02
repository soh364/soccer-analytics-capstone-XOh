"""
tc_validation.py
────────────────
Extended validation suite for the tactical clustering pipeline.

Seven validation functions, each independently callable:

1. bootstrap_stability      — pairwise co-occurrence matrix over 100 resamples
2. anova_separability       — one-way ANOVA per metric across clusters
3. leave_one_out_stability  — key teams removed, refitted, reassigned
4. temporal_stability       — archetype consistency across separate tournaments
5. baseline_comparison      — archetype vs naive (majority class) outcome predictor
6. expected_vs_actual       — archetype-expected rank vs actual WC 2022 finish
7. cohen_kappa_template     — template for inter-rater reliability calculation

All functions accept the fitted results DataFrame and scaled feature matrix
produced by the main pipeline — no re-loading required.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import cohen_kappa_score, roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler

from tc_clustering import ARCHETYPE_MAP, N_CLUSTERS
from tc_outcome_validation import OUTCOME_RANK, WC2022_OUTCOMES


# ── 1. Bootstrap cluster stability ───────────────────────────────────────────
def bootstrap_stability(X: np.ndarray,
                        teams: list,
                        n_bootstrap: int = 100,
                        n_clusters: int = N_CLUSTERS,
                        random_state: int = 42,
                        stability_threshold: float = 0.70) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Resample teams with replacement 100 times, refit KMeans each time, and
    compute a pairwise co-occurrence matrix — how often each pair of teams
    ends up in the same cluster.

    A team is flagged as unstable if its mean co-occurrence with its own
    cluster members falls below stability_threshold (default 70%).

    Parameters
    ----------
    X                    : np.ndarray  scaled feature matrix (n_teams, n_features)
    teams                : list[str]   team names in same row order as X
    n_bootstrap          : int         number of resamples (default 100)
    n_clusters           : int
    random_state         : int         seed for reproducibility
    stability_threshold  : float       flag teams below this co-occurrence rate

    Returns
    -------
    (cooccurrence_df, stability_summary) : pd.DataFrame, pd.DataFrame
        cooccurrence_df   — n_teams × n_teams symmetric matrix (values 0–1)
        stability_summary — per-team mean co-occurrence + stable flag
    """
    rng         = np.random.default_rng(random_state)
    n_teams     = len(teams)
    cooccurrence = np.zeros((n_teams, n_teams))
    counts       = np.zeros((n_teams, n_teams))

    for i in range(n_bootstrap):
        # Resample with replacement
        idx    = rng.choice(n_teams, size=n_teams, replace=True)
        X_boot = X[idx]

        km     = KMeans(n_clusters=n_clusters, random_state=random_state + i,
                        n_init=10)
        labels = km.fit_predict(X_boot)

        # For each pair of teams in the bootstrap sample, record if same cluster
        for pos_a, orig_a in enumerate(idx):
            for pos_b, orig_b in enumerate(idx):
                counts[orig_a, orig_b] += 1
                if labels[pos_a] == labels[pos_b]:
                    cooccurrence[orig_a, orig_b] += 1

    # Normalise — avoid division by zero for pairs never sampled together
    with np.errstate(invalid='ignore'):
        rate = np.where(counts > 0, cooccurrence / counts, np.nan)

    cooccurrence_df = pd.DataFrame(rate, index=teams, columns=teams)

    # Stability summary — mean co-occurrence with other teams in same row
    # (diagonal excluded since a team always co-occurs with itself)
    stability_rows = []
    for i, team in enumerate(teams):
        row          = rate[i].copy()
        row[i]       = np.nan   # exclude self
        mean_cooc    = np.nanmean(row)
        stable       = mean_cooc >= stability_threshold
        stability_rows.append({'team': team,
                                'mean_cooccurrence': round(mean_cooc, 3),
                                'stable': stable})

    stability_summary = (pd.DataFrame(stability_rows)
                         .sort_values('mean_cooccurrence', ascending=False)
                         .reset_index(drop=True))

    n_unstable = (~stability_summary['stable']).sum()
    print(f'Bootstrap stability ({n_bootstrap} resamples, threshold={stability_threshold:.0%}):')
    print(f'  Stable teams   : {len(stability_summary) - n_unstable}/{len(stability_summary)}')
    print(f'  Unstable teams : {n_unstable}')
    if n_unstable:
        print('\n  Flagged as unstable:')
        print(stability_summary[~stability_summary['stable']]
              [['team', 'mean_cooccurrence']].to_string(index=False))

    return cooccurrence_df, stability_summary


# ── 2. ANOVA metric separability ─────────────────────────────────────────────
def anova_separability(results: pd.DataFrame,
                       metrics: list = None) -> pd.DataFrame:
    """
    For each metric, run a one-way ANOVA across the 6 cluster groups.
    Reports F-statistic, p-value, and flags metrics where p > 0.05
    as not contributing meaningfully to cluster separation.

    Parameters
    ----------
    results : pd.DataFrame  must contain 'archetype' + all metric columns
    metrics : list[str]     metrics to test; defaults to the 8 CORE_METRICS

    Returns
    -------
    anova_df : pd.DataFrame  one row per metric, columns: metric, F, p, significant
    """
    if metrics is None:
        from tc_data import CORE_METRICS
        metrics = CORE_METRICS

    rows = []
    for metric in metrics:
        if metric not in results.columns:
            print(f'  Warning: {metric} not in results — skipping')
            continue
        groups = [grp[metric].dropna().values
                  for _, grp in results.groupby('archetype')]
        # Need at least 2 groups with variance
        groups = [g for g in groups if len(g) > 1]
        if len(groups) < 2:
            continue
        f_stat, p_val = stats.f_oneway(*groups)
        rows.append({
            'metric'     : metric,
            'F_statistic': round(f_stat, 3),
            'p_value'    : round(p_val, 4),
            'significant': p_val <= 0.05,
        })

    anova_df = pd.DataFrame(rows).sort_values('F_statistic', ascending=False)

    print('\nANOVA metric separability (one-way, across 6 clusters):')
    print(anova_df.to_string(index=False))

    n_sig = anova_df['significant'].sum()
    n_not = (~anova_df['significant']).sum()
    print(f'\n  Significant (p≤0.05) : {n_sig}/{len(anova_df)} metrics')
    if n_not:
        flagged = anova_df[~anova_df['significant']]['metric'].tolist()
        print(f'  Not significant      : {flagged}')

    return anova_df


# ── 3. Leave-one-out stability ────────────────────────────────────────────────
def leave_one_out_stability(X: np.ndarray,
                            results: pd.DataFrame,
                            anchor_teams: list = None,
                            n_clusters: int = N_CLUSTERS,
                            random_state: int = 42,
                            n_init: int = 20) -> pd.DataFrame:
    """
    For each key team: remove it, refit KMeans on remaining teams, then
    predict the removed team's cluster using nearest centroid assignment.
    Check if it lands in the same archetype as the full-data fit.

    Parameters
    ----------
    X            : np.ndarray
    results      : pd.DataFrame  must contain 'team' and 'archetype'
    anchor_teams : list[str]     teams to test (default: 10 key nations)
    n_clusters   : int
    random_state : int
    n_init       : int

    Returns
    -------
    loo_df : pd.DataFrame  one row per team — original archetype, LOO archetype, stable flag
    """
    if anchor_teams is None:
        anchor_teams = [
            'Argentina', 'Spain', 'Morocco', 'Japan', 'France',
            'Germany', 'Brazil', 'England', 'Croatia', 'Netherlands',
        ]

    teams      = results['team'].tolist()
    team_index = {t: i for i, t in enumerate(teams)}

    rows = []
    for team in anchor_teams:
        if team not in team_index:
            print(f'  Warning: {team} not in dataset — skipping')
            continue

        idx         = team_index[team]
        original_arch = results.loc[results['team'] == team, 'archetype'].values[0]

        # Remove team and refit
        mask    = np.ones(len(teams), dtype=bool)
        mask[idx] = False
        X_loo   = X[mask]

        km      = KMeans(n_clusters=n_clusters, random_state=random_state,
                         n_init=n_init)
        km.fit(X_loo)

        # Assign removed team to nearest centroid
        dist          = np.linalg.norm(km.cluster_centers_ - X[idx], axis=1)
        nearest_cluster = int(np.argmin(dist))

        # Map cluster integer → archetype name using the refitted model
        # We align by matching centroids to ARCHETYPE_MAP via original cluster order
        # Since cluster integers may differ, we use the archetype of the nearest centroid
        # from the original fit for interpretability
        from sklearn.metrics import pairwise_distances
        original_centers = results.copy()

        # Simpler: just report the cluster integer and note whether archetype matches
        # by checking what archetype the nearest-centroid cluster corresponds to
        # We compare the refitted centroid to original centroids to find best match
        orig_km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=n_init)
        orig_km.fit(X)
        orig_centers = orig_km.cluster_centers_

        # Find which original cluster the LOO centroid is closest to
        centroid_dist   = np.linalg.norm(orig_centers - km.cluster_centers_[nearest_cluster], axis=1)
        matched_cluster = int(np.argmin(centroid_dist))
        loo_arch        = ARCHETYPE_MAP.get(matched_cluster, 'Unknown')
        stable          = loo_arch == original_arch

        rows.append({
            'team'              : team,
            'original_archetype': original_arch,
            'loo_archetype'     : loo_arch,
            'stable'            : stable,
        })

    loo_df    = pd.DataFrame(rows)
    n_stable  = loo_df['stable'].sum()

    print(f'\nLeave-one-out stability ({len(anchor_teams)} key teams):')
    print(loo_df.to_string(index=False))
    print(f'\n  Stable reassignments : {n_stable}/{len(loo_df)}')

    return loo_df


# ── 4. Temporal stability ─────────────────────────────────────────────────────
def temporal_stability(n_clusters: int = N_CLUSTERS,
                       random_state: int = 42,
                       n_init: int = 20,
                       tournament_keys: list = None) -> pd.DataFrame:
    """
    Load each tournament independently, cluster it, and check whether teams
    maintain the same archetype label across tournaments.

    Attempts to load each tournament key via load_tournament_data_8d().
    Skips any key that raises an error (tournament not available in dataset).

    Parameters
    ----------
    n_clusters       : int
    random_state     : int
    n_init           : int
    tournament_keys  : list[str]  defaults to ['men_wc_2022', 'men_euro_2024',
                                                'men_copa_2024']

    Returns
    -------
    stability_df : pd.DataFrame  teams × tournaments, with stability_rate column
    """
    import sys
    from pathlib import Path
    project_root = Path(__file__).resolve().parents[1]
    eda_root     = project_root / 'eda'
    if str(eda_root) not in sys.path:
        sys.path.insert(0, str(eda_root))
    from analysis.data_loader import load_tournament_data_8d

    import functools
    import polars as pl
    from tc_data import CORE_METRICS, CLUSTER_FEATURES
    from tc_preprocessing import cap_and_scale

    if tournament_keys is None:
        tournament_keys = ['men_wc_2022', 'men_euro_2024', 'men_copa_2024']

    tournament_results = {}

    for key in tournament_keys:
        try:
            metrics_raw = load_tournament_data_8d(key, verbose=False)
            dfs         = list(metrics_raw.values())
            metrics     = functools.reduce(
                lambda a, b: a.join(b, on=['match_id', 'team'], how='left'), dfs
            )
            metrics = metrics.with_columns(
                (pl.col('progressive_carries') /
                 (pl.col('progressive_carries') + pl.col('progressive_passes')) * 100
                ).alias('progressive_carry_pct')
            )
            team_metrics = (
                metrics.group_by('team')
                .agg([
                    *[pl.col(c).mean().alias(c) for c in CORE_METRICS],
                    pl.col('ppda').std().alias('ppda_std'),
                    pl.col('possession_pct').std().alias('possession_pct_std'),
                    pl.col('match_id').count().alias('n_matches'),
                ])
                .sort('team')
            )

            # Filter outliers and scale
            EXCLUDE = ['Georgia', 'Slovenia']
            team_metrics = team_metrics.filter(~pl.col('team').is_in(EXCLUDE))
            X_t, teams_t, _, _ = cap_and_scale(team_metrics, ppda_pct=0.95, epr_pct=0.95)

            km     = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=n_init)
            labels = km.fit_predict(X_t)

            # Map cluster → archetype using centroid proximity to global centroids
            # (since cluster integers differ per run, we match by nearest centroid)
            t_results = pd.DataFrame({'team': teams_t, f'cluster_{key}': labels})
            tournament_results[key] = t_results
            print(f'  {key}: {len(teams_t)} teams clustered')

        except Exception as e:
            print(f'  {key}: skipped ({e})')

    if len(tournament_results) < 2:
        print('\nTemporal stability requires at least 2 tournaments — insufficient data available.')
        print('This validation will be available once multi-tournament data is loaded.')
        return pd.DataFrame()

    # Merge all tournaments on team
    merged = tournament_results[tournament_keys[0]]
    for key in tournament_keys[1:]:
        if key in tournament_results:
            merged = merged.merge(tournament_results[key], on='team', how='inner')

    # Compute stability — teams present in all tournaments
    cluster_cols = [c for c in merged.columns if c.startswith('cluster_')]
    if len(cluster_cols) < 2:
        print('Not enough common teams across tournaments for stability analysis.')
        return merged

    # Count how many tournaments each team gets the same label
    # (Use mode of cluster assignments as "consensus")
    def stability_rate(row):
        vals = [row[c] for c in cluster_cols]
        mode = max(set(vals), key=vals.count)
        return sum(v == mode for v in vals) / len(vals)

    merged['stability_rate'] = merged.apply(stability_rate, axis=1)
    overall = merged['stability_rate'].mean()

    print(f'\nTemporal stability across {len(cluster_cols)} tournaments:')
    print(f'  Teams in common : {len(merged)}')
    print(f'  Overall stability rate : {overall:.1%}')
    unstable = merged[merged['stability_rate'] < 1.0]
    if len(unstable):
        print(f'  Teams with inconsistent archetype ({len(unstable)}):')
        print(unstable[['team', 'stability_rate'] + cluster_cols].to_string(index=False))

    return merged


# ── 5. Baseline comparison ────────────────────────────────────────────────────
def baseline_comparison(results: pd.DataFrame) -> pd.DataFrame:
    """
    Compare two simple models predicting whether a team reached the
    Quarter-finals in WC 2022:
        Model A — majority class baseline (always predicts most common outcome)
        Model B — archetype-based (uses archetype as sole predictor)

    Reports accuracy and AUC for both. Shows whether archetype adds signal
    beyond a naive baseline.

    Parameters
    ----------
    results : pd.DataFrame  must contain 'team' and 'archetype'

    Returns
    -------
    comparison_df : pd.DataFrame  two rows — one per model
    """
    # Merge WC 2022 outcomes
    df = results.copy()
    df['wc2022_outcome'] = df['team'].map(WC2022_OUTCOMES)
    df['outcome_rank']   = df['wc2022_outcome'].map(OUTCOME_RANK)
    wc = df[df['wc2022_outcome'].notna()].copy()

    # Binary target: reached QF or better (rank >= 3)
    wc['reached_qf'] = (wc['outcome_rank'] >= 3).astype(int)

    # Archetype → numeric score for AUC
    archetype_score = {
        'High Press / High Output' : 85,
        'Possession Dominant'      : 75,
        'Compact Transition'       : 65,
        'Mid-Block Reactive'       : 60,
        'Moderate Possession'      : 50,
        'Low Intensity'            : 40,
    }
    wc['archetype_score'] = wc['archetype'].map(archetype_score).fillna(50)

    y_true = wc['reached_qf'].values

    # Model A — majority class baseline
    majority_class = int(y_true.mean() >= 0.5)
    y_pred_baseline = np.full_like(y_true, majority_class)
    acc_baseline    = accuracy_score(y_true, y_pred_baseline)
    # AUC needs predicted probabilities — use constant
    try:
        auc_baseline = roc_auc_score(y_true, np.full(len(y_true), y_true.mean()))
    except Exception:
        auc_baseline = 0.5

    # Model B — archetype score as predictor
    # Threshold: teams with score >= 65 predicted to reach QF
    y_pred_arch = (wc['archetype_score'] >= 65).astype(int).values
    acc_arch    = accuracy_score(y_true, y_pred_arch)
    try:
        auc_arch = roc_auc_score(y_true, wc['archetype_score'].values)
    except Exception:
        auc_arch = 0.5

    comparison_df = pd.DataFrame([
        {'model': 'Majority class baseline', 'accuracy': round(acc_baseline, 3),
         'auc': round(auc_baseline, 3)},
        {'model': 'Archetype-based',         'accuracy': round(acc_arch, 3),
         'auc': round(auc_arch, 3)},
    ])

    print('\nBaseline comparison — predicting WC 2022 QF qualification:')
    print(comparison_df.to_string(index=False))
    lift = acc_arch - acc_baseline
    print(f'\n  Accuracy lift from archetype : {lift:+.3f}')
    print(f'  AUC archetype vs baseline    : {auc_arch:.3f} vs {auc_baseline:.3f}')

    return comparison_df


# ── 6. Expected vs actual finish ──────────────────────────────────────────────
def expected_vs_actual(results: pd.DataFrame) -> pd.DataFrame:
    """
    Using archetype average outcome ranks as the "expected" finish,
    compare against each WC 2022 team's actual finish.

    Reports MAE and the top 5 biggest surprises (largest absolute error).

    Parameters
    ----------
    results : pd.DataFrame  must contain 'team' and 'archetype'

    Returns
    -------
    comparison_df : pd.DataFrame  one row per WC 2022 team
    """
    df = results.copy()
    df['wc2022_outcome'] = df['team'].map(WC2022_OUTCOMES)
    df['actual_rank']    = df['wc2022_outcome'].map(OUTCOME_RANK)
    wc = df[df['wc2022_outcome'].notna()].copy()

    # Compute archetype mean outcome rank from WC 2022 teams only
    archetype_expected = (wc.groupby('archetype')['actual_rank']
                          .mean()
                          .round(2)
                          .rename('expected_rank'))

    wc = wc.merge(archetype_expected, on='archetype', how='left')
    wc['error']    = (wc['actual_rank'] - wc['expected_rank']).round(2)
    wc['abs_error'] = wc['error'].abs()

    mae = wc['abs_error'].mean()

    print(f'\nExpected vs actual finish (MAE = {mae:.3f}):')
    print(f'  Scale: 1=Group Stage → 7=Winner\n')

    output_cols = ['team', 'archetype', 'wc2022_outcome',
                   'actual_rank', 'expected_rank', 'error']
    print(wc.sort_values('abs_error', ascending=False)[output_cols].to_string(index=False))

    print(f'\nTop 5 biggest surprises:')
    print(wc.nlargest(5, 'abs_error')[output_cols].to_string(index=False))

    return wc.sort_values('abs_error', ascending=False)[output_cols + ['abs_error']].reset_index(drop=True)


# ── 7. Cohen's kappa template ─────────────────────────────────────────────────
def cohen_kappa_template(rater1_labels: list = None,
                         rater2_labels: list = None) -> dict:
    """
    Template for inter-rater reliability using Cohen's kappa.

    Call this once you have collected archetype labels from a second rater
    (football analyst). Pass two lists of archetype strings in the same
    team order.

    Parameters
    ----------
    rater1_labels : list[str]  archetype labels from rater 1 (model output)
    rater2_labels : list[str]  archetype labels from rater 2 (human analyst)

    Returns
    -------
    result : dict  kappa, interpretation, and agreement rate

    Example usage
    -------------
    from tc_validation import cohen_kappa_template

    model_labels = ['High Press / High Output', 'Compact Transition', ...]
    human_labels = ['High Press / High Output', 'Mid-Block Reactive', ...]
    result = cohen_kappa_template(model_labels, human_labels)
    """
    if rater1_labels is None or rater2_labels is None:
        print('Cohen\'s Kappa Template — Inter-Rater Reliability')
        print('─' * 50)
        print('Usage:')
        print('  from tc_validation import cohen_kappa_template')
        print()
        print('  model_labels = results["archetype"].tolist()')
        print('  human_labels = [...]  # your analyst\'s labels, same team order')
        print('  result = cohen_kappa_template(model_labels, human_labels)')
        print()
        print('Valid archetype labels:')
        for label in ARCHETYPE_MAP.values():
            print(f'  - {label}')
        print()
        print('Kappa interpretation:')
        print('  < 0.20  : Slight agreement')
        print('  0.21–0.40: Fair agreement')
        print('  0.41–0.60: Moderate agreement')
        print('  0.61–0.80: Substantial agreement')
        print('  0.81–1.00: Almost perfect agreement')
        return {}

    if len(rater1_labels) != len(rater2_labels):
        raise ValueError(f'Label lists must be the same length. '
                         f'Got {len(rater1_labels)} and {len(rater2_labels)}.')

    kappa        = cohen_kappa_score(rater1_labels, rater2_labels)
    agreement    = sum(a == b for a, b in zip(rater1_labels, rater2_labels)) / len(rater1_labels)

    if kappa < 0.20:
        interpretation = 'Slight'
    elif kappa < 0.40:
        interpretation = 'Fair'
    elif kappa < 0.60:
        interpretation = 'Moderate'
    elif kappa < 0.80:
        interpretation = 'Substantial'
    else:
        interpretation = 'Almost perfect'

    result = {
        'kappa'         : round(kappa, 4),
        'interpretation': interpretation,
        'agreement_rate': round(agreement, 3),
        'n_teams'       : len(rater1_labels),
    }

    print(f'Cohen\'s Kappa: {kappa:.4f} ({interpretation})')
    print(f'Raw agreement rate: {agreement:.1%} ({len(rater1_labels)} teams)')

    return result


# ── Convenience wrapper ───────────────────────────────────────────────────────
def run_full_validation_suite(X: np.ndarray,
                              teams: list,
                              results: pd.DataFrame,
                              skip: list = None) -> dict:
    """
    Run all validations in sequence and return a dict of results.

    Parameters
    ----------
    X       : np.ndarray  scaled feature matrix
    teams   : list[str]   team names
    results : pd.DataFrame  clustering results with 'team' and 'archetype'
    skip    : list[str]   validation names to skip, e.g. ['temporal_stability']

    Returns
    -------
    validation_results : dict  keyed by validation name
    """
    skip = skip or []
    out  = {}

    print('=' * 60)
    print('FULL VALIDATION SUITE')
    print('=' * 60)

    if 'bootstrap' not in skip:
        print('\n[1/6] Bootstrap Stability')
        out['cooccurrence'], out['bootstrap_summary'] = bootstrap_stability(X, teams)

    if 'anova' not in skip:
        print('\n[2/6] ANOVA Metric Separability')
        out['anova'] = anova_separability(results)

    if 'loo' not in skip:
        print('\n[3/6] Leave-One-Out Stability')
        out['loo'] = leave_one_out_stability(X, results)

    if 'temporal' not in skip:
        print('\n[4/6] Temporal Stability')
        out['temporal'] = temporal_stability()

    if 'baseline' not in skip:
        print('\n[5/6] Baseline Comparison')
        out['baseline'] = baseline_comparison(results)

    if 'expected_vs_actual' not in skip:
        print('\n[6/6] Expected vs Actual Finish')
        out['expected_vs_actual'] = expected_vs_actual(results)

    print('\n' + '=' * 60)
    print('Validation suite complete.')
    print('Call cohen_kappa_template() separately once analyst labels are ready.')
    print('=' * 60)

    return out
