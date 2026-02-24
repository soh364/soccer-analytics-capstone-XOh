"""
player_metrics_eda.py
─────────────────────
Complete EDA for raw player metric files with 12 visualizations.
Expects files loaded from outputs/raw_metrics/recent_club_players/

VISUALIZATIONS:
  Basic (1-4):
    III.1 — Row count bar charts
    III.2 — Player coverage heatmap
    III.3 — Metric distributions (violin)
    III.4 — Minutes played distribution
  
  Existing Advanced (5-7):
    III.7 — Player archetype scatter (xG Chain vs Network, bubble=prog)
    III.8 — Player consistency (mean vs CV)
    III.9 — Cross-season trajectory (2026 WC players)
  
  New Advanced (8-10):
    III.10 — Quality-Minutes scatter (reliability vs talent)
    III.11 — Position-specific quality violins
    III.12 — Specialist matrix (attacking vs defensive 2D heatmap)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from analysis.visualization import save_figure
from matplotlib.lines import Line2D


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR    = Path("../outputs/raw_metrics/recent_club_players")
SEASONS     = ["2021_2022", "2022_2023", "2023_2024"]
FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)

MIN_MINUTES  = 400
MIN_MATCHES  = 5

# Files to load per season, with human-readable labels and the minutes column
FILE_CONFIG = {
    "xg__player__totals.csv": {
        "label":       "Goals − xG",
        "minutes_col": "minutes",
        "metrics": [
            {"metric_col": "goals_minus_xg", "metric_label": "Goals − xG"},
            {"metric_col": "xg",             "metric_label": "xG Volume"},
        ],
    },
    "progression__player__profile.csv": {
        "label":       "Progression Profile",
        "minutes_col": "total_mins",
        "metrics": [
            {"metric_col": "progressive_passes_p90",  "metric_label": "Progressive Passes p90"},
            {"metric_col": "progressive_carries_p90", "metric_label": "Progressive Carries p90"},
        ],
    },
    "advanced__player__xg_chain.csv": {
        "label":       "xG Chain",
        "minutes_col": "minutes_played",
        "metrics": [
            {"metric_col": "xg_chain_per90",       "metric_label": "xG Chain p90"},
            {"metric_col": "team_involvement_pct",  "metric_label": "Team Involvement %"},
        ],
    },
    "advanced__player__xg_buildup.csv": {
        "label":       "xG Buildup",
        "minutes_col": "minutes_played",
        "metrics": [
            {"metric_col": "xg_buildup_per90", "metric_label": "xG Buildup p90"},
        ],
    },
    "advanced__player__packing.csv": {
        "label":       "Packing",
        "minutes_col": None,
        "metrics": [
            {"metric_col": "avg_packing_per_pass", "metric_label": "Avg Packing"},
        ],
    },
    "advanced__player__network_centrality.csv": {
        "label":       "Network Centrality",
        "minutes_col": None,
        "metrics": [
            {"metric_col": "network_involvement_pct", "metric_label": "Network Involvement %"},
        ],
    },
    "defensive__player__pressures.csv": {
        "label":       "Pressures",
        "minutes_col": "minutes_played",
        "metrics": [
            {"metric_col": "pressures_per_90",     "metric_label": "Pressures p90"},
            {"metric_col": "pressure_success_pct", "metric_label": "Pressure Success %"},
        ],
    },
    "defensive__player__profile.csv": {
        "label":       "Defensive Actions",
        "minutes_col": None,
        "metrics": [
            {"metric_col": "total_defensive_actions", "metric_label": "Total Defensive Actions"},
            {"metric_col": "high_turnovers",          "metric_label": "High Turnovers"},
        ],
    },
}

# Colors
SEASON_COLORS = {
    "2021_2022": "#4dabf7",
    "2022_2023": "#2d6a4f",
    "2023_2024": "#f4a261",
}

SEASON_LABELS = {
    "2021_2022": "2021/22",
    "2022_2023": "2022/23",
    "2023_2024": "2023/24",
}

WC_2026_PLAYERS = {
    "Kylian Mbappé Lottin":          "France",
    "Jamal Musiala":                 "Germany",
    "Jude Bellingham":               "England",
    "Bukayo Saka":                   "England",
    "Florian Wirtz":                 "Germany",
    "Lautaro Javier Martínez":       "Argentina",
    "Rodrigo Hernández Cascante":    "Spain",
    "Pedri González López":          "Spain",
    "Vinícius José Paixão de Oliveira Júnior": "Brazil",
    "Phil Foden":                    "England",
    "Christian Pulisic":             "United States",
    "Granit Xhaka":                  "Switzerland",
    "Toni Kroos":                    "Germany",
    "Antoine Griezmann":             "France",
    "Achraf Hakimi Mouh":            "Morocco",
    "Alexis Mac Allister":           "Argentina",
    "João Félix Sequeira":           "Portugal",
    "Cody Mathès Gakpo":             "Netherlands",
    "Lamine Yamal Nasraoui Ebana":   "Spain",
    "William Saliba":                "France"
}

DEFENSIVE_PROFILE_COLORS = {
    "High Presser":        "#e63946",
    "Balanced Defender":   "#4dabf7",
    "Protector":           "#2d6a4f",
    "Limited Progression": "#adb5bd",
}


# ─────────────────────────────────────────────────────────────────────────────
# LOADER
# ─────────────────────────────────────────────────────────────────────────────

def load_all_players() -> dict:
    """
    Returns nested dict: data[season][filename] = DataFrame
    Adds a 'season_folder' column to each df for downstream use.
    """
    data = {}
    for season in SEASONS:
        season_dir = BASE_DIR / season
        data[season] = {}
        for fname in FILE_CONFIG:
            fpath = season_dir / fname
            if fpath.exists():
                df = pd.read_csv(fpath)
                df['season_folder'] = season
                data[season][fname] = df
            else:
                print(f"  [!] Missing: {season}/{fname}")
    return data


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS FOR ADVANCED VISUALIZATIONS
# ─────────────────────────────────────────────────────────────────────────────

def _get_file(player_data: dict, fname: str) -> pd.DataFrame | None:
    """Pull a single file across all scopes and seasons."""
    frames = []
    for scope, seasons in player_data.items():
        for season, files in seasons.items():
            if fname in files:
                df = files[fname]
                if hasattr(df, 'copy'):
                    df = df.copy()
                    df["season_folder"] = season
                    frames.append(df)
                else:
                    print(f"[!] Warning: {fname} in {season} is not a DataFrame, type={type(df)}")
    if not frames:
        print(f"[!] {fname} not found in player_data")
        return None
    return pd.concat(frames, ignore_index=True)


def _get_file_by_season(player_data: dict, fname: str) -> dict:
    """Returns {season: df} for a given file across all scopes."""
    result = {}
    for scope, seasons in player_data.items():
        for season, files in seasons.items():
            if fname in files:
                df = files[fname]
                # Safely copy DataFrame
                if hasattr(df, 'copy'):
                    df = df.copy()
                    df["season_folder"] = season
                    result[season] = df
                else:
                    print(f"[!] Warning: {fname} in {season} is not a DataFrame, type={type(df)}")
    return result


def _latest_season(player_data: dict) -> str:
    """Return the most recent season key present in player_data.
    
    player_data structure: {scope: {season: {fname: df}}}
    Example: {"recent_club_players": {"2021_2022": {...}, "2023_2024": {...}}}
    """
    seasons = []
    for scope, seasons_dict in player_data.items():
        # seasons_dict is {season: {fname: df}}
        seasons.extend(seasons_dict.keys())
    
    if not seasons:
        print("[!] No seasons found in player_data")
        return None
    
    # Sort and return latest
    latest = sorted(set(seasons))[-1]
    print(f"  Latest season found: {latest}")
    return latest


def weighted_mean(df, val_col, weight_col):
    df = df.dropna(subset=[val_col, weight_col])
    if df.empty or df[weight_col].sum() == 0:
        return np.nan
    return np.average(df[val_col], weights=df[weight_col])


def aggregate_player(df, val_col, weight_col="minutes_played",
                     min_mins=MIN_MINUTES, min_matches=MIN_MATCHES):
    """Aggregate match-level df to player level with minutes-weighted mean."""
    if df is None:
        return pd.DataFrame(columns=["player", "team", val_col, "total_mins", "n_matches"])
    if val_col not in df.columns:
        print(f"[!] '{val_col}' not in df. Available: {df.columns.tolist()}")
        return pd.DataFrame(columns=["player", "team", val_col, "total_mins", "n_matches"])

    result = []
    for (player, team), grp in df.groupby(["player", "team"]):
        if len(grp) < min_matches:
            continue
        total = grp[weight_col].sum() if weight_col in grp.columns else len(grp) * 90
        if total < min_mins:
            continue
        val = weighted_mean(grp, val_col, weight_col) if weight_col in grp.columns \
              else grp[val_col].mean()
        result.append({"player": player, "team": team,
                        val_col: val, "total_mins": total,
                        "n_matches": len(grp)})
    return pd.DataFrame(result)


# ─────────────────────────────────────────────────────────────────────────────
# BASIC VISUALIZATIONS (III.1-4)
# ─────────────────────────────────────────────────────────────────────────────

def plot_row_counts(data: dict, figsize=(13, 6)):
    """Grouped bar chart: row counts per file, split by season."""
    records = []
    for season, files in data.items():
        for fname, df in files.items():
            records.append({
                "season":  season,
                "file":    FILE_CONFIG[fname]["label"],
                "rows":    len(df),
            })

    df_counts = pd.DataFrame(records)
    files_ordered = [FILE_CONFIG[f]["label"] for f in FILE_CONFIG]

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#fafafa")

    x      = np.arange(len(files_ordered))
    n      = len(SEASONS)
    width  = 0.25

    for i, season in enumerate(SEASONS):
        subset = df_counts[df_counts["season"] == season]
        counts = [
            subset[subset["file"] == lbl]["rows"].values[0]
            if lbl in subset["file"].values else 0
            for lbl in files_ordered
        ]
        offset = (i - n / 2 + 0.5) * width
        bars = ax.bar(x + offset, counts, width,
                      color=SEASON_COLORS[season], alpha=0.88,
                      edgecolor="white", label=season.replace("_", "/"))
        for bar, val in zip(bars, counts):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 50,
                        f"{val:,}", ha="center", fontsize=8,
                        color="#495057")

    ax.set_xticks(x)
    ax.set_xticklabels(files_ordered, ha="right",
                       fontsize=9, rotation=30)
    ax.set_ylabel("Row Count", fontsize=12)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=10, title="Season", title_fontsize=10)
    ax.grid(axis="y", alpha=0.25, linestyle=":")
    ax.set_title(
        "Raw Record Counts per Metric File and Season",
        fontsize=14, fontweight="bold",
        loc="center", pad=14
    )

    plt.tight_layout()
    save_figure(fig, 'player_metrics_row_counts.png', dpi=180)
    plt.show()

def plot_row_counts(data: dict, figsize=(16, 6)):
    """Grouped bar chart: row counts per metric, split by season."""
    records = []
    for season, files in data.items():
        for fname, df in files.items():
            if fname not in FILE_CONFIG:
                continue
            for metric in FILE_CONFIG[fname]["metrics"]:
                col = metric["metric_col"]
                label = metric["metric_label"]
                # count non-null rows for this specific metric column
                count = df[col].dropna().shape[0] if col in df.columns else 0
                records.append({
                    "season": season,
                    "metric": label,
                    "rows":   count,
                })

    df_counts = pd.DataFrame(records)
    
    # preserve order from FILE_CONFIG
    metrics_ordered = [
        m["metric_label"]
        for fname in FILE_CONFIG
        for m in FILE_CONFIG[fname]["metrics"]
    ]

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#fafafa")

    x     = np.arange(len(metrics_ordered))
    n     = len(SEASONS)
    width = 0.25

    for i, season in enumerate(SEASONS):
        subset = df_counts[df_counts["season"] == season]
        counts = [
            subset[subset["metric"] == lbl]["rows"].values[0]
            if lbl in subset["metric"].values else 0
            for lbl in metrics_ordered
        ]
        offset = (i - n / 2 + 0.5) * width
        bars = ax.bar(x + offset, counts, width,
                      color=SEASON_COLORS[season], alpha=0.88,
                      edgecolor="white", label=season.replace("_", "/"))
        for bar, val in zip(bars, counts):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 50,
                        f"{val:,}", ha="center", fontsize=8,
                        color="#495057")

    ax.set_xticks(x)
    ax.set_xticklabels(metrics_ordered, ha="right",
                       fontsize=9, rotation=30)
    ax.set_ylabel("Row Count", fontsize=12)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=10, title="Season", title_fontsize=10)
    ax.grid(axis="y", alpha=0.25, linestyle=":")
    ax.set_title(
        "Raw Record Counts per Metric and Season",
        fontsize=14, fontweight="bold",
        loc="center", pad=14
    )

    plt.tight_layout()
    save_figure(fig, 'player_metrics_row_counts.png', dpi=180)
    plt.show()

def plot_player_metric_correlations(data: dict, figsize=(12, 10)):
    """Heatmap of inter-metric correlations across all 12 player metrics."""
    
    all_metrics = {}
    
    for fname, cfg in FILE_CONFIG.items():
        for metric in cfg["metrics"]:
            col = metric["metric_col"]
            label = metric["metric_label"]
            frames = []
            for season in SEASONS:
                if fname in data[season] and col in data[season][fname].columns:
                    frames.append(data[season][fname][[col]].rename(columns={col: label}))
            if frames:
                all_metrics[label] = pd.concat(frames, ignore_index=True)[label]
    
    # Build correlation df — no player join, just raw value distributions
    combined = pd.DataFrame(all_metrics)
    corr = combined.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#fafafa")
    
    sns.heatmap(
        corr, ax=ax, mask=mask,
        cmap="RdBu_r", center=0, vmin=-1, vmax=1,
        annot=True, fmt=".2f",
        annot_kws={"size": 8, "fontfamily": "monospace"},
        linewidths=0.5, linecolor="#f0f0f0",
        cbar_kws={"shrink": 0.8, "label": "Pearson r"},
        square=True,
    )
    
    ax.set_title(
        "Validation: Player Metric Correlation Matrix",
        fontsize=14, fontweight="bold", pad=14
    )
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.tick_params(axis="y", rotation=0, labelsize=8)
    
    plt.tight_layout()
    save_figure(fig, "player_metrics_corr.png", dpi=180)
    plt.show()

TRAIT_COLORS = {
    'Mobility_Intensity':  '#f03e3e',  # red
    'Progression':         '#1971c2',  # blue
    'Control':             '#2f9e44',  # green
    'Final_Third_Output':  '#e67700',  # orange
}

# Map metric labels to trait categories
METRIC_LABEL_TRAIT = {
    # Final_Third_Output
    "Goals − xG":          "Final_Third_Output",
    "xG Volume":           "Final_Third_Output",
    "xG Chain p90":        "Final_Third_Output",
    "Team Involvement %":  "Final_Third_Output",
    "xG Buildup p90":      "Final_Third_Output",
    # Progression
    "Progressive Passes p90":  "Progression",
    "Progressive Carries p90": "Progression",
    "Avg Packing":             "Progression",
    # Control
    "Network Involvement %":   "Control",
    # Mobility_Intensity
    "Pressures p90":           "Mobility_Intensity",
    "Pressure Success %":      "Mobility_Intensity",
    "Total Defensive Actions": "Mobility_Intensity",
    "High Turnovers":          "Mobility_Intensity",
}


def plot_metric_distributions_pl(data: dict, figsize=(16, 10)):
    metric_data = {}
    for fname, cfg in FILE_CONFIG.items():
        for metric in cfg["metrics"]:
            col = metric["metric_col"]
            label = metric["metric_label"]
            frames = []
            for season in SEASONS:
                if fname in data[season] and col in data[season][fname].columns:
                    frames.append(data[season][fname][[col, "season_folder"]].dropna())
            if frames:
                metric_data[label] = pd.concat(frames, ignore_index=True)

    fig, axes = plt.subplots(3, 4, figsize=figsize)
    fig.patch.set_facecolor("#ffffff")
    axes = axes.flatten()

    for i, (label, df) in enumerate(metric_data.items()):
        if i >= len(axes):
            break
        ax  = axes[i]
        ax.set_facecolor("#fafafa")
        col = df.columns[0]

        p99  = df[col].quantile(0.99)
        vals = df[col].clip(upper=p99).dropna().tolist()

        # trait colour
        trait = METRIC_LABEL_TRAIT.get(label, "Control")
        color = TRAIT_COLORS[trait]

        vp = ax.violinplot(vals, showmedians=True, showextrema=True)
        for pc in vp["bodies"]:
            pc.set_facecolor(color)
            pc.set_alpha(0.65)
        vp["cmedians"].set_color("#1d3557")
        vp["cmedians"].set_linewidth(2)
        for part in ["cbars", "cmins", "cmaxes"]:
            if part in vp:
                vp[part].set_color(color)
                vp[part].set_linewidth(1.2)

        median = np.median(vals)
        ax.text(1.18, median, f"{median:.2f}",
                va="center", fontsize=8,
                color="#1d3557", fontweight="bold")

        n_outliers = (df[col] > p99).sum()
        if n_outliers > 0:
            ax.text(0.5, 0.97, f"{n_outliers} outliers capped at p99",
                    transform=ax.transAxes, ha="center", fontsize=6.5,
                    color="#e63946")

        ax.set_title(label, fontsize=10, fontweight="bold", color="#1d3557")
        ax.set_xticks([])
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.25, linestyle=":")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    # legend
    legend_handles = [
        plt.matplotlib.patches.Patch(facecolor=col, alpha=0.75,
                                     label=trait.replace("_", " "))
        for trait, col in TRAIT_COLORS.items()
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=4,
               fontsize=10, frameon=False, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(
        "Raw Metric Distributions (p99 capped)",
        fontsize=14, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    save_figure(fig, "metric_distributions.png", dpi=180)
    plt.show()

def plot_minutes_distribution(data: dict, figsize=(13, 6)):
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.patch.set_facecolor("#ffffff")

    files_with_mins = {
        fname: cfg
        for fname, cfg in FILE_CONFIG.items()
        if cfg["minutes_col"] is not None
    }

    # Take the first 3 relevant files
    selected = list(files_with_mins.items())[:3]

    for i, (ax, (fname, cfg)) in enumerate(zip(axes, selected)):
        ax.set_facecolor("#fafafa")
        col = cfg["minutes_col"]

        # Aggregate minutes across all seasons
        frames = []
        for season in SEASONS:
            if fname in data[season] and col in data[season][fname].columns:
                frames.append(data[season][fname][[col]])

        if not frames:
            ax.set_visible(False)
            continue

        mins = pd.concat(frames)[col].dropna()
        mins = mins[mins > 0]

        # Plotting the distribution
        ax.hist(mins, bins=35, color="#4dabf7", alpha=0.8,
                edgecolor="white", linewidth=0.5)

        # Threshold line
        ax.axvline(MIN_MINUTES, color="#e63946", linestyle="--",
                   linewidth=1.5, label=f"Threshold ({MIN_MINUTES}m)")

        # Percentage annotation
        pct_below = (mins < MIN_MINUTES).mean() * 100
        ax.text(MIN_MINUTES + 25, ax.get_ylim()[1] * 0.80,
                f"{pct_below:.0f}% < {MIN_MINUTES}m",
                fontsize=9, color="#e63946", fontweight="bold")

        # Styling
        ax.set_xlabel("Minutes Played", fontsize=10)
        ax.set_ylabel("Player Count" if i == 0 else "", fontsize=10)
        ax.set_title(cfg["label"], fontsize=12, fontweight="bold", pad=10)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=8, loc="upper right", frameon=False)
        ax.grid(axis="y", alpha=0.2, linestyle="--")

    plt.subplots_adjust(left=0.1, right=0.9, top=0.82, bottom=0.15, wspace=0.25)

    # Place the title at exactly 0.5 (the middle of the image)
    # This will now align with the midpoint of your adjusted plots.
    fig.text(
        0.5, 0.94, 
        "Minutes Played Distribution", 
        fontsize=14, 
        fontweight="bold", 
        ha='center', 
        va='top'
    )

    save_figure(fig, 'player_mins_dis.png', dpi=180)
    plt.show()

# ─────────────────────────────────────────────────────────────────────────────
# EXISTING ADVANCED VISUALIZATIONS (III.7-9)
# ─────────────────────────────────────────────────────────────────────────────

def plot_player_archetype_scatter(player_data: dict, figures_dir: Path = FIGURES_DIR,
                                   figsize=(13, 9)):
    """III.7 — Player archetype scatter: xG Chain vs Network Involvement"""
    figures_dir.mkdir(exist_ok=True)

    season = _latest_season(player_data)
    if season is None:
        return

    def get_season(fname):
        return _get_file_by_season(player_data, fname).get(season)

    # Get xG Chain
    chain_df = get_season("advanced__player__xg_chain.csv")
    if chain_df is None:
        return
    
    chain = aggregate_player(chain_df, "xg_chain_per90", "minutes_played",
                              min_mins=MIN_MINUTES, min_matches=MIN_MATCHES)
    if chain.empty:
        return
    
    chain = chain[["player", "team", "xg_chain_per90"]]

    # Get Network
    net_df = get_season("advanced__player__network_centrality.csv")
    if net_df is None:
        return
    
    if "minutes_played" not in net_df.columns:
        net_df["minutes_played"] = 90
    
    network = aggregate_player(net_df, "network_involvement_pct", "minutes_played",
                                min_mins=MIN_MINUTES, min_matches=MIN_MATCHES)
    if network.empty:
        return
    
    network = network[["player", "team", "network_involvement_pct"]]

    # Get Progression (optional)
    prog_df = get_season("progression__player__profile.csv")
    if prog_df is not None:
        prog = (prog_df.groupby(["player", "team"])
                    .agg(prog_p90=("total_progressive_actions_p90", "mean"),
                         total_mins=("total_mins", "sum"))
                    .reset_index()
                    .query("total_mins >= @MIN_MINUTES")
                   [["player", "team", "prog_p90"]])
    else:
        prog = pd.DataFrame(columns=["player", "team", "prog_p90"])

    # Merge - aggregate by player only to avoid duplicates
    df = (chain.groupby("player")["xg_chain_per90"].mean().reset_index()
          .merge(network.groupby("player")["network_involvement_pct"].mean().reset_index(), 
                 on="player", how="inner")
          .merge(prog.groupby("player")["prog_p90"].mean().reset_index() if not prog.empty else pd.DataFrame(columns=["player", "prog_p90"]),
                 on="player", how="left"))

    if df.empty:
        return
    
    # Fill missing
    df["prog_p90"] = df["prog_p90"].fillna(df["prog_p90"].median() if not df["prog_p90"].isna().all() else 5.0)
    
    # Cap outliers
    df["xg_chain_per90"] = df["xg_chain_per90"].clip(upper=df["xg_chain_per90"].quantile(0.97))
    df["network_involvement_pct"] = df["network_involvement_pct"].clip(upper=df["network_involvement_pct"].quantile(0.97))
    
    # Bubble size
    df["bubble"] = ((df["prog_p90"] - df["prog_p90"].min()) /
                    (df["prog_p90"].max() - df["prog_p90"].min()) * 180 + 20)

    # Color by xG Chain quartiles
    df["color_group"] = pd.qcut(df["xg_chain_per90"], q=4, 
                                labels=["Low Attack", "Mid-Low Attack", "Mid-High Attack", "High Attack"], 
                                duplicates='drop')
    color_map = {
        "Low Attack": "#dee2e6",
        "Mid-Low Attack": "#4dabf7", 
        "Mid-High Attack": "#f4a261",
        "High Attack": "#e63946"
    }

    # Median lines
    xm = df["xg_chain_per90"].median()
    ym = df["network_involvement_pct"].median()

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#fafafa")

    # Scatter plot
    for group, grp in df.groupby("color_group"):
        color = color_map.get(group, "#adb5bd")
        ax.scatter(grp["xg_chain_per90"], grp["network_involvement_pct"],
                   s=grp["bubble"], c=color, alpha=0.65,
                   edgecolors="white", linewidths=0.5, zorder=3)

    # Median lines
    ax.axvline(xm, color="#868e96", linestyle=":", lw=1.2, alpha=0.7)
    ax.axhline(ym, color="#868e96", linestyle=":", lw=1.2, alpha=0.7)

    # Quadrant labels
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    kw = dict(fontsize=8.5, fontstyle="italic", alpha=0.7)
    
    ax.text(xm + (xmax-xm)*0.05, ym + (ymax-ym)*0.6,
            "CREATIVE\nHUBS", color="#e63946", **kw)
    ax.text(xm + (xmax-xm)*0.05, ymin + (ym-ymin)*0.05,
            "PURE\nATTACKERS", color="#f4a261", **kw)
    ax.text(xmin + (xm-xmin)*0.05, ym + (ymax-ym)*0.6,
            "DEEP\nPLAYMAKERS", color="#4dabf7", **kw)
    ax.text(xmin + (xm-xmin)*0.05, ymin + (ym-ymin)*0.05,
            "DEFENSIVE\nSPECIALISTS", color="#dee2e6", **kw)

    # Label WC 2026 players
    wc_df = df[df["player"].isin(WC_2026_PLAYERS.keys())]
    
    for _, row in wc_df.iterrows():
        short = row["player"].split()[0] + " " + row["player"].split()[1]
        
        # Get color for this player's group
        player_color = color_map.get(row["color_group"], "#adb5bd")
        
        # Highlight point
        ax.scatter(row["xg_chain_per90"], row["network_involvement_pct"], 
                   c=player_color, s=100, edgecolors="#343a40", 
                   linewidths=1.5, zorder=5)
        
        # Add label
        ax.annotate(short, (row["xg_chain_per90"], row["network_involvement_pct"]),
                    xytext=(5, 4), textcoords="offset points",
                    fontsize=7, 
                    color="#343a40", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=player_color, alpha=0.8))

    # Legend
    handles = [
        mpatches.Patch(color=c, label=g, alpha=0.75)
        for g, c in color_map.items()
    ]
    ax.legend(handles=handles, fontsize=8, frameon=False, 
              title="Attack Level", title_fontsize=8,
              loc="upper right")

    ax.set_xlabel("xG Chain p90  (attacking involvement)", fontsize=10)
    ax.set_ylabel("Network Involvement %  (structural centrality)", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.2, linestyle=":", zorder=0)
    ax.set_title(
        "Player Archetype Scatter Plot\n"
        f"xG Chain × Network Involvement  |  {SEASON_LABELS.get(season, season)} (minimum {MIN_MINUTES} mins)",
        fontsize=14, fontweight="bold", 
        loc="center", pad=14
    )

    

    plt.tight_layout()
    save_figure(fig, 'player_archetype_scatter.png', dpi=180)
    plt.show()


def plot_player_consistency(player_data: dict, figures_dir: Path = FIGURES_DIR,
                            figsize=(12, 9)):
    """III.8 — Player consistency: mean performance vs coefficient of variation"""
    figures_dir.mkdir(exist_ok=True)
    season = _latest_season(player_data)
    fname  = "advanced__player__xg_chain.csv"
    df     = _get_file_by_season(player_data, fname).get(season)

    if df is None:
        print(f"[!] {fname} not found for {season}")
        return

    valid = (df.groupby(["player", "team"])
            .filter(lambda x:
                len(x) >= MIN_MATCHES and
                x["minutes_played"].sum() >= MIN_MINUTES))

    # Aggregate by player only (across all teams) to avoid duplicates
    stats = (valid.groupby("player")["xg_chain_per90"]
                .agg(mean="mean", std="std", n="count")
                .reset_index()
                .dropna())

    stats["cv"] = stats["std"] / stats["mean"].replace(0, np.nan)
    stats        = stats.dropna(subset=["cv"])

    stats["mean"] = stats["mean"].clip(upper=stats["mean"].quantile(0.97))
    stats["cv"]   = stats["cv"].clip(upper=stats["cv"].quantile(0.97))

    xm = stats["mean"].median()
    ym = stats["cv"].median()

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#fafafa")

    def quadrant_color(row):
        if row["mean"] >= xm and row["cv"] <= ym:  return "#2d6a4f"
        if row["mean"] >= xm and row["cv"] > ym:   return "#f4a261"
        if row["mean"] < xm  and row["cv"] <= ym:  return "#4dabf7"
        return "#dee2e6"

    stats["color"] = stats.apply(quadrant_color, axis=1)

    ax.scatter(stats["mean"], stats["cv"],
               c=stats["color"], s=40, alpha=0.65,
               edgecolors="white", linewidths=0.5, zorder=3)

    ax.axvline(xm, color="#868e96", linestyle=":", lw=1.2, alpha=0.7)
    ax.axhline(ym, color="#868e96", linestyle=":", lw=1.2, alpha=0.7)

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    kw = dict(fontsize=8.5, fontstyle="italic", alpha=0.7)
    ax.text(xm + (xmax-xm)*0.05, ymin + (ym-ymin)*0.05,
            "ELITE\nCONSISTENT", color="#2d6a4f", **kw)
    ax.text(xm + (xmax-xm)*0.05, ym + (ymax-ym)*0.6,
            "ELITE\nVOLATILE",   color="#f4a261", **kw)
    ax.text(xmin + (xm-xmin)*0.05, ymin + (ym-ymin)*0.05,
            "AVERAGE\nCONSISTENT", color="#4dabf7", **kw)
    ax.text(xmin + (xm-xmin)*0.05, ym + (ymax-ym)*0.6,
            "AVERAGE\nVOLATILE",   color="#adb5bd", **kw)

    from adjustText import adjust_text

    # Highlight WC players
    wc_df = stats[stats["player"].isin(WC_2026_PLAYERS.keys())]

    texts = []
    for _, row in wc_df.iterrows():
        short = row["player"].split()[0] + " " + row["player"].split()[1]
        
        # Highlight the point
        ax.scatter(row["mean"], row["cv"], c=row["color"],
                s=100, edgecolors="#343a40", linewidths=1.5,
                zorder=5)
        
        # Create text annotation
        txt = ax.annotate(short, (row["mean"], row["cv"]),
                        fontsize=7, 
                        color="#343a40", fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=row["color"], alpha=0.8))
        texts.append(txt)

    # Auto-adjust to avoid overlap
    adjust_text(texts, 
                arrowprops=dict(arrowstyle='-', color='gray', lw=0.5, alpha=0.5),
                expand_points=(1.5, 1.5),
                force_points=(0.5, 0.5))

    handles = [
        mpatches.Patch(color="#2d6a4f", label="Elite Consistent"),
        mpatches.Patch(color="#f4a261", label="Elite Volatile"),
        mpatches.Patch(color="#4dabf7", label="Average Consistent"),
        mpatches.Patch(color="#dee2e6", label="Average Volatile"),
    ]
    ax.legend(handles=handles, fontsize=8, frameon=False, loc="upper right")

    ax.set_xlabel("Mean xG Chain p90  (attacking quality)", fontsize=10)
    ax.set_ylabel("Coefficient of Variation  (higher = less consistent)", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.2, linestyle=":", zorder=0)
    ax.set_title(
        "Player Consistency: Quality vs Reliability\n"
        f"xG Chain mean vs match-to-match variance  |  "
        f"{SEASON_LABELS.get(season, season)} (min {MIN_MATCHES} matches)",
        fontsize=14, fontweight="bold", 
        loc="center", pad=14
    )

    plt.tight_layout()
    save_figure(fig, 'player_consistency.png', dpi=180)
    plt.show()


def plot_player_trajectory(player_data: dict, figures_dir: Path = FIGURES_DIR,
                            figsize=(14, 10)):
    """III.9 — Cross-season trajectory for key 2026 WC players"""
    figures_dir.mkdir(exist_ok=True)
    
    METRIC_FILES = {
        "advanced__player__xg_chain.csv":          ("xg_chain_per90",              "minutes_played"),
        "advanced__player__network_centrality.csv": ("network_involvement_pct",     "minutes_played"),
        "progression__player__profile.csv":         ("total_progressive_actions_p90","total_mins"),
    }

    all_seasons = sorted({s for scope in player_data.values() for s in scope.keys()})

    records = []
    for season in all_seasons:
        season_dfs = []
        for fname, (col, mins_col) in METRIC_FILES.items():
            df = _get_file_by_season(player_data, fname).get(season)
            if df is None:
                continue
            if mins_col not in df.columns:
                df[mins_col] = 90

            agg = aggregate_player(df, col, mins_col,
                                   min_mins=MIN_MINUTES, min_matches=1)
            if agg.empty:
                continue

            mn, mx = agg[col].min(), agg[col].max()
            agg[f"{col}_norm"] = (agg[col] - mn) / (mx - mn) if mx != mn else 0.5
            season_dfs.append(agg[["player", "team", f"{col}_norm"]])

        if not season_dfs:
            continue

        merged = season_dfs[0]
        for sdf in season_dfs[1:]:
            merged = merged.merge(sdf, on=["player", "team"], how="outer")

        norm_cols = [c for c in merged.columns if c.endswith("_norm")]
        merged["composite"] = merged[norm_cols].mean(axis=1)
        merged["season"]    = season
        records.append(merged[["player", "team", "composite", "season"]])

    if not records:
        print("[!] No trajectory data found")
        return

    traj = pd.concat(records, ignore_index=True)

    wc_traj = traj[traj["player"].isin(WC_2026_PLAYERS.keys())]
    present = wc_traj.groupby("player")["season"].count()
    valid_players = present[present >= 2].index
    wc_traj = wc_traj[wc_traj["player"].isin(valid_players)]

    if wc_traj.empty:
        print("[!] No WC 2026 players found in trajectory data")
        return

    def classify(grp):
        vals = grp.set_index("season")["composite"]
        if len(vals) < 2:
            return "Insufficient data"
        first = vals.iloc[0]
        last  = vals.iloc[-1]
        delta = last - first
        if delta > 0.1:   return "Ascending"
        if delta < -0.1:  return "Declining"
        if vals.std() > 0.15: return "Inconsistent"
        return "Peak / Stable"

    traj_class = (wc_traj.groupby("player")
                         .apply(classify)
                         .reset_index()
                         .rename(columns={0: "trajectory"}))
    wc_traj = wc_traj.merge(traj_class, on="player")

    TRAJ_COLORS = {
        "Ascending":        "#2d6a4f",
        "Peak / Stable":    "#4dabf7",
        "Declining":        "#e63946",
        "Inconsistent":     "#f4a261",
        "Insufficient data":"#adb5bd",
    }

    x_vals   = list(range(len(all_seasons)))
    x_labels = [SEASON_LABELS.get(s, s) for s in all_seasons]

    # Select top 12 WC players only
    top_players = [
        "Kylian Mbappé Lottin", "Jude Bellingham",
        "Lamine Yamal Nasraoui Ebana", "Rodrigo Hernández Cascante",
        "Federico Santiago Valverde Dipetta", "Granit Xhaka", "Lautaro Javier Martínez",
        "Cody Mathès Gakpo", "Bukayo Saka", 
    ]
    
    wc_traj = wc_traj[wc_traj["player"].isin(top_players)]
    
    # Create 3x4 grid
    fig, axes = plt.subplots(3, 4, figsize=figsize)
    fig.patch.set_facecolor("#ffffff")
    axes = axes.flatten()

    for idx, (player, grp) in enumerate(wc_traj.groupby("player")):
        if idx >= 12:
            break
            
        ax = axes[idx]
        ax.set_facecolor("#fafafa")
        
        grp = grp.sort_values("season")
        traj_type = grp["trajectory"].iloc[0]
        color = TRAJ_COLORS.get(traj_type, "#adb5bd")
        
        xs = [all_seasons.index(s) for s in grp["season"] if s in all_seasons]
        ys = grp["composite"].tolist()
        
        # Reference line at median
        ax.axhline(0.5, color='gray', linestyle=':', linewidth=0.8, alpha=0.3, zorder=1)
        
        # Shade under curve
        ax.fill_between(xs, 0, ys, color=color, alpha=0.15, zorder=2)
        
        # Plot line
        ax.plot(xs, ys, color=color, linewidth=2.5, marker='o', 
                markersize=10, markeredgecolor='white', markeredgewidth=1,
                alpha=0.85, zorder=3)
        
        # Format
        ax.set_xticks(x_vals)
        ax.set_xticklabels(x_labels, fontsize=7.5)
        ax.set_ylim(0, 1.05)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(['0', '.25', '.5', '.75', '1'], fontsize=7)
        
        # Y-label only on left column
        if idx % 4 == 0:
            ax.set_ylabel("Score", fontsize=8)
        
        ax.spines[["top", "right"]].set_visible(False)
        
        # Title = player name
        short = player.split()[0] + " " + player.split()[1]
        ax.set_title(short, fontsize=9.5, fontweight="bold", 
                    color=color, pad=8)
        
        # Add trajectory label (smaller, bottom corner)
        ax.text(0.98, 0.05, traj_type, transform=ax.transAxes,
                fontsize=6.5, ha='right', va='bottom', color=color,
                bbox=dict(boxstyle='round,pad=0.25', fc='white', ec=color, alpha=0.8, linewidth=0.8))

    # Hide unused subplots
    for idx in range(len(wc_traj.groupby("player")), 12):
        axes[idx].set_visible(False)

    fig.suptitle(
        "III.9 — Player Development Trajectories (2021-2024)\n"
        "Normalized composite score: xG Chain + Network + Progression",
        fontsize=13, fontweight="bold", y=0.995
    )

    plt.tight_layout()
    save_figure(fig, 'player_trajectory.png', dpi=180)
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# NEW ADVANCED VISUALIZATIONS (III.10-12)
# ─────────────────────────────────────────────────────────────────────────────

def plot_quality_minutes_scatter(player_data: dict, figures_dir: Path = FIGURES_DIR,
                                 figsize=(14, 9)):
    """III.10 — Quality-Minutes scatter: reliability vs talent"""
    figures_dir.mkdir(exist_ok=True)
    
    season = _latest_season(player_data)
    print(f"  Using season: {season}")
    
    chain_df = _get_file_by_season(player_data, "advanced__player__xg_chain.csv").get(season)
    
    if chain_df is None:
        print("[!] xG Chain data not found - skipping quality-minutes scatter")
        return
    
    print(f"  xG Chain: {len(chain_df)} records")
    
    # Aggregate to player level (NO minimum filter for this viz - that's the point!)
    player_stats = (chain_df.groupby(["player", "team"])
                    .agg(
                        xg_chain_per90=("xg_chain_per90", "mean"),
                        total_minutes=("minutes_played", "sum"),
                        matches=("match_id", "nunique")
                    )
                    .reset_index())
    
    print(f"  Player stats: {len(player_stats)} players")
    
    if player_stats.empty:
        print("[!] No player stats generated - skipping")
        return
    # Cap extreme outliers for visualization
    player_stats["xg_chain_per90"] = player_stats["xg_chain_per90"].clip(
        upper=player_stats["xg_chain_per90"].quantile(0.98)  # Cap at 98th percentile
    )

    # Also cap extreme minutes (those 30k minute players)
    player_stats["total_minutes"] = player_stats["total_minutes"].clip(
        upper=player_stats["total_minutes"].quantile(0.99)  # Cap at 99th percentile
    )
    
    median_quality = player_stats["xg_chain_per90"].median()
    
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#fafafa")
    
    reliable = player_stats["total_minutes"] >= 270
    colors = ["#e63946" if not r else "#2d6a4f" for r in reliable]
    
    ax.scatter(
        player_stats["total_minutes"],
        player_stats["xg_chain_per90"],
        s=player_stats["matches"] * 10 + 20,
        c=colors,
        alpha=0.5,
        edgecolors="white",
        linewidths=0.5,
        zorder=3
    )
    
    ax.axvline(270, color="#868e96", linestyle="--", linewidth=2, 
               label="Reliability Threshold (270 mins)", zorder=2)
    ax.axhline(median_quality, color="#868e96", linestyle=":", linewidth=1.5,
               label=f"Median Quality ({median_quality:.2f})", zorder=2)
    
    xmin, xmax = 0, player_stats["total_minutes"].max() * 1.05
    ymin, ymax = ax.get_ylim()
    
    kw = dict(fontsize=9, fontstyle="italic", 
              alpha=0.6, fontweight="bold")

    ax.text(xmax * 0.6, ymin + (median_quality - ymin) * 0.15, "WORKHORSES", 
            color="#4dabf7", ha="center", **kw)
    ax.text(150, ymin + (median_quality - ymin) * 0.15, "BENCH WARMERS", 
            color="#adb5bd", ha="center", **kw)
    
    handles = [
        mpatches.Patch(color="#2d6a4f", label=f"Reliable (≥270 mins, n={reliable.sum()})"),
        mpatches.Patch(color="#e63946", label=f"Small Sample (<270 mins, n={(~reliable).sum()})"),
    ]
    ax.legend(handles=handles, fontsize=9, frameon=False, loc="upper right")
    
    ax.set_xlabel("Total Minutes Played", fontsize=11)
    ax.set_ylabel("xG Chain per-90 (Quality Score)", fontsize=11)
    ax.set_xlim(0, 10000)  # Most players under 10k minutes per season
    ax.set_ylim(0, 1.5)    # Cap quality score display
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.2, linestyle=":", zorder=0)
    
    ax.set_title(
        "Quality vs Playing Time: The Reliability Challenge "
        f"{SEASON_LABELS.get(season, season)}",
        fontsize=14, fontweight="bold",
        loc="center", pad=14
    )
    
    plt.tight_layout()
    save_figure(fig, 'quality_minutes_scatter.png', dpi=180)
    plt.show()


def plot_position_quality_violins(player_data: dict, figures_dir: Path = FIGURES_DIR,
                                  figsize=(15, 6)):
    """III.11 — Position-specific quality distributions"""
    figures_dir.mkdir(exist_ok=True)
    
    season = _latest_season(player_data)
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.patch.set_facecolor("#ffffff")
    
    metrics = [
        ("xg__player__totals.csv", "goals_minus_xg", "Forwards: Finishing"),
        ("advanced__player__network_centrality.csv", "network_involvement_pct", "Midfielders: Network %"),
        ("defensive__player__profile.csv", "total_defensive_actions", "Defenders: Def Actions")
    ]
    
    for ax, (fname, metric, title) in zip(axes, metrics):
        df = _get_file_by_season(player_data, fname).get(season)
        
        if df is None or metric not in df.columns:
            ax.set_visible(False)
            continue
        
        ax.set_facecolor("#fafafa")
        
        if "minutes_played" in df.columns:
            df = df[df["minutes_played"] >= 270]
        elif "total_mins" in df.columns:
            df = df[df["total_mins"] >= 270]
        
        vals = df[metric].dropna()
        p99 = vals.quantile(0.99)
        p1 = vals.quantile(0.01)
        vals_clipped = vals.clip(lower=p1, upper=p99)
        
        vp = ax.violinplot([vals_clipped], showmedians=True, showextrema=True)
        for pc in vp["bodies"]:
            pc.set_facecolor("#4dabf7")
            pc.set_alpha(0.65)
        vp["cmedians"].set_color("#1d3557")
        vp["cmedians"].set_linewidth(2)
        
        p75 = vals.quantile(0.75)
        p50 = vals.quantile(0.50)
        p25 = vals.quantile(0.25)
        
        for pval, label in [(p75, "P75"), (p50, "P50"), (p25, "P25")]:
            ax.axhline(pval, color="#868e96", linestyle=":", linewidth=1, alpha=0.5)
            ax.text(1.15, pval, f"{label}: {pval:.1f}", fontsize=8, va="center",
                   color="#495057")
        
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xticks([])
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.2, linestyle=":")
    
    fig.suptitle(
        "III.11 — Position-Specific Quality Distributions\n"
        f"What makes a player 'good' depends on position  |  {SEASON_LABELS.get(season, season)}, min 270 mins",
        fontsize=12, fontweight="bold",
        y=1.02
    )
    
    plt.tight_layout()
    save_figure(fig, 'position_quality_violins.png', dpi=180)
    plt.show()


def plot_specialist_matrix(player_data: dict, figures_dir: Path = FIGURES_DIR,
                           figsize=(12, 10)):
    """III.12 — Specialist matrix: attacking vs defensive quality 2D heatmap"""
    figures_dir.mkdir(exist_ok=True)
    
    season = _latest_season(player_data)
    
    chain_df = _get_file_by_season(player_data, "advanced__player__xg_chain.csv").get(season)
    prog_df = _get_file_by_season(player_data, "progression__player__profile.csv").get(season)
    def_df = _get_file_by_season(player_data, "defensive__player__profile.csv").get(season)
    press_df = _get_file_by_season(player_data, "defensive__player__pressures.csv").get(season)
    
    if chain_df is None or def_df is None:
        print("[!] Required data not found")
        return
    
    attack = aggregate_player(chain_df, "xg_chain_per90", "minutes_played")
    prog = aggregate_player(prog_df, "total_progressive_actions_p90", "total_mins") if prog_df is not None else None
    
    if prog is not None:
        attack = attack.merge(prog[["player", "team", "total_progressive_actions_p90"]], 
                             on=["player", "team"], how="left")
        attack["attacking_quality"] = (
            0.6 * attack["xg_chain_per90"].fillna(0) +
            0.4 * attack["total_progressive_actions_p90"].fillna(0)
        )
    else:
        attack["attacking_quality"] = attack["xg_chain_per90"]
    
    defense = aggregate_player(def_df, "total_defensive_actions", "minutes_played")
    press = aggregate_player(press_df, "pressures_per_90", "minutes_played") if press_df is not None else None
    
    if press is not None:
        defense = defense.merge(press[["player", "team", "pressures_per_90"]],
                               on=["player", "team"], how="left")
        defense["defensive_quality"] = (
            0.5 * defense["total_defensive_actions"].fillna(0) +
            0.5 * defense["pressures_per_90"].fillna(0)
        )
    else:
        defense["defensive_quality"] = defense["total_defensive_actions"]
    
    df = attack[["player", "team", "attacking_quality"]].merge(
        defense[["player", "team", "defensive_quality"]],
        on=["player", "team"],
        how="inner"
    )
    
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 100))
    df[["attacking_quality", "defensive_quality"]] = scaler.fit_transform(
        df[["attacking_quality", "defensive_quality"]]
    )
    
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#fafafa")
    
    h = ax.hist2d(df["attacking_quality"], df["defensive_quality"],
                  bins=25, cmap="YlOrRd", cmin=1, alpha=0.8)
    
    plt.colorbar(h[3], ax=ax, label="Player Density")
    
    ax.plot([0, 100], [0, 100], color="white", linewidth=2.5, 
            linestyle="--", alpha=0.9, label="Balanced Line")
    
    kw = dict(fontsize=11, fontweight="bold", alpha=0.8)
    ax.text(75, 15, "⚡ PURE ATTACKERS", color="#e63946", ha="center", **kw)
    ax.text(15, 75, "🛡️ DEFENSIVE\nSPECIALISTS", color="#2d6a4f", ha="center", **kw)
    ax.text(75, 75, "🌟 COMPLETE\n(Rare!)", color="#4dabf7", ha="center", **kw)
    ax.text(15, 15, "⚪ LIMITED", color="#adb5bd", ha="center", **kw)
    
    top_attackers = df.nlargest(3, "attacking_quality")
    top_defenders = df.nlargest(3, "defensive_quality")
    df["balance_score"] = abs(df["attacking_quality"] - df["defensive_quality"])
    top_balanced = df.nsmallest(3, "balance_score")
    
    for _, player in top_attackers.iterrows():
        short = player["player"].split()[-1]
        ax.annotate(short, (player["attacking_quality"], player["defensive_quality"]),
                   xytext=(5, 5), textcoords="offset points",
                   fontsize=8, color="#e63946", fontweight="bold",
                   bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#e63946", alpha=0.7))
    
    for _, player in top_defenders.iterrows():
        short = player["player"].split()[-1]
        ax.annotate(short, (player["attacking_quality"], player["defensive_quality"]),
                   xytext=(5, 5), textcoords="offset points",
                   fontsize=8, color="#2d6a4f", fontweight="bold",
                   bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#2d6a4f", alpha=0.7))
    
    for _, player in top_balanced.iterrows():
        short = player["player"].split()[-1]
        ax.annotate(short, (player["attacking_quality"], player["defensive_quality"]),
                   xytext=(5, 5), textcoords="offset points",
                   fontsize=8, color="#4dabf7", fontweight="bold",
                   bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#4dabf7", alpha=0.7))
    
    ax.set_xlabel("Attacking Quality (0-100)", fontsize=11)
    ax.set_ylabel("Defensive Quality (0-100)", fontsize=11)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.2, linestyle=":", zorder=0)
    ax.legend(fontsize=9, frameon=False, loc="upper left")
    
    ax.set_title(
        "III.12 — The Specialist Matrix: Attacking vs Defensive Quality\n"
        f"No complete players exist — everyone specializes  |  {SEASON_LABELS.get(season, season)}, min {MIN_MINUTES} mins",
        fontsize=12, fontweight="bold",
        loc="left", pad=14
    )
    
    plt.tight_layout()
    save_figure(fig, 'specialist_matrix.png', dpi=180)
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN EXECUTION
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """Run all player metrics EDA visualizations."""
    
    print("="*70)
    print("PLAYER METRICS EDA - COMPREHENSIVE ANALYSIS")
    print("="*70)
    print()
    
    print("[1/10] Loading data...")
    data = load_all()
    player_data = {"recent_club_players": data}
    print(f"       Loaded {len(SEASONS)} seasons")
    print()
    
    print("[2/10] Creating row count bar charts...")
    plot_row_counts(data)
    
    print("[3/10] Creating coverage heatmap...")
    plot_coverage_heatmap(data)
    
    print("[4/10] Creating metric distributions...")
    plot_metric_distributions_pl(data)
    
    print("[5/10] Creating minutes distribution...")
    plot_minutes_distribution(data)
    
    print("[6/10] Creating player archetype scatter...")
    plot_player_archetype_scatter(player_data, FIGURES_DIR)
    
    print("[7/10] Creating player consistency analysis...")
    plot_player_consistency(player_data, FIGURES_DIR)
    
    print("[8/10] Creating cross-season trajectory...")
    plot_player_trajectory(player_data, FIGURES_DIR)
    
    print("[9/10] Creating quality-minutes scatter...")
    plot_quality_minutes_scatter(player_data, FIGURES_DIR)
    
    print("[10/10] Creating position-specific quality violins...")
    plot_position_quality_violins(player_data, FIGURES_DIR)
    
    print("[11/10] Creating specialist matrix...")
    plot_specialist_matrix(player_data, FIGURES_DIR)
    
    print()
    print("="*70)
    print("✅ ALL VISUALIZATIONS COMPLETE")
    print("="*70)
    print(f"\nOutputs saved to: {FIGURES_DIR.absolute()}/")
    print("\nGenerated files:")
    print("  1. player_metrics_row_counts.png")
    print("  2. coverage_heatmap.png")
    print("  3. metric_distributions.png")
    print("  4. player_mins_dis.png")
    print("  5. player_archetype_scatter.png")
    print("  6. player_consistency.png")
    print("  7. player_trajectory.png")
    print("  8. quality_minutes_scatter.png")
    print("  9. position_quality_violins.png")
    print(" 10. specialist_matrix.png")
    print()