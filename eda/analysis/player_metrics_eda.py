"""
player_metrics_eda.py
─────────────────────
Standalone EDA for raw player metric files.
Expects files loaded from outputs/raw_metrics/recent_club_players/

Six charts:
  III.1 — Row counts per file and season
  III.2 — Player and team coverage heatmap per file
  III.3 — Distribution of raw metrics (violin plots)
  III.4 — Minutes played distribution
  III.5 — Top players per metric after minutes filter
  III.6 — Club / league distribution of players in dataset
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR    = Path("../outputs/raw_metrics/recent_club_players")
SEASONS     = ["2021_2022", "2022_2023", "2023_2024"]
FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)

# Files to load per season, with human-readable labels and the minutes column
FILE_CONFIG = {
    "xg__player__totals.csv": {
        "label":       "xG Totals",
        "minutes_col": "minutes",
        "metric_col":  "goals_minus_xg",
        "metric_label":"Goals − xG",
    },
    "progression__player__profile.csv": {
        "label":       "Progression Profile",
        "minutes_col": "total_mins",
        "metric_col":  "total_progressive_actions_p90",
        "metric_label":"Progressive Actions p90",
    },
    "advanced__player__xg_chain.csv": {
        "label":       "xG Chain",
        "minutes_col": "minutes_played",
        "metric_col":  "xg_chain_per90",
        "metric_label":"xG Chain p90",
    },
    "advanced__player__xg_buildup.csv": {
        "label":       "xG Buildup",
        "minutes_col": "minutes_played",
        "metric_col":  "xg_buildup_per90",
        "metric_label":"xG Buildup p90",
    },
    "advanced__player__network_centrality.csv": {
        "label":       "Network Centrality",
        "minutes_col": None,   # no minutes col — use match count proxy
        "metric_col":  "network_involvement_pct",
        "metric_label":"Network Involvement %",
    },
    "defensive__player__pressures.csv": {
        "label":       "Pressures",
        "minutes_col": "minutes_played",
        "metric_col":  "pressures_per_90",
        "metric_label":"Pressures p90",
    },
    "defensive__player__pressures.csv": {
        "label": "Pressures",
        "minutes_col": "minutes_played",
        "metric_col": "pressures_per_90",
        "metric_label": "Pressures p90",
    },
}

# Minimum minutes for sanity-check charts
MIN_MINUTES = 400

# Colors
SEASON_COLORS = {
    "2021_2022": "#4dabf7",
    "2022_2023": "#2d6a4f",
    "2023_2024": "#f4a261",
}


# ─────────────────────────────────────────────────────────────────────────────
# LOADER
# ─────────────────────────────────────────────────────────────────────────────

def load_all() -> dict:
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
# III.1 — ROW COUNTS PER FILE AND SEASON
# ─────────────────────────────────────────────────────────────────────────────

def plot_row_counts(data: dict, figsize=(13, 6)):
    """
    Grouped bar chart: row counts per file, split by season.
    Shows which files are richest and how data grows season-over-season.
    """
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
                        f"{val:,}", ha="center", fontsize=6.5,
                        fontfamily="monospace", color="#495057")

    ax.set_xticks(x)
    ax.set_xticklabels(files_ordered, rotation=30, ha="right",
                       fontsize=9, fontfamily="monospace")
    ax.set_ylabel("Row Count", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=8, frameon=False, title="Season",
              title_fontsize=8)
    ax.grid(axis="y", alpha=0.25, linestyle=":")
    ax.set_title(
        "III.1 — Raw Record Counts per Metric File and Season\n"
        "xG Chain and Network Centrality are the richest sources; "
        "xG Totals are the most selective (min shots filter)",
        fontsize=11, fontweight="bold", fontfamily="monospace",
        loc="left", pad=14
    )

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "3_1_row_counts.png",
                dpi=180, bbox_inches="tight", facecolor="#ffffff")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# III.2 — PLAYER COVERAGE HEATMAP
# ─────────────────────────────────────────────────────────────────────────────

def plot_coverage_heatmap(data: dict, figsize=(12, 5)):
    """
    Heatmap: unique player counts per file × season.
    Shows missingness pattern — which files cover which seasons well.
    """
    labels  = [FILE_CONFIG[f]["label"] for f in FILE_CONFIG]
    matrix  = np.zeros((len(SEASONS), len(FILE_CONFIG)))

    for i, season in enumerate(SEASONS):
        for j, fname in enumerate(FILE_CONFIG):
            if fname in data[season]:
                matrix[i, j] = data[season][fname]["player"].nunique()

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("#ffffff")

    sns.heatmap(
        matrix, ax=ax,
        xticklabels=labels,
        yticklabels=[s.replace("_", "/") for s in SEASONS],
        cmap="YlGn", annot=True, fmt=".0f",
        annot_kws={"size": 9, "fontfamily": "monospace"},
        linewidths=0.5, linecolor="#f0f0f0",
        cbar_kws={"shrink": 0.7, "label": "Unique Players"},
    )

    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right",
                       fontsize=9, fontfamily="monospace")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0,
                       fontsize=9, fontfamily="monospace")
    ax.set_title(
        "III.2 — Unique Player Coverage per File × Season\n"
        "Cells show how many distinct players appear in each metric file",
        fontsize=11, fontweight="bold", fontfamily="monospace",
        loc="left", pad=14
    )

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "3_2_coverage_heatmap.png",
                dpi=180, bbox_inches="tight", facecolor="#ffffff")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# III.3 — METRIC DISTRIBUTIONS (VIOLIN)
# ─────────────────────────────────────────────────────────────────────────────

def plot_metric_distributions(data: dict, figsize=(16, 8)):
    """
    Violin plots for key metrics across all seasons combined.
    Flags extreme outliers from low-minutes players.
    """
    # Collect all seasons into one df per metric
    metric_data = {}
    for fname, cfg in FILE_CONFIG.items():
        col = cfg["metric_col"]
        frames = []
        for season in SEASONS:
            if fname in data[season] and col in data[season][fname].columns:
                frames.append(data[season][fname][[col, "season_folder"]].dropna())
        if frames:
            metric_data[cfg["metric_label"]] = pd.concat(frames, ignore_index=True)

    n_metrics = len(metric_data)
    fig, axes  = plt.subplots(2, 4, figsize=figsize)
    fig.patch.set_facecolor("#ffffff")
    axes = axes.flatten()

    for i, (label, df) in enumerate(metric_data.items()):
        if i >= len(axes):
            break
        ax  = axes[i]
        ax.set_facecolor("#fafafa")
        col = df.columns[0]

        # Cap extreme outliers for display only
        p99 = df[col].quantile(0.99)
        vals = df[col].clip(upper=p99).dropna().tolist()

        vp = ax.violinplot(vals, showmedians=True, showextrema=True)
        for pc in vp["bodies"]:
            pc.set_facecolor("#4dabf7")
            pc.set_alpha(0.65)
        vp["cmedians"].set_color("#1d3557")
        vp["cmedians"].set_linewidth(2)

        median = np.median(vals)
        ax.text(1.18, median, f"{median:.2f}",
                va="center", fontsize=8,
                fontfamily="monospace", color="#1d3557", fontweight="bold")

        n_outliers = (df[col] > p99).sum()
        if n_outliers > 0:
            ax.text(0.5, 0.97, f"{n_outliers} outliers capped at p99",
                    transform=ax.transAxes, ha="center", fontsize=6.5,
                    color="#e63946", fontfamily="monospace")

        ax.set_title(label, fontsize=8.5, fontweight="bold",
                     fontfamily="monospace")
        ax.set_xticks([])
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.25, linestyle=":")

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        "III.3 — Raw Metric Distributions (all seasons, p99 capped)\n"
        "Outliers flagged — mostly low-minutes players inflating per-90 metrics",
        fontsize=11, fontweight="bold", fontfamily="monospace", y=1.02
    )
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "3_3_metric_distributions.png",
                dpi=180, bbox_inches="tight", facecolor="#ffffff")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# III.4 — MINUTES PLAYED DISTRIBUTION
# ─────────────────────────────────────────────────────────────────────────────

def plot_minutes_distribution(data: dict, figsize=(13, 5)):
    """
    Histogram of minutes played across files that have a minutes column.
    Annotates the MIN_MINUTES threshold — shows what share of data is reliable.
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.patch.set_facecolor("#ffffff")

    files_with_mins = {
        fname: cfg
        for fname, cfg in FILE_CONFIG.items()
        if cfg["minutes_col"] is not None
    }

    # Pick 3 most representative files
    selected = list(files_with_mins.items())[:3]

    for ax, (fname, cfg) in zip(axes, selected):
        ax.set_facecolor("#fafafa")
        col = cfg["minutes_col"]

        frames = []
        for season in SEASONS:
            if fname in data[season] and col in data[season][fname].columns:
                frames.append(data[season][fname][[col]])

        if not frames:
            ax.set_visible(False)
            continue

        mins = pd.concat(frames)[col].dropna()
        mins = mins[mins > 0]

        ax.hist(mins, bins=40, color="#4dabf7", alpha=0.75,
                edgecolor="white", linewidth=0.5)

        # Threshold line
        ax.axvline(MIN_MINUTES, color="#e63946", linestyle="--",
                   linewidth=1.5, label=f"Min threshold ({MIN_MINUTES} mins)")

        pct_below = (mins < MIN_MINUTES).mean() * 100
        ax.text(MIN_MINUTES + 20, ax.get_ylim()[1] * 0.85,
                f"{pct_below:.0f}% below\nthreshold",
                fontsize=8, fontfamily="monospace", color="#e63946")

        ax.set_xlabel("Minutes Played", fontsize=9)
        ax.set_ylabel("Player Count", fontsize=9)
        ax.set_title(cfg["label"], fontsize=9,
                     fontweight="bold", fontfamily="monospace")
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=7.5, frameon=False)
        ax.grid(axis="y", alpha=0.25, linestyle=":")

    fig.suptitle(
        "III.4 — Minutes Played Distribution\n"
        f"Players below {MIN_MINUTES} mins have unreliable per-90 metrics "
        "— volume threshold is non-negotiable",
        fontsize=11, fontweight="bold", fontfamily="monospace", y=1.02
    )
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "3_4_minutes_distribution.png",
                dpi=180, bbox_inches="tight", facecolor="#ffffff")
    plt.show()

"""
player_metrics_advanced_eda.py
──────────────────────────────
Three sophisticated player EDA charts:

  III.7 — Player archetype scatter
          xG Chain vs Network Involvement, bubble = progressive actions,
          colour = defensive profile category

  III.8 — Player consistency
          Mean performance vs coefficient of variation across matches

  III.9 — Cross-season trajectory
          Normalised metric score across 2021/22 → 2022/23 → 2023/24
          for key 2026 World Cup players
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

MIN_MINUTES  = 400
MIN_MATCHES  = 3

SEASON_LABELS = {
    "2021_2022": "2021/22",
    "2022_2023": "2022/23",
    "2023_2024": "2023/24",
    "2021/2022": "2021/22",
    "2022/2023": "2022/23",
    "2023/2024": "2023/24",
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
    "Federico Santiago Valverde Dipetta": "Uruguay",
    "Christian Pulisic":             "United States",
    "Granit Xhaka":                  "Switzerland",
    "Toni Kroos":                    "Germany",
    "Antoine Griezmann":             "France",
    "Achraf Hakimi Mouh":            "Morocco",
    "Alexis Mac Allister":           "Argentina",
    "João Félix Sequeira":           "Portugal",
    "Cody Mathès Gakpo":             "Netherlands",
    "Lamine Yamal Nasraoui Ebana":   "Spain",
}

DEFENSIVE_PROFILE_COLORS = {
    "High Presser":        "#e63946",
    "Balanced Defender":   "#4dabf7",
    "Protector":           "#2d6a4f",
    "Limited Progression": "#adb5bd",
}


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _get_file(player_data: dict, fname: str) -> pd.DataFrame | None:
    """
    Pull a single file across all scopes and seasons from the player_data dict.
    player_data structure: {scope: {season: {fname: df}}}
    Returns a single concatenated df with a 'season_folder' column.
    """
    frames = []
    for scope, seasons in player_data.items():
        for season, files in seasons.items():
            if fname in files:
                df = files[fname].copy()
                df["season_folder"] = season
                frames.append(df)
    if not frames:
        print(f"[!] {fname} not found in player_data")
        return None
    return pd.concat(frames, ignore_index=True)


def _get_file_by_season(player_data: dict, fname: str) -> dict:
    """
    Returns {season: df} for a given file across all scopes.
    Used by trajectory chart which needs per-season data.
    """
    result = {}
    for scope, seasons in player_data.items():
        for season, files in seasons.items():
            if fname in files:
                df = files[fname].copy()
                df["season_folder"] = season
                result[season] = df
    return result


def _latest_season(player_data: dict) -> str:
    """Return the most recent season key present in player_data."""
    seasons = []
    for scope, s_dict in player_data.items():
        seasons.extend(s_dict.keys())
    return sorted(set(seasons))[-1]


def weighted_mean(df, val_col, weight_col):
    df = df.dropna(subset=[val_col, weight_col])
    if df.empty or df[weight_col].sum() == 0:
        return np.nan
    return np.average(df[val_col], weights=df[weight_col])


def aggregate_player(df, val_col, weight_col="minutes_played",
                     min_mins=MIN_MINUTES, min_matches=MIN_MATCHES):
    """
    Aggregate match-level df to player level with minutes-weighted mean.
    Accepts None gracefully — returns empty df.
    """
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
# III.7 — PLAYER ARCHETYPE SCATTER
# ─────────────────────────────────────────────────────────────────────────────

def plot_player_archetype_scatter(player_data: dict, figures_dir: Path = Path("figures"),
                                   figsize=(13, 9)):
    """
    xG Chain p90 (x) vs Network Involvement % (y)
    Bubble size = progressive actions p90
    Colour = defensive profile category
    Most recent season only, 400+ mins filter.
    """
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(exist_ok=True)

    season = _latest_season(player_data)

    def get_season(fname):
        return _get_file_by_season(player_data, fname).get(season)

    chain = aggregate_player(
        get_season("advanced__player__xg_chain.csv"),
        "xg_chain_per90", "minutes_played"
    )[["player", "team", "xg_chain_per90"]]

    net_df = get_season("advanced__player__network_centrality.csv")
    if net_df is not None and "minutes_played" not in net_df.columns:
        net_df["minutes_played"] = 90
    network = aggregate_player(
        net_df, "network_involvement_pct", "minutes_played"
    )[["player", "team", "network_involvement_pct"]]

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

    def_df = get_season("defensive__player__profile.csv")
    if def_df is not None:
        def_profile = (def_df.groupby(["player", "team"])["defensive_profile"]
                              .agg(lambda x: x.mode()[0] if len(x) > 0 else "Unknown")
                              .reset_index())
    else:
        def_profile = pd.DataFrame(columns=["player", "team", "defensive_profile"])

    # Join all
    df = (chain
          .merge(network,     on=["player", "team"], how="inner")
          .merge(prog,        on=["player", "team"], how="left")
          .merge(def_profile, on=["player", "team"], how="left"))

    df = df.dropna(subset=["xg_chain_per90", "network_involvement_pct"])
    df["prog_p90"]          = df["prog_p90"].fillna(df["prog_p90"].median())
    df["defensive_profile"] = df["defensive_profile"].fillna("Unknown")

    # Cap for display
    df["xg_chain_per90"]       = df["xg_chain_per90"].clip(upper=df["xg_chain_per90"].quantile(0.97))
    df["network_involvement_pct"] = df["network_involvement_pct"].clip(
        upper=df["network_involvement_pct"].quantile(0.97))
    df["bubble"] = ((df["prog_p90"] - df["prog_p90"].min()) /
                    (df["prog_p90"].max() - df["prog_p90"].min()) * 180 + 20)

    # Median lines for quadrants
    xm = df["xg_chain_per90"].median()
    ym = df["network_involvement_pct"].median()

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#fafafa")

    for profile, grp in df.groupby("defensive_profile"):
        color = DEFENSIVE_PROFILE_COLORS.get(profile, "#adb5bd")
        ax.scatter(grp["xg_chain_per90"], grp["network_involvement_pct"],
                   s=grp["bubble"], c=color, alpha=0.55,
                   edgecolors="white", linewidths=0.5, zorder=3)

    # Median lines
    ax.axvline(xm, color="#868e96", linestyle=":", lw=1.2, alpha=0.7)
    ax.axhline(ym, color="#868e96", linestyle=":", lw=1.2, alpha=0.7)

    # Quadrant labels
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    kw = dict(fontsize=8, fontfamily="monospace", fontstyle="italic",
              color="#868e96", alpha=0.65)
    ax.text(xmin + 0.002, ymax - 0.8,  "Deep Playmakers",     **kw)
    ax.text(xmax - 0.01,  ymax - 0.8,  "Complete Midfielders", **kw, ha="right")
    ax.text(xmin + 0.002, ymin + 0.3,  "Defensive Specialists", **kw)
    ax.text(xmax - 0.01,  ymin + 0.3,  "Pure Attackers",      **kw, ha="right")

    # Label notable players
    LABEL_PLAYERS = set(WC_2026_PLAYERS.keys())
    for _, row in df[df["player"].isin(LABEL_PLAYERS)].iterrows():
        short = row["player"].split()[0] + " " + row["player"].split()[-1]
        ax.annotate(short, (row["xg_chain_per90"], row["network_involvement_pct"]),
                    xytext=(5, 4), textcoords="offset points",
                    fontsize=7, fontfamily="monospace", color="#343a40")

    # Legends
    profile_handles = [
        mpatches.Patch(color=c, label=p, alpha=0.75)
        for p, c in DEFENSIVE_PROFILE_COLORS.items()
    ]
    size_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#adb5bd",
               markersize=s, label=l, alpha=0.75)
        for s, l in [(4, "Low prog"), (8, "Med prog"), (12, "High prog")]
    ]
    leg1 = ax.legend(handles=profile_handles, fontsize=8, frameon=False,
                     title="Defensive Profile", title_fontsize=8,
                     loc="upper left")
    ax.add_artist(leg1)
    ax.legend(handles=size_handles, fontsize=8, frameon=False,
              title="Progressive Actions", title_fontsize=8,
              loc="lower right")

    ax.set_xlabel("xG Chain p90  (attacking involvement)", fontsize=10)
    ax.set_ylabel("Network Involvement %  (structural centrality)", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.2, linestyle=":", zorder=0)
    ax.set_title(
        "III.7 — Player Archetype Map\n"
        f"xG Chain × Network Involvement × Progressive Actions  |  "
        f"{SEASON_LABELS.get(season, season)}, min {MIN_MINUTES} mins",
        fontsize=11, fontweight="bold", fontfamily="monospace",
        loc="left", pad=14
    )

    plt.tight_layout()
    plt.savefig(figures_dir / "3_7_player_archetype_scatter.png",
                dpi=180, bbox_inches="tight", facecolor="#ffffff")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# III.8 — PLAYER CONSISTENCY
# ─────────────────────────────────────────────────────────────────────────────

def plot_player_consistency(player_data: dict, figures_dir: Path = Path("figures"),
                            figsize=(12, 9)):
    """
    Mean xG Chain p90 (x) vs Coefficient of Variation across matches (y, inverted).
    Lower CV = more consistent. Labelled quadrants.
    Most recent season, min MIN_MATCHES matches.
    """
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(exist_ok=True)
    season = _latest_season(player_data)
    fname  = "advanced__player__xg_chain.csv"
    df     = _get_file_by_season(player_data, fname).get(season)

    if df is None:
        print(f"[!] {fname} not found for {season}")
        return

    # Filter to players with enough matches and minutes
    valid = (df.groupby(["player", "team"])
               .filter(lambda x:
                   len(x) >= MIN_MATCHES and
                   x["minutes_played"].sum() >= MIN_MINUTES))

    stats = (valid.groupby(["player", "team"])["xg_chain_per90"]
                  .agg(mean="mean", std="std", n="count")
                  .reset_index()
                  .dropna())

    stats["cv"] = stats["std"] / stats["mean"].replace(0, np.nan)
    stats        = stats.dropna(subset=["cv"])

    # Cap extremes for display
    stats["mean"] = stats["mean"].clip(upper=stats["mean"].quantile(0.97))
    stats["cv"]   = stats["cv"].clip(upper=stats["cv"].quantile(0.97))

    xm = stats["mean"].median()
    ym = stats["cv"].median()

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#fafafa")

    # Colour by quadrant
    def quadrant_color(row):
        if row["mean"] >= xm and row["cv"] <= ym:  return "#2d6a4f"   # elite consistent
        if row["mean"] >= xm and row["cv"] > ym:   return "#f4a261"   # elite volatile
        if row["mean"] < xm  and row["cv"] <= ym:  return "#4dabf7"   # avg consistent
        return "#dee2e6"                                                # avg volatile

    stats["color"] = stats.apply(quadrant_color, axis=1)

    ax.scatter(stats["mean"], stats["cv"],
               c=stats["color"], s=40, alpha=0.65,
               edgecolors="white", linewidths=0.5, zorder=3)

    ax.axvline(xm, color="#868e96", linestyle=":", lw=1.2, alpha=0.7)
    ax.axhline(ym, color="#868e96", linestyle=":", lw=1.2, alpha=0.7)

    # Quadrant labels
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    kw = dict(fontsize=8.5, fontfamily="monospace", fontstyle="italic", alpha=0.7)
    ax.text(xm + (xmax-xm)*0.05, ymin + (ym-ymin)*0.05,
            "ELITE\nCONSISTENT", color="#2d6a4f", **kw)
    ax.text(xm + (xmax-xm)*0.05, ym + (ymax-ym)*0.6,
            "ELITE\nVOLATILE",   color="#f4a261", **kw)
    ax.text(xmin + (xm-xmin)*0.05, ymin + (ym-ymin)*0.05,
            "AVERAGE\nCONSISTENT", color="#4dabf7", **kw)
    ax.text(xmin + (xm-xmin)*0.05, ym + (ymax-ym)*0.6,
            "AVERAGE\nVOLATILE",   color="#adb5bd", **kw)

    # Label WC 2026 players
    wc_df = stats[stats["player"].isin(WC_2026_PLAYERS.keys())]
    for _, row in wc_df.iterrows():
        short = row["player"].split()[0] + " " + row["player"].split()[-1]
        ax.annotate(short, (row["mean"], row["cv"]),
                    xytext=(5, 4), textcoords="offset points",
                    fontsize=7.5, fontfamily="monospace",
                    color="#343a40", fontweight="bold")
        ax.scatter(row["mean"], row["cv"], c=row["color"],
                   s=80, edgecolors="#343a40", linewidths=1.2,
                   zorder=5)

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
        "III.8 — Player Consistency: Quality vs Reliability\n"
        f"xG Chain mean vs match-to-match variance  |  "
        f"{SEASON_LABELS.get(season, season)}, min {MIN_MATCHES} matches",
        fontsize=11, fontweight="bold", fontfamily="monospace",
        loc="left", pad=14
    )

    plt.tight_layout()
    plt.savefig(figures_dir / "3_8_player_consistency.png",
                dpi=180, bbox_inches="tight", facecolor="#ffffff")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# III.9 — CROSS-SEASON TRAJECTORY
# ─────────────────────────────────────────────────────────────────────────────

def plot_player_trajectory(player_data: dict, figures_dir: Path = Path("figures"),
                            figsize=(14, 10)):
    """
    Bump chart / small multiples showing normalised composite score
    across all available seasons for key 2026 WC players.
    """
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(exist_ok=True)
    # Build a simple composite: mean of normalised xg_chain + network + prog
    METRIC_FILES = {
        "advanced__player__xg_chain.csv":          ("xg_chain_per90",              "minutes_played"),
        "advanced__player__network_centrality.csv": ("network_involvement_pct",     "minutes_played"),
        "progression__player__profile.csv":         ("total_progressive_actions_p90","total_mins"),
    }

    # Get all seasons present across all scopes
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

            # Normalise 0-1 within season
            mn, mx = agg[col].min(), agg[col].max()
            agg[f"{col}_norm"] = (agg[col] - mn) / (mx - mn) if mx != mn else 0.5
            season_dfs.append(agg[["player", "team", f"{col}_norm"]])

        if not season_dfs:
            continue

        # Merge all metrics for this season
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

    # Filter to WC 2026 players present in at least 2 seasons
    wc_traj = traj[traj["player"].isin(WC_2026_PLAYERS.keys())]
    present = wc_traj.groupby("player")["season"].count()
    valid_players = present[present >= 2].index
    wc_traj = wc_traj[wc_traj["player"].isin(valid_players)]

    if wc_traj.empty:
        print("[!] No WC 2026 players found in trajectory data — check player name spelling")
        return

    # Classify trajectory
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

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#fafafa")

    for player, grp in wc_traj.groupby("player"):
        grp  = grp.sort_values("season")
        traj_type = grp["trajectory"].iloc[0]
        color     = TRAJ_COLORS.get(traj_type, "#adb5bd")

        xs = [all_seasons.index(s) for s in grp["season"] if s in all_seasons]
        ys = grp["composite"].tolist()

        ax.plot(xs, ys, color=color, linewidth=1.8,
                alpha=0.75, zorder=3)
        ax.scatter(xs, ys, color=color, s=40,
                   edgecolors="white", linewidths=0.6,
                   zorder=4, alpha=0.85)

        # Label at last point
        short = player.split()[0] + " " + player.split()[-1]
        ax.annotate(short, (xs[-1], ys[-1]),
                    xytext=(6, 0), textcoords="offset points",
                    fontsize=7.5, fontfamily="monospace",
                    color=color, va="center")

    ax.set_xticks(x_vals)
    ax.set_xticklabels(x_labels, fontsize=10, fontfamily="monospace")
    ax.set_ylabel("Normalised Composite Score  (0–1)", fontsize=10)
    ax.set_xlim(-0.15, 2.5)
    ax.set_ylim(-0.05, 1.1)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.2, linestyle=":")

    handles = [
        mpatches.Patch(color=c, label=t)
        for t, c in TRAJ_COLORS.items()
        if t != "Insufficient data"
    ]
    ax.legend(handles=handles, fontsize=8.5, frameon=False,
              title="Trajectory", title_fontsize=8,
              loc="upper left")

    ax.set_title(
        "III.9 — Cross-Season Trajectory: Key 2026 World Cup Players\n"
        "Normalised composite score (xG Chain + Network + Progression)  |  "
        "Validates time-decay weighting",
        fontsize=11, fontweight="bold", fontfamily="monospace",
        loc="left", pad=14
    )

    plt.tight_layout()
    plt.savefig(figures_dir / "3_9_player_trajectory.png",
                dpi=180, bbox_inches="tight", facecolor="#ffffff")
    plt.show()