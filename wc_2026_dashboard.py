"""
2026 FIFA World Cup Readiness Dashboard
Team XOH — Soomi Oh, Yoo Mi Oh | Capstone Analytics Project
"""

from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html, callback_context
from dash.dependencies import ALL
import numpy as np

# ─────────────────────────────────────────────
#  DATA LOADING
# ─────────────────────────────────────────────
DATA_DIR = Path(__file__).parent

mc_df      = pd.read_csv(DATA_DIR / "composite_score" / "outputs" / "monte_carlo_2026.csv")
ta_df      = pd.read_csv(DATA_DIR / "tactical_clustering" / "outputs" / "team_archetypes.csv")
pq_df      = pd.read_csv(DATA_DIR / "player_score" / "outputs" / "player_quality_2026.csv")

pd_df = pd.read_csv(DATA_DIR / "player_score" / "outputs" / "player_details_2026.csv")
pd_df.columns = [f"{col}_{i}" if pd_df.columns.tolist().count(col) > 1 and i > 0 
                 else col for i, col in enumerate(pd_df.columns)]
pd_df = pd_df.rename(columns={"country": "team"})


# Normalise join keys
pq_df = pq_df.rename(columns={"country": "team"})
pd_df = pd_df.rename(columns={"country": "team"})

# Master table: monte_carlo is the 48-team source of truth
master = mc_df.copy()
master = master.merge(ta_df, on="team", how="left", suffixes=("", "_tac"))
master = master.merge(pq_df, on="team", how="left")

# Fill missing tactical data label
master["has_tactical"] = master["archetype"].notna()
master["archetype"] = master["archetype"].fillna("No Tactical Data")

# Numeric coerce
for col in ["final_score", "readiness_score", "player_quality_score",
            "p_champion", "p_group_exit", "gmm_confidence",
            "ppda", "possession_pct", "defensive_line_height",
            "field_tilt_pct", "npxg", "epr",
            "progressive_carry_pct", "avg_xg_per_buildup_possession"]:
    if col in master.columns:
        master[col] = pd.to_numeric(master[col], errors="coerce")

# ─────────────────────────────────────────────
#  THEME
# ─────────────────────────────────────────────
THEME = {
    "bg":          "#0D0E1F",
    "card":        "#161729",
    "card2":       "#1E2038",
    "border":      "#2A2C4A",
    "border_glow": "#F49D52",
    "text":        "#FFFFFF",
    "muted":       "#8B8FA8",
    "accent":      "#F49D52",
    "accent2":     "#759ACE",
    "green":       "#34D399",
    "red":         "#F87171",
    "yellow":      "#FBBF24",
}

ARCHETYPE_COLORS = {
    "High Press / High Output": "#F49D52",
    "Possession Dominant":      "#759ACE",
    "Compact Transition":       "#34D399",
    "Mid-Block Reactive":       "#FBBF24",
    "Moderate Possession":      "#A78BFA",
    "Low Intensity":            "#F87171",
    "No Tactical Data":         "#4B5563",
}

def hex_to_rgba(hex_color, alpha=0.2):
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

PLOTLY_LAYOUT = dict(
    plot_bgcolor=THEME["card"],
    paper_bgcolor=THEME["card"],
    font=dict(color=THEME["text"], family="DM Sans, sans-serif", size=13),
    xaxis=dict(gridcolor=THEME["border"], linecolor=THEME["border"], zeroline=False),
    yaxis=dict(gridcolor=THEME["border"], linecolor=THEME["border"], zeroline=False),
    margin=dict(l=40, r=20, t=40, b=40),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,0,0,0)"),
    hoverlabel=dict(bgcolor=THEME["card2"], bordercolor=THEME["border"],
                    font_color=THEME["text"], font_family="DM Sans, sans-serif"),
)

# Style helpers
def card(children, style=None, **kwargs):
    base = {
        "backgroundColor": THEME["card"],
        "borderRadius": "14px",
        "border": f"1px solid {THEME['border']}",
        "padding": "24px",
        "boxShadow": "0 4px 24px rgba(0,0,0,0.5)",
    }
    if style:
        base.update(style)
    return html.Div(children, style=base, **kwargs)

def section_header(title, subtitle=None):
    return html.Div([
        html.H3(title, style={
            "color": THEME["text"], "margin": "0 0 4px 0",
            "fontSize": "18px", "fontWeight": "700",
            "fontFamily": "DM Sans, sans-serif",
        }),
        html.P(subtitle, style={
            "color": THEME["muted"], "margin": "0",
            "fontSize": "13px", "fontFamily": "DM Sans, sans-serif",
        }) if subtitle else None,
    ], style={"marginBottom": "20px"})

def stat_card(label, value_id, accent_color=None):
    color = accent_color or THEME["accent"]
    return card(
        style={"borderLeft": f"4px solid {color}", "padding": "20px 24px"},
        children=[
            html.Div(label, style={
                "color": THEME["muted"], "fontSize": "12px",
                "textTransform": "uppercase", "letterSpacing": "1px",
                "fontFamily": "DM Sans, sans-serif", "marginBottom": "8px",
            }),
            html.Div(id=value_id, style={
                "color": THEME["text"], "fontSize": "28px",
                "fontWeight": "700", "fontFamily": "JetBrains Mono, monospace",
            }),
        ],
    )

# ─────────────────────────────────────────────
#  STATIC FIGURES (computed once)
# ─────────────────────────────────────────────

def make_scatter():
    df = master.dropna(subset=["final_score", "player_quality_score"]).copy()
    df["bubble_size"] = df["p_champion"].fillna(0) * 6 + 6
    df["hover"] = df.apply(lambda r: (
        f"<b>{r['team']}</b><br>"
        f"Archetype: {r['archetype']}<br>"
        f"Readiness: {r['readiness_score']:.1f}<br>"
        f"Tactical: {r['final_score']:.1f}<br>"
        f"Player Quality: {r['player_quality_score']:.1f}<br>"
        f"Champion Prob: {r['p_champion']:.1f}%"
    ), axis=1)

    fig = go.Figure()
    for arch, grp in df.groupby("archetype"):
        fig.add_trace(go.Scatter(
            x=grp["final_score"], y=grp["player_quality_score"],
            mode="markers+text",
            name=arch,
            marker=dict(
                size=grp["bubble_size"],
                color=ARCHETYPE_COLORS.get(arch, "#888"),
                opacity=0.85,
                line=dict(color="rgba(255,255,255,0.15)", width=1),
            ),
            text=grp["team"],
            textposition="top center",
            textfont=dict(size=10, color="rgba(255,255,255,0.7)"),
            hovertext=grp["hover"],
            hoverinfo="text",
            customdata=grp["team"],
        ))



    fig.update_layout(**PLOTLY_LAYOUT)
    fig.update_layout(
        xaxis_title="Tactical Score",
        yaxis_title="Player Quality Score",
        legend=dict(orientation="v", x=1.01, y=1, bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
        margin=dict(l=50, r=160, t=30, b=50),
        height=460,
    )


    # Add quadrant annotations
    fig.add_annotation(x=25, y=62, text="Strong Squad,<br>Weak System",
        showarrow=False, font=dict(color=THEME["muted"], size=10),
        bgcolor="rgba(0,0,0,0.3)")
    fig.add_annotation(x=78, y=62, text="Elite",
        showarrow=False, font=dict(color=THEME["accent"], size=10),
        bgcolor="rgba(0,0,0,0.3)")
    fig.add_annotation(x=25, y=30, text="Underdogs",
        showarrow=False, font=dict(color=THEME["muted"], size=10),
        bgcolor="rgba(0,0,0,0.3)")
    fig.add_annotation(x=78, y=30, text="Good System,<br>Thin Squad",
        showarrow=False, font=dict(color=THEME["muted"], size=10),
        bgcolor="rgba(0,0,0,0.3)")
    # Quadrant lines
    fig.add_hline(y=50, line_dash="dot", line_color=THEME["border"], line_width=1)
    fig.add_vline(x=55, line_dash="dot", line_color=THEME["border"], line_width=1)
    return fig


def make_archetype_donut():
    counts = master.groupby("archetype").agg(
        Count=("team", "count"),
        AvgReadiness=("readiness_score", "mean")
    ).reset_index()
    colors = [ARCHETYPE_COLORS.get(a, "#888") for a in counts["archetype"]]
    fig = go.Figure(go.Pie(
        labels=counts["archetype"],
        values=counts["Count"],
        hole=0.55,
        marker=dict(colors=colors, line=dict(color=THEME["bg"], width=2)),
        textinfo="label+value",
        textfont=dict(size=11),
        hovertemplate="<b>%{label}</b><br>Teams: %{value}<br>Avg Readiness: %{customdata:.1f}<extra></extra>",
        customdata=counts["AvgReadiness"],
    ))
    

    fig.update_layout(**PLOTLY_LAYOUT)
    fig.update_layout(
        showlegend=False,
        margin=dict(l=10, r=10, t=10, b=10),
        height=300,
        annotations=[dict(
            text="<b>6</b><br><span style='font-size:11px'>Archetypes</span>",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20, color=THEME["text"], family="JetBrains Mono, monospace"),
        )],
    )

    return fig


def make_mc_bars():
    df = master.sort_values("readiness_score", ascending=True).head(48)
    stages = [
        ("p_group_exit",    "Group Exit",   "#4B5563"),
        ("p_round_of_32",   "Round of 32",  "#6B7280"),
        ("p_round_of_16",   "Round of 16",  "#759ACE"),
        ("p_quarter_final", "Quarter-Final","#A78BFA"),
        ("p_semi_final",    "Semi-Final",   "#FBBF24"),
        ("p_runner_up",     "Runner-Up",    "#F49D52"),
        ("p_champion",      "Champion",     "#34D399"),
    ]
    fig = go.Figure()
    for col, label, color in stages:
        if col in df.columns:
            fig.add_trace(go.Bar(
                y=df["team"], x=df[col],
                name=label,
                orientation="h",
                marker_color=color,
                hovertemplate=f"<b>%{{y}}</b><br>{label}: %{{x:.1f}}%<extra></extra>",
            ))
    fig.update_layout(**PLOTLY_LAYOUT)
    fig.update_layout(
        barmode="stack",
        xaxis_title="Probability (%)",
        yaxis_title="",
        xaxis=dict(range=[0, 100], gridcolor=THEME["border"]),
        legend=dict(orientation="h", y=1.02, x=0, bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
        height=1200,
        margin=dict(l=120, r=20, t=50, b=40),
    )
    return fig


def make_radar(archetype_name):
    metrics = ["ppda", "possession_pct", "defensive_line_height",
               "field_tilt_pct", "npxg", "epr",
               "progressive_carry_pct", "avg_xg_per_buildup_possession"]
    labels  = ["PPDA", "Possession%", "Def Line Height",
               "Field Tilt%", "npxG", "EPR",
               "Prog Carry%", "xG/Buildup"]

    df = ta_df.copy()
    for m in metrics:
        df[m] = pd.to_numeric(df[m], errors="coerce")

    # Normalise each metric 0-1
    for m in metrics:
        mn, mx = df[m].min(), df[m].max()
        df[m + "_norm"] = (df[m] - mn) / (mx - mn + 1e-9)

    norm_cols = [m + "_norm" for m in metrics]
    arch_avg = df[df["archetype"] == archetype_name][norm_cols].mean().values
    overall_avg = df[norm_cols].mean().values

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=list(overall_avg) + [overall_avg[0]],
        theta=labels + [labels[0]],
        fill="toself", name="All Teams Avg",
        line_color=THEME["muted"], fillcolor="rgba(139,143,168,0.1)",
    ))
    fig.add_trace(go.Scatterpolar(
        r=list(arch_avg) + [arch_avg[0]],
        theta=labels + [labels[0]],
        fill="toself", name=archetype_name,
        line_color=ARCHETYPE_COLORS.get(archetype_name, THEME["accent"]),
        #fillcolor="rgba(244,157,82,0.2)",
        fillcolor=hex_to_rgba(ARCHETYPE_COLORS.get(archetype_name, THEME["accent"]), 0.2),
        #fillcolor=ARCHETYPE_COLORS.get(archetype_name, THEME["accent"]).replace(")", ",0.2)").replace("rgb", "rgba") if ARCHETYPE_COLORS.get(archetype_name, "").startswith("rgb") else ARCHETYPE_COLORS.get(archetype_name, THEME["accent"]) + "33",
    ))
    fig.update_layout(**PLOTLY_LAYOUT)
    fig.update_layout(
        polar=dict(
            bgcolor=THEME["card"],
            radialaxis=dict(visible=True, range=[0, 1], gridcolor=THEME["border"],
                            tickfont=dict(color=THEME["muted"], size=9)),
            angularaxis=dict(gridcolor=THEME["border"], linecolor=THEME["border"],
                             tickfont=dict(color=THEME["text"], size=11)),
        ),
        showlegend=True,
        height=360,
        margin=dict(l=50, r=50, t=40, b=40),
        legend=dict(orientation="h", y=-0.1, x=0.5, xanchor="center"),
    )
    return fig


def make_team_waterfall(team_name):
    row = master[master["team"] == team_name]
    if row.empty:
        return go.Figure()
    row = row.iloc[0]
    stages, values, colors = [], [], []

    if pd.notna(row.get("archetype_score")):
        stages.append("Base Archetype")
        values.append(float(row["archetype_score"]))
        colors.append(THEME["accent2"])
    if pd.notna(row.get("blend_score")) and pd.notna(row.get("archetype_score")):
        diff = float(row["blend_score"]) - float(row["archetype_score"])
        stages.append("GMM Blend")
        values.append(diff)
        colors.append(THEME["green"] if diff >= 0 else THEME["red"])
    if pd.notna(row.get("final_score")) and pd.notna(row.get("blend_score")):
        ev_disc = float(row["final_score"]) - float(row.get("quality_adjustment", 0) or 0) - float(row["blend_score"])
        if abs(ev_disc) > 0.01:
            stages.append("Evidence Discount")
            values.append(ev_disc)
            colors.append(THEME["red"] if ev_disc < 0 else THEME["green"])
    if pd.notna(row.get("quality_adjustment")):
        stages.append("Quality Adj")
        values.append(float(row["quality_adjustment"]))
        colors.append(THEME["green"] if float(row["quality_adjustment"]) >= 0 else THEME["red"])
    if pd.notna(row.get("final_score")):
        stages.append("Tactical Score")
        values.append(float(row["final_score"]))
        colors.append(THEME["accent"])
    if pd.notna(row.get("readiness_score")):
        lift = float(row["readiness_score"]) - float(row.get("final_score") or 0)
        stages.append("Player + Context Lift")
        values.append(lift)
        colors.append(THEME["green"] if lift >= 0 else THEME["red"])
        stages.append("Readiness Score")
        values.append(float(row["readiness_score"]))
        colors.append(THEME["yellow"])

    # Simple bar waterfall
    cumulative = []
    running = 0
    for i, (s, v) in enumerate(zip(stages, values)):
        if s in ("Base Archetype", "Tactical Score", "Readiness Score"):
            cumulative.append({"stage": s, "value": v, "base": 0, "is_total": True})
            running = v
        else:
            cumulative.append({"stage": s, "value": v, "base": running, "is_total": False})
            running += v

    fig = go.Figure()
    for i, d in enumerate(cumulative):
        if d["is_total"]:
            fig.add_trace(go.Bar(
                x=[d["stage"]], y=[d["value"]],
                marker_color=colors[i], name=d["stage"],
                text=[f"{d['value']:.1f}"], textposition="outside",
                textfont=dict(color=THEME["text"], size=12, family="JetBrains Mono, monospace"),
                showlegend=False,
                hovertemplate=f"<b>{d['stage']}</b><br>Score: {d['value']:.1f}<extra></extra>",
            ))
        else:
            fig.add_trace(go.Bar(
                x=[d["stage"]], y=[d["value"]], base=d["base"],
                marker_color=colors[i], name=d["stage"],
                text=[f"{'+' if d['value']>=0 else ''}{d['value']:.1f}"],
                textposition="outside",
                textfont=dict(color=THEME["text"], size=12, family="JetBrains Mono, monospace"),
                showlegend=False,
                hovertemplate=f"<b>{d['stage']}</b><br>Δ: {d['value']:+.1f}<extra></extra>",
            ))
    fig.update_layout( **PLOTLY_LAYOUT)
    fig.update_layout(
        barmode="overlay",
        yaxis=dict(range=[0, 110], gridcolor=THEME["border"]),
        height=320,
        margin=dict(l=20, r=20, t=20, b=20),
    )
    return fig


def make_player_heatmap(team_name):
    mask = pd_df["team"].values == team_name
    players = pd_df.iloc[mask.nonzero()[0]].copy().reset_index(drop=True)

    if players.empty:
        return None
    trait_cols = ["Final_Third_Output_score", "Progression_score",
                  "Control_score", "Mobility_Intensity_score"]
    for c in trait_cols:
        players[c] = pd.to_numeric(players[c], errors="coerce")
    players = players.dropna(subset=trait_cols, how="all")
    players = players.sort_values("final_score", ascending=False)

    z = players[trait_cols].fillna(0).values
    labels_x = ["Final Third", "Progression", "Control", "Mobility/Intensity"]
    labels_y = [f"{r['player']} ({r['position_archetype']})" for _, r in players.iterrows()]

    fig = go.Figure(go.Heatmap(
        z=z, x=labels_x, y=labels_y,
        colorscale=[[0, "#1E2038"], [0.5, "#759ACE"], [1, "#F49D52"]],
        zmin=0, zmax=100,
        text=[[f"{v:.0f}" for v in row] for row in z],
        texttemplate="%{text}",
        textfont=dict(size=11),
        hovertemplate="<b>%{y}</b><br>%{x}: %{z:.1f}<extra></extra>",
        showscale=True,

        colorbar=dict(
            title=dict(text="Score", font=dict(color=THEME["text"])),
            tickfont=dict(color=THEME["text"]),
            bgcolor=THEME["card"], bordercolor=THEME["border"],
        ),
    ))
    fig.update_layout( **PLOTLY_LAYOUT)
    fig.update_layout(
        height=max(250, len(players) * 45 + 60),
        margin=dict(l=180, r=80, t=20, b=50),
        xaxis=dict(side="top"),
        yaxis=dict(autorange="reversed"),
    )
    return fig


# ─────────────────────────────────────────────
#  LAYOUT HELPERS
# ─────────────────────────────────────────────
def nav_tabs():
    tab_style = {
        "backgroundColor": "transparent",
        "color": THEME["muted"],
        "border": "none",
        "borderBottom": f"2px solid transparent",
        "padding": "10px 24px",
        "fontSize": "14px",
        "fontWeight": "600",
        "fontFamily": "DM Sans, sans-serif",
        "cursor": "pointer",
        "transition": "all 0.2s ease",
    }
    tab_selected_style = {
        **tab_style,
        "color": THEME["accent"],
        "borderBottom": f"2px solid {THEME['accent']}",
    }
    return dcc.Tabs(
        id="main-tabs", value="war-room",
        style={"borderBottom": f"1px solid {THEME['border']}", "marginBottom": "28px"},
        colors={"border": "transparent", "primary": THEME["accent"], "background": "transparent"},
        children=[
            dcc.Tab(label="⚔️  War Room",       value="war-room",   style=tab_style, selected_style=tab_selected_style),
            dcc.Tab(label="🎯  Archetypes",      value="archetypes", style=tab_style, selected_style=tab_selected_style),
            dcc.Tab(label="🔍  Team Deep Dive",  value="team-dive",  style=tab_style, selected_style=tab_selected_style),
            dcc.Tab(label="🎲  Monte Carlo",     value="monte-carlo",style=tab_style, selected_style=tab_selected_style),
            dcc.Tab(label="📋  About & Limits",  value="about",      style=tab_style, selected_style=tab_selected_style),
        ],
    )


# ─────────────────────────────────────────────
#  APP
# ─────────────────────────────────────────────
app = Dash(__name__, suppress_callback_exceptions=True)
app.title = "WC 2026 Readiness — Team XOH"

app.layout = html.Div(
    style={"backgroundColor": THEME["bg"], "minHeight": "100vh", "padding": "0"},
    children=[
        html.Link(rel="stylesheet",
            href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=DM+Sans:wght@300;400;500;600;700&display=swap"),

        # ── Header ──
        html.Div(
            style={
                "background": f"linear-gradient(135deg, {THEME['card']} 0%, {THEME['card2']} 100%)",
                "borderBottom": f"1px solid {THEME['border']}",
                "padding": "28px 48px 24px",
            },
            children=[
                html.Div(style={"maxWidth": "1400px", "margin": "0 auto"}, children=[
                    html.Div(style={"display": "flex", "alignItems": "center", "justifyContent": "space-between"}, children=[
                        html.Div([
                            html.H1("⚽ WC 2026 Readiness Dashboard", style={
                                "color": THEME["text"], "margin": "0 0 6px 0",
                                "fontSize": "28px", "fontWeight": "700",
                                "fontFamily": "DM Sans, sans-serif",
                            }),
                            html.P("48-Nation Data-Driven Readiness Score | Team XOH — Soomi Oh, Yoo Mi Oh | Capstone Analytics",
                                style={"color": THEME["muted"], "margin": "0", "fontSize": "13px", "fontFamily": "DM Sans, sans-serif"}),
                        ]),
                        html.Div([
                            html.Div("⚠️ Data Limitations", style={"color": THEME["yellow"], "fontWeight": "600", "fontSize": "12px", "marginBottom": "4px"}),
                            html.Div("Tactical: StatsBomb WC2022, Euro2024, Copa2024, AFCON2023 only. 9 teams have no tactical data. Player scores from club data (2021–25); coverage varies by league.",
                                style={"color": THEME["muted"], "fontSize": "11px", "maxWidth": "420px",
                                       "fontFamily": "DM Sans, sans-serif", "lineHeight": "1.5"}),
                        ], style={
                            "backgroundColor": "rgba(251,191,36,0.07)",
                            "border": f"1px solid rgba(251,191,36,0.25)",
                            "borderRadius": "8px", "padding": "10px 14px",
                        }),
                    ]),
                ]),
            ],
        ),

        # ── Main Content ──
        html.Div(style={"maxWidth": "1400px", "margin": "0 auto", "padding": "28px 48px"}, children=[
            nav_tabs(),
            html.Div(id="tab-content"),
        ]),
    ],
)


# ─────────────────────────────────────────────
#  TAB CONTENT ROUTER
# ─────────────────────────────────────────────
@app.callback(Output("tab-content", "children"), Input("main-tabs", "value"))
def render_tab(tab):
    if tab == "war-room":
        return war_room_layout()
    elif tab == "archetypes":
        return archetypes_layout()
    elif tab == "team-dive":
        return team_dive_layout()
    elif tab == "monte-carlo":
        return monte_carlo_layout()
    elif tab == "about":
        return about_layout()
    return html.Div("Select a tab")


def _signal_disagreement_panel():
    """War Room panel: teams where the three signals disagree most."""
    fig, df = make_signal_disagreement()

    # Annotation blurbs for the most interesting cases
    notable = {
        "Belgium":       "Tactically weak (Mid-Block), elite player pool. Classic golden generation tension.",
        "Italy":         "Lowest tactical score among qualifiers. Player quality carries them.",
        "United States": "The Pochettino effect — tactical score matches France and Germany on paper (High Press / High Output), then drops sharply on player quality. The most asymmetric profile in the tournament.",
        "Norway":        "No tactical data (absent from all 4 tournaments). Haaland severely undercovered — Champions League not in dataset. Score 41.92, highest volatility in the framework. Almost entirely FIFA fallback.",
        "Ghana":         "No player quality score. Tactical data thin. Model has very low confidence here.",
        "Senegal":       "Good AFCON tactical score. Player scoring limited by European-skewed club coverage.",
        "Morocco":       "Strong tactical identity. Player data coverage moderate — AFCON/WC-only exposure.",
    }

    chips = []
    for _, row in df.iterrows():
        note = notable.get(row["team"])
        chips.append(html.Div(style={
            "backgroundColor": THEME["card2"], "borderRadius": "10px",
            "padding": "14px 16px", "border": f"1px solid {THEME['border']}",
        }, children=[
            html.Div(style={"display": "flex", "justifyContent": "space-between",
                            "alignItems": "flex-start", "marginBottom": "6px"}, children=[
                html.Div(row["team"], style={"color": THEME["text"], "fontWeight": "700",
                                             "fontSize": "14px", "fontFamily": "DM Sans, sans-serif"}),
                html.Div(f"Δ {row['rank_std']:.1f}", style={
                    "color": THEME["yellow"], "fontFamily": "JetBrains Mono, monospace",
                    "fontSize": "12px", "fontWeight": "600",
                }),
            ]),
            html.Div(row["gap_label"], style={
                "color": THEME["accent2"], "fontSize": "11px",
                "fontFamily": "DM Sans, sans-serif", "marginBottom": "6px",
            }),
            html.Div(note or "Signals diverge — inspect Team Deep Dive for details.", style={
                "color": THEME["muted"], "fontSize": "11px",
                "fontFamily": "DM Sans, sans-serif", "lineHeight": "1.5",
            }),
        ]))

    return card([
        section_header(
            "⚠️  Where Our Signals Disagree Most",
            "Teams where readiness rank, player quality rank, and tactical rank diverge significantly — "
            "the model is least certain here. Higher Δ = bigger disagreement between the three signals."
        ),
        html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 380px", "gap": "20px"}, children=[
            dcc.Graph(figure=fig, config={"displayModeBar": False}),
            html.Div(style={"display": "flex", "flexDirection": "column", "gap": "10px",
                            "maxHeight": "420px", "overflowY": "auto"}, children=chips),
        ]),
    ])


# ─────────────────────────────────────────────
#  TAB 1 — WAR ROOM
# ─────────────────────────────────────────────
def war_room_layout():
    top3 = master.nlargest(3, "p_champion")[["team", "p_champion", "readiness_score", "archetype"]]
    underdog = master[master["readiness_score"] < 52].nlargest(1, "p_champion")

    return html.Div([
        # Stat cards
        html.Div(style={
            "display": "grid", "gridTemplateColumns": "repeat(4, 1fr)", "gap": "16px", "marginBottom": "24px"
        }, children=[
            stat_card("🏆 Top Favourite", "stat-favourite", THEME["accent"]),
            stat_card("📊 Avg Readiness Score", "stat-avg-readiness", THEME["accent2"]),
            stat_card("🔥 Strongest Player Pool", "stat-best-squad", THEME["green"]),
            stat_card("⚡ Biggest Underdog", "stat-underdog", THEME["yellow"]),
        ]),

        # Scatter + Donut row
        html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 320px", "gap": "20px", "marginBottom": "24px"}, children=[
            card([
                section_header("Tactical Score vs Player Quality",
                    "Bubble size = Champion probability. Quadrant lines at tactical=55, player quality=50."),
                dcc.Graph(figure=make_scatter(), config={"displayModeBar": False}),
            ]),
            html.Div([
                card([
                    section_header("Tactical Archetypes", "48 WC2026 nations"),
                    dcc.Graph(figure=make_archetype_donut(), config={"displayModeBar": False}),
                ], style={"marginBottom": "16px"}),
                card([
                    section_header("Archetype Legend"),
                    html.Div([
                        html.Div(style={"display": "flex", "alignItems": "center", "gap": "10px", "marginBottom": "8px"}, children=[
                            html.Div(style={"width": "12px", "height": "12px", "borderRadius": "50%",
                                           "backgroundColor": color, "flexShrink": "0"}),
                            html.Div(name, style={"color": THEME["muted"], "fontSize": "12px",
                                                  "fontFamily": "DM Sans, sans-serif"}),
                        ])
                        for name, color in ARCHETYPE_COLORS.items()
                    ]),
                ]),
            ]),
        ]),

        # Top contenders table
        card([
            section_header("Top 10 Contenders", "Ranked by Monte Carlo champion probability"),
            html.Div(style={"overflowX": "auto"}, children=[
                html.Table(style={"width": "100%", "borderCollapse": "collapse"}, children=[
                    html.Thead(html.Tr([
                        html.Th(h, style={
                            "color": THEME["muted"], "fontSize": "11px", "textTransform": "uppercase",
                            "letterSpacing": "1px", "padding": "8px 16px", "textAlign": "left" if i == 0 else "center",
                            "borderBottom": f"1px solid {THEME['border']}",
                            "fontFamily": "DM Sans, sans-serif",
                        })
                        for i, h in enumerate(["Team", "Group", "Archetype", "Readiness", "Tactical", "Player Quality", "Champion %", "Group Exit %"])
                    ])),
                    html.Tbody([
                        html.Tr(
                            style={"borderBottom": f"1px solid {THEME['border']}",
                                   "backgroundColor": "rgba(255,255,255,0.02)" if i % 2 else "transparent"},
                            children=[
                                html.Td(style={"padding": "12px 16px"}, children=[
                                    html.Div(style={"display": "flex", "alignItems": "center", "gap": "10px"}, children=[
                                        html.Span(f"#{int(row['rank'])}", style={
                                            "color": THEME["accent"] if row["rank"] <= 3 else THEME["muted"],
                                            "fontFamily": "JetBrains Mono, monospace", "fontSize": "12px",
                                            "minWidth": "28px",
                                        }),
                                        html.Span(row["team"], style={"color": THEME["text"], "fontWeight": "600",
                                                                       "fontFamily": "DM Sans, sans-serif"}),
                                    ]),
                                ]),
                                html.Td(row["group"], style={"textAlign": "center", "color": THEME["muted"],
                                    "fontFamily": "JetBrains Mono, monospace", "fontSize": "13px"}),
                                html.Td(style={"textAlign": "center"}, children=[
                                    html.Span(row["archetype"], style={
                                        "backgroundColor": ARCHETYPE_COLORS.get(row["archetype"], "#888") + "22",
                                        "color": ARCHETYPE_COLORS.get(row["archetype"], "#888"),
                                        "fontSize": "11px", "padding": "3px 8px", "borderRadius": "20px",
                                        "fontFamily": "DM Sans, sans-serif", "whiteSpace": "nowrap",
                                    }),
                                ]),
                                *[html.Td(
                                    f"{row[col]:.1f}" if pd.notna(row.get(col)) else "—",
                                    style={"textAlign": "center", "color": THEME["text"],
                                           "fontFamily": "JetBrains Mono, monospace", "fontSize": "13px",
                                           "padding": "12px 16px"}
                                ) for col in ["readiness_score", "final_score", "player_quality_score"]],
                                html.Td(style={"textAlign": "center", "padding": "12px 16px"}, children=[
                                    html.Span(f"{row['p_champion']:.1f}%", style={
                                        "color": THEME["green"], "fontFamily": "JetBrains Mono, monospace",
                                        "fontWeight": "700", "fontSize": "14px",
                                    }),
                                ]),
                                html.Td(f"{row['p_group_exit']:.1f}%", style={
                                    "textAlign": "center", "color": THEME["red"],
                                    "fontFamily": "JetBrains Mono, monospace", "fontSize": "13px", "padding": "12px 16px",
                                }),
                            ]
                        )
                        for i, (_, row) in enumerate(master.nlargest(10, "p_champion").iterrows())
                    ]),
                ]),
            ]),
        ]),

        # Signal disagreement panel
        html.Div(style={"marginTop": "20px"}, children=[_signal_disagreement_panel()]),
    ])


@app.callback(
    [Output("stat-favourite", "children"),
     Output("stat-avg-readiness", "children"),
     Output("stat-best-squad", "children"),
     Output("stat-underdog", "children")],
    Input("main-tabs", "value"),
)
def update_war_room_stats(tab):
    fav = master.nlargest(1, "p_champion").iloc[0]
    best_squad = master.nlargest(1, "player_quality_score").iloc[0]
    underdog = master[master["readiness_score"] < 53].nlargest(1, "p_champion")
    ug_text = f"{underdog.iloc[0]['team']} ({underdog.iloc[0]['p_champion']:.1f}%)" if not underdog.empty else "—"

    return (
        f"{fav['team']} ({fav['p_champion']:.1f}%)",
        f"{master['readiness_score'].mean():.1f}",
        f"{best_squad['team']} ({best_squad['player_quality_score']:.1f})",
        ug_text,
    )


# ─────────────────────────────────────────────
#  TAB 2 — ARCHETYPES
# ─────────────────────────────────────────────
def archetypes_layout():
    archetypes = [a for a in ARCHETYPE_COLORS if a != "No Tactical Data"]
    return html.Div([
        html.Div(style={"display": "flex", "gap": "12px", "marginBottom": "24px", "flexWrap": "wrap"}, children=[
            html.Button(a, id={"type": "arch-btn", "index": a},
                style={
                    "backgroundColor": ARCHETYPE_COLORS[a] + "22",
                    "color": ARCHETYPE_COLORS[a],
                    "border": f"1px solid {ARCHETYPE_COLORS[a]}55",
                    "borderRadius": "20px", "padding": "8px 18px",
                    "fontSize": "13px", "fontWeight": "600",
                    "fontFamily": "DM Sans, sans-serif", "cursor": "pointer",
                })
            for a in archetypes
        ]),
        html.Div(id="archetype-content"),
    ])


@app.callback(
    Output("archetype-content", "children"),
    Input({"type": "arch-btn", "index": ALL}, "n_clicks"),
    prevent_initial_call=False,
)
def update_archetype(n_clicks_list):
    ctx = callback_context
    if not ctx.triggered or not any(n for n in n_clicks_list if n):
        arch = "High Press / High Output"
    else:
        triggered_id = ctx.triggered[0]["prop_id"]
        import json
        arch = json.loads(triggered_id.split(".")[0])["index"]

    color = ARCHETYPE_COLORS.get(arch, THEME["accent"])
    arch_teams = master[master["archetype"] == arch].sort_values("readiness_score", ascending=False)
    ta_teams = ta_df[ta_df["archetype"] == arch].copy()

    return html.Div([
        html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "20px", "marginBottom": "20px"}, children=[
            card([
                section_header(f"{arch} — Tactical Radar",
                    f"{len(ta_teams)} teams · Avg readiness: {arch_teams['readiness_score'].mean():.1f}"),
                dcc.Graph(figure=make_radar(arch), config={"displayModeBar": False}),
            ]),
            card([
                section_header("Teams in this Archetype", "Sorted by readiness score"),
                html.Div(style={"maxHeight": "320px", "overflowY": "auto"}, children=[
                    html.Div(style={
                        "display": "flex", "alignItems": "center", "justifyContent": "space-between",
                        "padding": "10px 0", "borderBottom": f"1px solid {THEME['border']}",
                    }, children=[
                        html.Div(style={"display": "flex", "alignItems": "center", "gap": "12px"}, children=[
                            html.Div(f"#{i+1}", style={
                                "color": THEME["muted"], "fontFamily": "JetBrains Mono, monospace",
                                "fontSize": "12px", "minWidth": "24px",
                            }),
                            html.Div([
                                html.Div(row["team"], style={"color": THEME["text"], "fontWeight": "600",
                                                              "fontSize": "14px", "fontFamily": "DM Sans, sans-serif"}),
                                html.Div(f"Group {row['group']}", style={"color": THEME["muted"],
                                    "fontSize": "11px", "fontFamily": "DM Sans, sans-serif"}),
                            ]),
                        ]),
                        html.Div(style={"textAlign": "right"}, children=[
                            html.Div(f"{row['readiness_score']:.1f}", style={
                                "color": color, "fontFamily": "JetBrains Mono, monospace",
                                "fontWeight": "700", "fontSize": "16px",
                            }),
                            html.Div(f"Champion: {row['p_champion']:.1f}%", style={
                                "color": THEME["muted"], "fontSize": "11px",
                                "fontFamily": "DM Sans, sans-serif",
                            }),
                        ]),
                    ])
                    for i, (_, row) in enumerate(arch_teams.iterrows())
                ]),
            ]),
        ]),

        # GMM blend note for mixed teams
        card([
            section_header("Style Blends (GMM Confidence < 1.0)",
                "Teams sitting on the boundary between two tactical archetypes"),
            html.Div(style={"display": "flex", "flexWrap": "wrap", "gap": "12px"}, children=[
                html.Div(style={
                    "backgroundColor": THEME["card2"], "borderRadius": "10px",
                    "padding": "12px 16px", "border": f"1px solid {THEME['border']}",
                    "minWidth": "200px",
                }, children=[
                    html.Div(row["team"], style={"color": THEME["text"], "fontWeight": "600",
                                                  "fontFamily": "DM Sans, sans-serif", "marginBottom": "6px"}),
                    html.Div(style={"display": "flex", "alignItems": "center", "gap": "6px"}, children=[
                        html.Div(style={
                            "backgroundColor": ARCHETYPE_COLORS.get(row["archetype"], "#888") + "44",
                            "borderRadius": "4px", "padding": "2px 8px",
                            "color": ARCHETYPE_COLORS.get(row["archetype"], "#888"),
                            "fontSize": "11px", "fontFamily": "DM Sans, sans-serif",
                        }, children=f"{int(float(row['gmm_confidence'])*100)}% {row['archetype'].split('/')[0].strip()}"),
                        html.Div("→", style={"color": THEME["muted"]}),
                        html.Div(style={
                            "backgroundColor": ARCHETYPE_COLORS.get(row.get("second_archetype",""), "#888") + "44",
                            "borderRadius": "4px", "padding": "2px 8px",
                            "color": ARCHETYPE_COLORS.get(row.get("second_archetype",""), "#888"),
                            "fontSize": "11px", "fontFamily": "DM Sans, sans-serif",
                        }, children=f"{int((1-float(row['gmm_confidence']))*100)}% {(row.get('second_archetype') or '').split('/')[0].strip()}"),
                    ]),
                ])
                for _, row in ta_df[
                    (ta_df["archetype"] == arch) &
                    (pd.to_numeric(ta_df["gmm_confidence"], errors="coerce") < 1.0)
                ].iterrows()
            ] or [html.Div("All teams in this archetype are pure (GMM confidence = 1.0)",
                           style={"color": THEME["muted"], "fontFamily": "DM Sans, sans-serif", "fontSize": "13px"})]),
        ]),
    ])


# ─────────────────────────────────────────────
#  TAB 3 — TEAM DEEP DIVE
# ─────────────────────────────────────────────
def team_dive_layout():
    team_options = [{"label": f"{r['team']} (Group {r['group']})", "value": r["team"]}
                    for _, r in master.sort_values("readiness_score", ascending=False).iterrows()]
    return html.Div([
        card(style={"marginBottom": "20px"}, children=[
            html.Div(style={"display": "flex", "alignItems": "center", "gap": "20px"}, children=[
                html.Div("Select Team:", style={"color": THEME["text"], "fontWeight": "600",
                                                 "fontFamily": "DM Sans, sans-serif", "whiteSpace": "nowrap"}),
                dcc.Dropdown(
                    id="team-dropdown",
                    options=team_options,
                    value="Spain",
                    clearable=False,
                    style={"flex": "1"},
                    className="dark-dropdown",
                ),
            ]),
        ]),
        html.Div(id="team-dive-content"),
    ])


@app.callback(Output("team-dive-content", "children"), Input("team-dropdown", "value"))
def update_team_dive(team_name):
    if not team_name:
        return html.Div()
    row = master[master["team"] == team_name]
    if row.empty:
        return html.Div("Team not found", style={"color": THEME["muted"]})
    row = row.iloc[0]
    arch_color = ARCHETYPE_COLORS.get(row["archetype"], THEME["accent"])

    wf_fig = make_team_waterfall(team_name)
    hm_fig = make_player_heatmap(team_name)

    pq_row = pq_df[pq_df["team"] == team_name]
    top_player = pq_row.iloc[0]["top_player"] if not pq_row.empty else "—"
    top_score = pq_row.iloc[0]["top_player_score"] if not pq_row.empty else "—"
    n_players = pq_row.iloc[0]["n_players_scored"] if not pq_row.empty else "—"
    confidence = pq_row.iloc[0]["player_coverage_confidence"] if not pq_row.empty else "—"

    return html.Div([
        # Team header
        card(style={"marginBottom": "20px", "borderLeft": f"4px solid {arch_color}"}, children=[
            html.Div(style={"display": "flex", "justifyContent": "space-between", "alignItems": "flex-start"}, children=[
                html.Div([
                    html.H2(team_name, style={"color": THEME["text"], "margin": "0 0 8px 0",
                                               "fontFamily": "DM Sans, sans-serif", "fontSize": "28px", "fontWeight": "700"}),
                    html.Div(style={"display": "flex", "gap": "10px", "alignItems": "center"}, children=[
                        html.Span(row["archetype"], style={
                            "backgroundColor": arch_color + "22", "color": arch_color,
                            "fontSize": "13px", "padding": "4px 12px", "borderRadius": "20px",
                            "fontFamily": "DM Sans, sans-serif", "fontWeight": "600",
                        }),
                        html.Span(f"Group {row['group']}", style={
                            "color": THEME["muted"], "fontSize": "13px", "fontFamily": "DM Sans, sans-serif",
                        }),
                        html.Span(f"Rank #{int(row['rank'])}", style={
                            "color": THEME["muted"], "fontSize": "13px", "fontFamily": "DM Sans, sans-serif",
                        }),
                    ]),
                ]),
                html.Div(style={"textAlign": "right"}, children=[
                    html.Div(f"{row['readiness_score']:.1f}", style={
                        "color": THEME["accent"], "fontSize": "48px", "fontWeight": "700",
                        "fontFamily": "JetBrains Mono, monospace", "lineHeight": "1",
                    }),
                    html.Div("Readiness Score", style={"color": THEME["muted"], "fontSize": "12px",
                                                        "fontFamily": "DM Sans, sans-serif", "marginTop": "4px"}),
                ]),
            ]),
        ]),

        # Score breakdown + player info
        html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "20px", "marginBottom": "20px"}, children=[
            card([
                section_header("Score Decomposition", "How the readiness score was built layer by layer"),
                dcc.Graph(figure=wf_fig, config={"displayModeBar": False}),
            ]),
            card([
                section_header("Player Intelligence"),
                html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "12px"}, children=[
                    html.Div(style={"backgroundColor": THEME["card2"], "borderRadius": "10px", "padding": "16px"}, children=[
                        html.Div("⭐ Top Player", style={"color": THEME["muted"], "fontSize": "11px",
                                                         "textTransform": "uppercase", "letterSpacing": "1px",
                                                         "fontFamily": "DM Sans, sans-serif", "marginBottom": "8px"}),
                        html.Div(str(top_player), style={"color": THEME["text"], "fontWeight": "600",
                                                          "fontFamily": "DM Sans, sans-serif", "fontSize": "14px"}),
                        html.Div(f"Score: {top_score}", style={"color": THEME["accent"],
                                                                "fontFamily": "JetBrains Mono, monospace", "fontSize": "13px"}),
                    ]),
                    html.Div(style={"backgroundColor": THEME["card2"], "borderRadius": "10px", "padding": "16px"}, children=[
                        html.Div("Players Scored", style={"color": THEME["muted"], "fontSize": "11px",
                                                           "textTransform": "uppercase", "letterSpacing": "1px",
                                                           "fontFamily": "DM Sans, sans-serif", "marginBottom": "8px"}),
                        html.Div(str(n_players), style={"color": THEME["text"], "fontWeight": "700",
                                                          "fontFamily": "JetBrains Mono, monospace", "fontSize": "24px"}),
                        html.Div(f"Confidence: {float(confidence)*100:.0f}%" if confidence != "—" else "—",
                                 style={"color": THEME["muted"], "fontSize": "12px", "fontFamily": "DM Sans, sans-serif"}),
                    ]),
                    html.Div(style={"backgroundColor": THEME["card2"], "borderRadius": "10px", "padding": "16px"}, children=[
                        html.Div("Champion %", style={"color": THEME["muted"], "fontSize": "11px",
                                                       "textTransform": "uppercase", "letterSpacing": "1px",
                                                       "fontFamily": "DM Sans, sans-serif", "marginBottom": "8px"}),
                        html.Div(f"{row['p_champion']:.1f}%", style={"color": THEME["green"], "fontWeight": "700",
                                                                       "fontFamily": "JetBrains Mono, monospace", "fontSize": "24px"}),
                    ]),
                    html.Div(style={"backgroundColor": THEME["card2"], "borderRadius": "10px", "padding": "16px"}, children=[
                        html.Div("Group Exit %", style={"color": THEME["muted"], "fontSize": "11px",
                                                         "textTransform": "uppercase", "letterSpacing": "1px",
                                                         "fontFamily": "DM Sans, sans-serif", "marginBottom": "8px"}),
                        html.Div(f"{row['p_group_exit']:.1f}%", style={"color": THEME["red"], "fontWeight": "700",
                                                                         "fontFamily": "JetBrains Mono, monospace", "fontSize": "24px"}),
                    ]),
                ]),

                html.Div(style={"marginTop": "16px"}, children=[
                    section_header("Monte Carlo Outcome Distribution"),
                    *[html.Div(style={"marginBottom": "8px"}, children=[
                        html.Div(style={"display": "flex", "justifyContent": "space-between",
                                        "marginBottom": "4px"}, children=[
                            html.Span(label, style={"color": THEME["muted"], "fontSize": "12px",
                                                     "fontFamily": "DM Sans, sans-serif"}),
                            html.Span(f"{row.get(col, 0):.1f}%", style={"color": THEME["text"], "fontSize": "12px",
                                                                          "fontFamily": "JetBrains Mono, monospace"}),
                        ]),
                        html.Div(style={"backgroundColor": THEME["border"], "borderRadius": "4px", "height": "6px"}, children=[
                            html.Div(style={
                                "backgroundColor": color, "borderRadius": "4px", "height": "6px",
                                "width": f"{min(row.get(col, 0), 100)}%",
                            }),
                        ]),
                    ]) for label, col, color in [
                        ("🏆 Champion",     "p_champion",      THEME["green"]),
                        ("🥈 Runner-Up",    "p_runner_up",     THEME["accent"]),
                        ("4️⃣  Semi-Final",  "p_semi_final",    THEME["accent2"]),
                        ("8️⃣  Quarter-Final","p_quarter_final", "#A78BFA"),
                        ("⬇️  Group Exit",  "p_group_exit",    THEME["red"]),
                    ]],
                ]),
            ]),
        ]),

        # Player heatmap
        card([
            section_header("Player Trait Heatmap",
                "Individual player scores across 4 trait categories from StatsBomb club data (2021–25). Coverage varies — not all players are included."),
            html.Div(
                dcc.Graph(figure=hm_fig, config={"displayModeBar": False})
                if hm_fig is not None
                else html.Div("No detailed player data available for this team.",
                              style={"color": THEME["muted"], "fontFamily": "DM Sans, sans-serif",
                                     "padding": "20px 0", "textAlign": "center"}),
            ),
        ]) if hm_fig is not None else card([
            section_header("Player Trait Heatmap"),
            html.Div("No detailed player data available for this team.",
                     style={"color": THEME["muted"], "fontFamily": "DM Sans, sans-serif",
                            "padding": "20px 0", "textAlign": "center"}),
        ]),
    ])


# ─────────────────────────────────────────────
#  TAB 4 — MONTE CARLO
# ─────────────────────────────────────────────
def monte_carlo_layout():
    return html.Div([
        card(style={"marginBottom": "20px"}, children=[
            section_header("10,000 Simulations of the Unmeasurable",
                "We ran the tournament 10,000 times. The field is genuinely open — Spain, France, and Argentina sit between 7–8.5% each. "
                "High Press / High Output teams hold 6 of the top 7 spots. Not coincidence — that is the outcome validation finding from the tactical clustering playing out at scale across 10,000 tournaments."),
            dcc.Graph(figure=make_mc_bars(), config={"displayModeBar": False},
                      style={"height": "1200px"}),
        ]),

        # Group view
        card([
            section_header("The Draw Matters Too", "Champion probability by group — where the measurement meets tournament bracket reality"),
            html.Div(style={
                "display": "grid", "gridTemplateColumns": "repeat(4, 1fr)", "gap": "12px"
            }, children=[
                html.Div(style={
                    "backgroundColor": THEME["card2"], "borderRadius": "10px",
                    "padding": "14px", "border": f"1px solid {THEME['border']}",
                }, children=[
                    html.Div(f"Group {group}", style={
                        "color": THEME["accent"], "fontWeight": "700", "fontSize": "14px",
                        "fontFamily": "DM Sans, sans-serif", "marginBottom": "10px",
                    }),
                    *[html.Div(style={
                        "display": "flex", "justifyContent": "space-between",
                        "alignItems": "center", "padding": "5px 0",
                        "borderBottom": f"1px solid {THEME['border']}22",
                    }, children=[
                        html.Span(row["team"], style={"color": THEME["text"], "fontSize": "12px",
                                                       "fontFamily": "DM Sans, sans-serif"}),
                        html.Span(f"{row['p_champion']:.1f}%", style={
                            "color": THEME["green"], "fontFamily": "JetBrains Mono, monospace",
                            "fontSize": "12px", "fontWeight": "600",
                        }),
                    ]) for _, row in master[master["group"] == group].sort_values("p_champion", ascending=False).iterrows()],
                ])
                for group in sorted(master["group"].unique())
            ]),
        ]),
    ])


# ─────────────────────────────────────────────
#  CSS injection for dropdowns
# ─────────────────────────────────────────────
app.index_string = '''
<!DOCTYPE html>
<html>
<head>
    {%metas%}
    <title>{%title%}</title>
    {%favicon%}
    {%css%}
    <style>
        * { box-sizing: border-box; }
        body { margin: 0; padding: 0; }
        ::-webkit-scrollbar { width: 6px; height: 6px; }
        ::-webkit-scrollbar-track { background: #0D0E1F; }
        ::-webkit-scrollbar-thumb { background: #2A2C4A; border-radius: 3px; }
        ::-webkit-scrollbar-thumb:hover { background: #F49D52; }

        .Select-control, .Select-menu-outer, .VirtualizedSelectFocusedOption,
        .VirtualizedSelectOption {
            background-color: #161729 !important;
            border-color: #2A2C4A !important;
            color: #FFFFFF !important;
        }
        .Select-value-label { color: #FFFFFF !important; }
        .Select-placeholder { color: #8B8FA8 !important; }
        .Select-arrow { border-top-color: #8B8FA8 !important; }
        .Select--single > .Select-control .Select-value { color: #FFFFFF !important; }
        .Select-menu { background-color: #161729 !important; }
        .Select-option { background-color: #161729 !important; color: #FFFFFF !important; }
        .Select-option:hover, .Select-option.is-focused {
            background-color: #1E2038 !important;
        }
        .Select-option.is-selected { background-color: #F49D5222 !important; color: #F49D52 !important; }
        .Select-input > input { color: #FFFFFF !important; }

        .tab--selected { border-color: #F49D52 !important; }
    </style>
</head>
<body>
    {%app_entry%}
    <footer>
        {%config%}
        {%scripts%}
        {%renderer%}
    </footer>
</body>
</html>
'''

def make_signal_disagreement():
    """
    For each team, compute rank by: readiness_score, player_quality_score, final_score (tactical).
    Disagreement = std of the three ranks. High std = model is uncertain / signals conflict.
    """
    df = master.copy()
    df["readiness_rank"]  = df["readiness_score"].rank(ascending=False)
    df["player_rank"]     = df["player_quality_score"].rank(ascending=False)
    df["tactical_rank"]   = df["final_score"].rank(ascending=False)

    df["rank_std"] = df[["readiness_rank","player_rank","tactical_rank"]].std(axis=1)
    df = df.sort_values("rank_std", ascending=False).head(15)

    # Gap labels: best signal vs worst signal per team
    def gap_label(row):
        ranks = {"Readiness": row["readiness_rank"],
                 "Player Quality": row["player_rank"],
                 "Tactical": row["tactical_rank"]}
        ranks = {k: v for k, v in ranks.items() if pd.notna(v)}
        if not ranks:
            return "—"
        best  = min(ranks, key=ranks.get)
        worst = max(ranks, key=ranks.get)
        return f"{best} ↑  vs  {worst} ↓"

    df["gap_label"] = df.apply(gap_label, axis=1)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=df["team"], x=df["readiness_rank"],
        name="Readiness Rank", orientation="h",
        marker_color=THEME["accent"],
        hovertemplate="<b>%{y}</b><br>Readiness rank: #%{x:.0f}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        y=df["team"], x=df["player_rank"],
        name="Player Quality Rank", orientation="h",
        marker_color=THEME["green"],
        hovertemplate="<b>%{y}</b><br>Player quality rank: #%{x:.0f}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        y=df["team"], x=df["tactical_rank"],
        name="Tactical Rank", orientation="h",
        marker_color=THEME["accent2"],
        hovertemplate="<b>%{y}</b><br>Tactical rank: #%{x:.0f}<extra></extra>",
    ))

    fig.update_layout(**PLOTLY_LAYOUT)
    fig.update_layout(
        barmode="group",
        xaxis_title="Rank (lower = better)",
        xaxis=dict(autorange="reversed"),
        yaxis=dict(autorange="reversed"),
        legend=dict(orientation="h", y=1.08, x=0, bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
        height=420,
        margin=dict(l=130, r=20, t=50, b=40),
    )
    return fig, df[["team","gap_label","rank_std"]].reset_index(drop=True)


# ─────────────────────────────────────────────
#  TAB 5 — ABOUT & LIMITATIONS
# ─────────────────────────────────────────────
def about_layout():
    def limit_card(number, title, body, color=None):
        c = color or THEME["yellow"]
        return html.Div(style={
            "backgroundColor": THEME["card2"],
            "borderRadius": "12px",
            "border": f"1px solid {THEME['border']}",
            "borderLeft": f"4px solid {c}",
            "padding": "20px 24px",
            "marginBottom": "16px",
        }, children=[
            html.Div(style={"display": "flex", "alignItems": "flex-start", "gap": "16px"}, children=[
                html.Div(f"L{number}", style={
                    "color": c, "fontFamily": "JetBrains Mono, monospace",
                    "fontWeight": "700", "fontSize": "20px", "minWidth": "32px",
                    "marginTop": "2px",
                }),
                html.Div([
                    html.Div(title, style={
                        "color": THEME["text"], "fontWeight": "700",
                        "fontSize": "15px", "fontFamily": "DM Sans, sans-serif",
                        "marginBottom": "8px",
                    }),
                    html.Div(body, style={
                        "color": THEME["muted"], "fontSize": "13px",
                        "fontFamily": "DM Sans, sans-serif", "lineHeight": "1.7",
                    }),
                ]),
            ]),
        ])

    def section_block(icon, title, children):
        return card(style={"marginBottom": "20px"}, children=[
            html.Div(style={"display": "flex", "alignItems": "center", "gap": "10px", "marginBottom": "20px",
                            "paddingBottom": "16px", "borderBottom": f"1px solid {THEME['border']}"}, children=[
                html.Span(icon, style={"fontSize": "22px"}),
                html.H3(title, style={"color": THEME["text"], "margin": "0",
                                      "fontSize": "18px", "fontWeight": "700",
                                      "fontFamily": "DM Sans, sans-serif"}),
            ]),
            *children,
        ])

    def inline_tag(text, color):
        return html.Span(text, style={
            "backgroundColor": color + "22", "color": color,
            "fontSize": "11px", "padding": "2px 8px", "borderRadius": "12px",
            "fontFamily": "DM Sans, sans-serif", "fontWeight": "600",
            "marginRight": "6px", "whiteSpace": "nowrap",
        })

    def bullet(text):
        return html.Div(style={"display": "flex", "gap": "10px", "marginBottom": "8px"}, children=[
            html.Span("·", style={"color": THEME["accent"], "fontWeight": "700", "flexShrink": "0"}),
            html.Span(text, style={"color": THEME["muted"], "fontSize": "13px",
                                   "fontFamily": "DM Sans, sans-serif", "lineHeight": "1.6"}),
        ])

    return html.Div([

        # ── What this is ──
        section_block("📖", "What This Framework Is", [
            
            html.P([
                "The 2026 FIFA World Cup is the most complex edition ever staged. 48 nations. Three host confederations. "
                "A genuinely open competitive field. And — as always — an enormous industry of predictions, most of which "
                "quietly ignore the question underneath all the others: ",
                html.Em("how do you actually measure a national team's readiness before a ball is kicked?",
                style={"color": THEME["text"]}),
                " This is our attempt to answer that. Built entirely from StatsBomb open data by ",
                html.Strong("Team XOH — Soomi Oh & Yoo Mi Oh", style={"color": THEME["accent"]}),
                " as a GT OMSA Capstone project."
            ], style={"color": THEME["muted"], "fontSize": "14px",
                      "fontFamily": "DM Sans, sans-serif", "lineHeight": "1.8", "margin": "0 0 16px 0"}),

            html.Div(style={"display": "grid", "gridTemplateColumns": "repeat(3, 1fr)", "gap": "12px"}, children=[
                html.Div(style={"backgroundColor": THEME["card"], "borderRadius": "10px",
                                "padding": "16px", "border": f"1px solid {THEME['border']}"}, children=[
                    html.Div("Step 1 — Tactical Identity", style={"color": THEME["accent2"], "fontWeight": "700",
                        "fontSize": "13px", "fontFamily": "DM Sans, sans-serif", "marginBottom": "8px"}),
                    html.Div("KMeans clustering on 8 tactical metrics across 398 matches from WC 2022, Euro 2024, "
                             "Copa América 2024, and AFCON 2023. 69 teams grouped into 6 style archetypes.",
                        style={"color": THEME["muted"], "fontSize": "12px",
                               "fontFamily": "DM Sans, sans-serif", "lineHeight": "1.6"}),
                ]),
                html.Div(style={"backgroundColor": THEME["card"], "borderRadius": "10px",
                                "padding": "16px", "border": f"1px solid {THEME['border']}"}, children=[
                    html.Div("Step 2 — Player Quality", style={"color": THEME["green"], "fontWeight": "700",
                        "fontSize": "13px", "fontFamily": "DM Sans, sans-serif", "marginBottom": "8px"}),
                    html.Div("13 metrics across 4 trait categories scored within positional archetypes, "
                             "using 5 seasons of club data (2021–25) with time decay and Bayesian shrinkage.",
                        style={"color": THEME["muted"], "fontSize": "12px",
                               "fontFamily": "DM Sans, sans-serif", "lineHeight": "1.6"}),
                ]),
                html.Div(style={"backgroundColor": THEME["card"], "borderRadius": "10px",
                                "padding": "16px", "border": f"1px solid {THEME['border']}"}, children=[
                    html.Div("Step 3 — Readiness + Simulation", style={"color": THEME["accent"], "fontWeight": "700",
                        "fontSize": "13px", "fontFamily": "DM Sans, sans-serif", "marginBottom": "8px"}),
                    html.Div("8-component composite score combining tactical + player + managerial stability + "
                             "host advantage. 10,000 Monte Carlo tournament simulations on top.",
                        style={"color": THEME["muted"], "fontSize": "12px",
                               "fontFamily": "DM Sans, sans-serif", "lineHeight": "1.6"}),
                ]),
            ]),
        ]),

        # ── What it is NOT ──
        section_block("🚫", "What This Framework Is Not", [
            html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "12px"}, children=[
                html.Div(style={"backgroundColor": THEME["card"], "borderRadius": "10px", "padding": "16px",
                                "border": f"1px solid {THEME['border']}"}, children=[
                    inline_tag("NOT this", THEME["red"]),
                    html.Div(text, style={"color": THEME["muted"], "fontSize": "13px",
                                         "fontFamily": "DM Sans, sans-serif", "lineHeight": "1.6",
                                         "marginTop": "8px"}),
                ])
                for text in [
                    "A prediction. The readiness score is a point estimate under uncertainty. The Monte Carlo output is a probability distribution, not a forecast.",
                    "A quality ranking in disguise. Archetypes describe style, not ability. Argentina and Namibia share High Press / High Output — they differ by player score, not archetype.",
                    "A complete picture. Injuries, squad form, referee luck, and tournament bracket draw are not modelled. Real tournaments are noisier than any model.",
                    "A claim about individual players. Palacios ranking above Messi in our data is a known artefact of intra-archetype percentiles and StatsBomb coverage — not a genuine footballing claim.",
                ]
            ]),
        ]),

        # ── Data Limitations ──
        section_block("🗄️", "Data Limitations", [
            limit_card(1, "Tournament Context Not Weighted at Clustering Level",
                "All match-level metrics are treated equally regardless of whether they came from WC 2022, Euro 2024, "
                "Copa América 2024, or AFCON 2023. A team pressing weak AFCON opposition produces the same PPDA as "
                "pressing a World Cup finalist. Purely-AFCON teams (Nigeria, DR Congo, Egypt) may cluster into "
                "higher-scoring archetypes than their WC-level quality justifies. Partial correction is applied via "
                "the non-WC HP/HO cap and the wc_presence_weight discount, but the fundamental clustering is based "
                "on a shared metric space across all tournaments.",
                THEME["yellow"]),
            limit_card(2, "Style Taxonomy ≠ Quality Ranking — Italy and Belgium",
                "Italy (FIFA #11) and Belgium (FIFA #7) score below their standing because both play styles that "
                "historically underperform at World Cups (Moderate Possession and Mid-Block Reactive), and both had "
                "poor WC 2022 outcomes (Italy absent, Belgium group stage). The model correctly prices their current "
                "tactical reality — these are not errors. Player quality substantially lifts both in the composite "
                "Readiness Score, which is exactly what the two-layer framework predicts.",
                THEME["accent2"]),
            limit_card(3, "Small Sample Teams (3 Matches)",
                "16 teams have only 3 tournament matches — a single group stage. Their tactical profiles are based on "
                "a very thin sample and may not represent their true style. The sample_weight floor (0.73 for 3 matches) "
                "partially discounts them, but they remain the lowest-confidence tactical assignments in the framework.",
                THEME["red"]),
            limit_card(4, "StatsBomb Club Coverage Skews European",
                "Player scoring relies on StatsBomb open data which covers European club competitions far more "
                "comprehensively than African, Asian, and CONCACAF leagues. The median coverage confidence across "
                "all 48 nations is 0.18 — the typical country has fewer than 2 players scored out of an 11-player "
                "baseline. Only 7 countries exceed 0.8 confidence. Confidence tiers: Full (1.0) — France, Spain, "
                "Germany, Brazil, Argentina. Strong (0.8–0.95) — England, Netherlands, Portugal, Croatia. "
                "Partial (0.6–0.8) — Colombia, Canada, Ecuador, Uruguay. Thin or near-zero — everyone else. "
                "Ghana has no player quality score at all.",
                THEME["yellow"]),
            limit_card(5, "9 Teams Have No Tactical Data",
                "Norway, Sweden, Bosnia & Herzegovina, Iraq, Jordan, Uzbekistan, Haiti, New Zealand, and Curaçao "
                "do not appear in any of the 4 StatsBomb tournament datasets. They receive no tactical archetype "
                "assignment and are scored on player quality only, with a confidence penalty applied.",
                THEME["muted"]),
            limit_card(6, "Monte Carlo Match Model Is Simplified",
                "Each match is simulated using a logistic function on readiness score difference only (σ=15, "
                "calibrated so a 10-point gap → ~75% win probability). There are no draw mechanics beyond the "
                "formula, no injury simulation, no squad rotation modelling, and no bracket seeding effects "
                "beyond the actual WC 2026 group draw.",
                THEME["accent"]),

            limit_card(7, "Player Detail Records Cover 38 of 48 Nations",
                "The player trait heatmap in Team Deep Dive is only available for nations whose players appear "
                "in the 2026 WC roster data cross-referenced with StatsBomb club coverage. 10 nations have no "
                "individual player records — their composite readiness score still includes a player quality "
                "signal via the FIFA fallback mechanism, but no trait-level breakdown is available. "
                "Of the 500+ players scored in the full pipeline, only the ~165 confirmed rostered players "
                "are shown here — the rest were scored for analysis but did not make final squads.",
    THEME["muted"]),
        ]),

        # ── Key methodological choices ──
        section_block("⚙️", "Key Methodological Choices", [
            html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "16px"}, children=[
                html.Div([
                    html.Div(title, style={"color": THEME["text"], "fontWeight": "600", "fontSize": "14px",
                                           "fontFamily": "DM Sans, sans-serif", "marginBottom": "6px"}),
                    html.Div(body, style={"color": THEME["muted"], "fontSize": "13px",
                                         "fontFamily": "DM Sans, sans-serif", "lineHeight": "1.6"}),
                ], style={"backgroundColor": THEME["card"], "borderRadius": "10px",
                          "padding": "16px", "border": f"1px solid {THEME['border']}"})
                for title, body in [
                    ("Georgia & Slovenia excluded from clustering",
                     "Both teams had extreme PPDA outliers (>40) that distorted cluster geometry. Neither qualifies for WC 2026, so exclusion has no downstream effect on readiness scores."),
                    ("KMeans k=6 selected over k=2",
                     "k=2 wins on pure statistical grounds — the data is a continuum. k=6 was chosen because it produces tactically interpretable, stable groupings that map to real football concepts."),
                    ("GMM blend for boundary teams",
                     "Teams with GMM confidence < 1.0 sit genuinely between archetypes. Their tactical score is a probability-weighted blend of both archetype scores rather than a hard assignment."),
                    ("Intra-archetype percentiles for players",
                     "Players are ranked within their position archetype, not globally. A defensive midfielder is judged against other DMs — not against Mbappe. This prevents position bias in scoring."),
                    ("270-minute minimum threshold",
                     "Players with fewer than 270 minutes receive Bayesian shrinkage toward the archetype mean. This prevents small-sample cameos from inflating or deflating scores."),
                    ("Time decay across seasons",
                     "2023/24 season weighted highest, 2022/23 next, 2021/22 lowest. Reflects the assumption that recent club form is more predictive of 2026 performance than form from 3 years ago."),
                ]
            ]),
        ]),

        # Footer
        html.Div(style={
            "textAlign": "center", "padding": "24px 0 8px",
            "borderTop": f"1px solid {THEME['border']}",
            "color": THEME["muted"], "fontSize": "12px",
            "fontFamily": "DM Sans, sans-serif",
        }, children=[
            html.P("Team XOH — Soomi Oh, Yoo Mi Oh · GT OMSA Capstone · April 2026",
                   style={"margin": "0 0 4px 0"}),
            html.P("Data: StatsBomb Open Data, FIFA Offical Websites and Guardians 2025 top 100 football players. All scores and probabilities are model outputs, not official forecasts.",
                   style={"margin": "0"}),
        ]),
    ])


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  WC 2026 Readiness Dashboard — Team XOH")
    print("  http://127.0.0.1:8050")
    print("=" * 60 + "\n")
    app.run(debug=True, host="127.0.0.1", port=8050)
