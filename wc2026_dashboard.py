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
        marker=dict(colors=colors, line=dict(color=THEME["bg"], width=3)),
        textinfo="value",
        textposition="outside",
        textfont=dict(size=12, color=THEME["text"], family="JetBrains Mono, monospace"),
        hovertemplate="<b>%{label}</b><br>Teams: %{value}<br>Avg Readiness: %{customdata:.1f}<extra></extra>",
        customdata=counts["AvgReadiness"],
        showlegend=True,
    ))
    fig.update_layout(**PLOTLY_LAYOUT)
    fig.update_layout(
        showlegend=True,
        margin=dict(l=40, r=160, t=40, b=40),
        height=320,
        legend=dict(
            orientation="v",
            x=1.02, y=0.5,
            xanchor="left",
            yanchor="middle",
            font=dict(size=11, color=THEME["muted"], family="DM Sans, sans-serif"),
            bgcolor="rgba(0,0,0,0)",
        ),
        annotations=[dict(
            text="<b>48</b><br><span style='font-size:10px'>nations</span>",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color=THEME["text"], family="JetBrains Mono, monospace"),
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

WATERFALL_EXPLANATIONS = {
    "Base Archetype":        "Starting score assigned to this team's tactical style, derived from WC 2022 outcome validation.",
    "GMM Blend":             "Adjustment for teams sitting between two archetypes. Zero means a clean single-archetype assignment.",
    "Evidence Discount":     "Penalty for thin data — teams with fewer matches or no WC presence get discounted. Reflects lower confidence in the tactical signal.",
    "Quality Adj":           "Fine-tuning within the archetype based on npxG and EPR relative to archetype peers. ±4 points max.",
    "Tactical Score":        "Final tactical component after all four layers. Feeds into the composite at 20% weight.",
    "Player + Context Lift": "Contribution from player quality (35%), FIFA ranking (15%), managerial stability (10%), host advantage (10%), and squad age (10%).",
    "Readiness Score":       "The composite score. Weighted sum of all eight components across tactical, player, and contextual signals.",
}

def make_comparison_chart(primary_team, compare_teams):
    teams = [primary_team] + compare_teams
    components = {
        "Readiness":      "readiness_score",
        "Player Quality": "player_quality_score",
        "Tactical":       "final_score",
        "Champion %":     "p_champion",
        "Group Exit %":   "p_group_exit",
    }
    colors = [THEME["accent"], THEME["accent2"], THEME["green"], "#A78BFA"]

    fig = go.Figure()
    for i, team in enumerate(teams):
        t_row = master[master["team"] == team]
        if t_row.empty:
            continue
        t_row = t_row.iloc[0]
        fig.add_trace(go.Bar(
            name=team,
            x=list(components.keys()),
            y=[float(t_row.get(col, 0)) if pd.notna(t_row.get(col)) else 0
               for col in components.values()],
            marker_color=colors[i % len(colors)],
            hovertemplate="<b>" + team + "</b><br>%{x}: %{y:.1f}<extra></extra>",
        ))

    fig.update_layout(**PLOTLY_LAYOUT)
    fig.update_layout(
        barmode="group",
        height=320,
        margin=dict(l=40, r=20, t=20, b=40),
        legend=dict(orientation="h", y=1.08, x=0, bgcolor="rgba(0,0,0,0)", font=dict(size=12)),
        yaxis=dict(gridcolor=THEME["border"]),
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

    players["final_score_num"] = pd.to_numeric(players["final_score"], errors="coerce").fillna(0)
    final_col = players["final_score_num"].values.reshape(-1, 1)
    z = np.hstack([players[trait_cols].fillna(0).values, final_col])
    labels_x = ["Final Third", "Progression", "Control", "Mobility/Intensity", "Overall"]

    labels_y = [f"[{r.get('coverage_tier','?')}]  {r['player']} · {r['position_archetype']}"
                for _, r in players.iterrows()]
    

    fig = go.Figure(go.Heatmap(
        z=z, x=labels_x, y=labels_y,
        colorscale=[[0, "#1E2038"], [0.5, "#759ACE"], [1, "#F49D52"]],
        zmin=0, zmax=100,
        text=[[f"{v:.0f}" if not np.isnan(v) else "" for v in row] for row in z],
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
            section_header("Why These Boundaries Are Real But Not Rigid",
                "Bootstrap stability analysis across 100 resamples — what happens when you shuffle the teams and re-cluster"),
            html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr 1fr", "gap": "16px"}, children=[
                html.Div(style={"backgroundColor": THEME["card2"], "borderRadius": "10px", "padding": "16px",
                                "border": f"1px solid {THEME['border']}"}, children=[
                    html.Div("0 of 69", style={"color": THEME["accent"], "fontSize": "28px", "fontWeight": "700",
                                               "fontFamily": "JetBrains Mono, monospace", "marginBottom": "6px"}),
                    html.Div("teams exceed the 70% co-occurrence threshold in bootstrap resampling.",
                             style={"color": THEME["muted"], "fontSize": "13px", "fontFamily": "DM Sans, sans-serif",
                                    "lineHeight": "1.6"}),
                ]),
                html.Div(style={"backgroundColor": THEME["card2"], "borderRadius": "10px", "padding": "16px",
                                "border": f"1px solid {THEME['border']}"}, children=[
                    html.Div("k=2 wins", style={"color": THEME["accent2"], "fontSize": "28px", "fontWeight": "700",
                                                "fontFamily": "JetBrains Mono, monospace", "marginBottom": "6px"}),
                    html.Div("statistically with silhouette 0.294, nearly double any other k. This is the primary structural finding: international football tactics exist on a continuum. k=6 is chosen for interpretability, not statistical optimality.",
                             style={"color": THEME["muted"], "fontSize": "13px", "fontFamily": "DM Sans, sans-serif",
                                    "lineHeight": "1.6"}),
                ]),
                html.Div(style={"backgroundColor": THEME["card2"], "borderRadius": "10px", "padding": "16px",
                                "border": f"1px solid {THEME['border']}"}, children=[
                    html.Div("ARI 0.455", style={"color": THEME["green"], "fontSize": "28px", "fontWeight": "700",
                                                 "fontFamily": "JetBrains Mono, monospace", "marginBottom": "6px"}),
                    html.Div("GMM cross-validation score at k=6 — both models independently find similar groupings, confirming the structure is real.",
                             style={"color": THEME["muted"], "fontSize": "13px", "fontFamily": "DM Sans, sans-serif",
                                    "lineHeight": "1.6"}),
                ]),
            ]),
            html.Div("k=6 was chosen not because it wins statistically, but because it produces tactically interpretable, stable groupings that map to real football concepts. "
                     "The six archetypes are useful simplifications of a continuous space — not hard truths.",
                     style={"color": THEME["muted"], "fontSize": "12px", "fontFamily": "DM Sans, sans-serif",
                            "lineHeight": "1.6", "marginTop": "16px",
                            "borderTop": f"1px solid {THEME['border']}", "paddingTop": "12px"}),
        ]),

    ])


# ─────────────────────────────────────────────
#  TAB 3 — TEAM DEEP DIVE
# ─────────────────────────────────────────────
def team_dive_layout():
    team_options = [{"label": f"{r['team']} (Group {r['group']})", "value": r["team"]}
                    for _, r in master.sort_values("readiness_score", ascending=False).iterrows()]
    compare_options = [{"label": r["team"], "value": r["team"]}
                       for _, r in master.sort_values("readiness_score", ascending=False).iterrows()]
    return html.Div([
        card(style={"marginBottom": "20px"}, children=[
            html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "20px",
                            "alignItems": "center"}, children=[
                html.Div(style={"display": "flex", "alignItems": "center", "gap": "12px"}, children=[
                    html.Div("Team:", style={"color": THEME["text"], "fontWeight": "600",
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
                html.Div(style={"display": "flex", "alignItems": "center", "gap": "12px"}, children=[
                    html.Div("Compare vs:", style={"color": THEME["muted"], "fontWeight": "600",
                                                    "fontFamily": "DM Sans, sans-serif",
                                                    "whiteSpace": "nowrap", "fontSize": "13px"}),
                    dcc.Dropdown(
                        id="compare-dropdown",
                        options=compare_options,
                        value=["France", "Argentina"],
                        multi=True,
                        clearable=True,
                        placeholder="Select teams to compare...",
                        style={"flex": "1"},
                        className="dark-dropdown",
                    ),
                ]),
            ]),
        ]),
        html.Div(id="team-dive-content"),
    ])


@app.callback(
    Output("compare-dropdown", "value"),
    Input("team-dropdown", "value"),
    prevent_initial_call=True,
)
def update_compare_defaults(team_name):
    """Auto-populate compare dropdown with same-archetype peers when team changes."""
    if not team_name:
        return ["France", "Argentina"]
    row = master[master["team"] == team_name]
    if row.empty:
        return ["France", "Argentina"]
    arch = row.iloc[0]["archetype"]
    peers = master[
        (master["archetype"] == arch) &
        (master["team"] != team_name)
    ].sort_values("readiness_score", ascending=False)["team"].head(3).tolist()
    return peers if peers else ["France", "Argentina"]


@app.callback(
    Output("team-dive-content", "children"),
    [Input("team-dropdown", "value"),
     Input("compare-dropdown", "value")]
)
def update_team_dive(team_name, compare_teams):
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
    confidence_raw = pq_row.iloc[0]["player_coverage_confidence"] if not pq_row.empty else 0
    confidence_val = float(confidence_raw) if confidence_raw != "—" else 0.0

    # Top player position from details
    top_player_pos = "—"
    top_player_tier = "—"
    mask = pd_df["team"].values == team_name
    top_player_detail = pd_df.iloc[mask.nonzero()[0]].copy().reset_index(drop=True)
    if not top_player_detail.empty:
        top_player_detail["final_score"] = pd.to_numeric(top_player_detail["final_score"], errors="coerce")
        best = top_player_detail.nlargest(1, "final_score")
        if not best.empty:
            top_player_pos = best.iloc[0].get("archetype_label", best.iloc[0].get("position_archetype", "—"))
            top_player_tier = best.iloc[0].get("coverage_tier", "—")

    # Confidence tier
    if confidence_val >= 0.8:
        conf_color = THEME["green"]
        conf_badge = "HIGH CONFIDENCE"
        conf_note = "Full data coverage — player and tactical signals both present and reliable."
    elif confidence_val >= 0.4:
        conf_color = THEME["yellow"]
        conf_badge = "PARTIAL CONFIDENCE"
        conf_note = f"Coverage {int(confidence_val*100)}% — directionally correct, not precise."
    else:
        conf_color = THEME["red"]
        conf_badge = "LOW CONFIDENCE"
        conf_note = f"Coverage {int(confidence_val*100)}% — largely FIFA fallback. Treat with caution."

    # Story callout
    arch = row["archetype"]
    rank = int(row["rank"])
    pq_score = float(pq_row["player_quality_score"].values[0]) if not pq_row.empty and pd.notna(pq_row["player_quality_score"].values[0]) else None
    pq_rank = int(master["player_quality_score"].rank(ascending=False)[master["team"] == team_name].values[0]) if pq_score else None
    tac_score = float(row["final_score"]) if pd.notna(row.get("final_score")) else None
    tac_rank = int(master["final_score"].rank(ascending=False)[master["team"] == team_name].values[0]) if tac_score else None

    arch_story = {
        "High Press / High Output": "plays the historically dominant WC style — the archetype that has produced every champion in the dataset",
        "Possession Dominant":      "controls games through possession but this archetype has historically underperformed at World Cups relative to its quality",
        "Compact Transition":       "built to absorb pressure and strike on the counter — effective against stronger opponents, vulnerable to sustained pressure",
        "Mid-Block Reactive":       "defends deep and plays on the break — historically the hardest archetype to win a tournament with",
        "Moderate Possession":      "a balanced but unspectacular style — neither the pressing intensity of the elite nor the defensive discipline of the reactive teams",
        "Low Intensity":            "low pressing, low possession — the archetype with the weakest historical WC record",
        "No Tactical Data":         "has no tactical data from the four tournaments used for clustering — score relies entirely on player quality and external signals",
    }.get(arch, "")

    if confidence_val >= 0.8 and pq_rank and tac_rank:
        if pq_rank < tac_rank - 5:
            tension = f"player quality ranks #{pq_rank} but tactical score ranks #{tac_rank} — strong squad in a historically limited system"
        elif tac_rank < pq_rank - 5:
            tension = f"tactical system ranks #{tac_rank} but player quality ranks #{pq_rank} — the system is ahead of the available players"
        else:
            tension = f"tactical and player signals are well-aligned (tactical #{tac_rank}, player quality #{pq_rank})"
        callout = f"Rank #{rank} of 48. {team_name} {arch_story}. {tension.capitalize()}."
    elif confidence_val >= 0.4:
        callout = f"Rank #{rank} of 48. Partial coverage ({int(confidence_val*100)}%). Archetype: {arch}. Directional ranking reliable; precise score less so."
    else:
        callout = f"Rank #{rank} of 48. Very limited coverage ({int(confidence_val*100)}%). Score is primarily FIFA fallback — treat with caution."

    # Monte Carlo context
    top_champ = master.nlargest(1, "p_champion").iloc[0]
    champ_context = f"vs #{int(top_champ['rank'])} {top_champ['team']} at {top_champ['p_champion']:.1f}%" \
                    if top_champ["team"] != team_name else "— top of the field"

    # Comparison chart
    compare_fig = None
    if compare_teams:
        compare_teams_clean = [t for t in (compare_teams if isinstance(compare_teams, list) else [compare_teams])
                               if t != team_name][:3]
        if compare_teams_clean:
            compare_fig = make_comparison_chart(team_name, compare_teams_clean)

    # Same-archetype context label
    arch_peers = master[
        (master["archetype"] == arch) & (master["team"] != team_name)
    ].sort_values("readiness_score", ascending=False)["team"].head(3).tolist()
    peer_label = f"Same archetype peers: {', '.join(arch_peers)}" if arch_peers else ""

    # Coverage tier colours
    tier_colors = {"A": THEME["green"], "B": THEME["accent2"], "C": THEME["yellow"], "D": THEME["red"]}
    tier_descs  = {
        "A": "2+ club seasons — full confidence",
        "B": "1 club season — good confidence",
        "C": "Tournament data only — lower confidence",
        "D": "Guardian list only — external proxy",
    }

    return html.Div([

        # ── Story callout banner ──
        html.Div(style={
            "backgroundColor": conf_color + "11",
            "border": f"1px solid {conf_color}44",
            "borderLeft": f"4px solid {conf_color}",
            "borderRadius": "12px", "padding": "16px 20px", "marginBottom": "20px",
            "display": "flex", "justifyContent": "space-between", "alignItems": "flex-start",
        }, children=[
            html.Div([
                html.Div(callout, style={"color": THEME["text"], "fontSize": "14px",
                                         "fontFamily": "DM Sans, sans-serif", "lineHeight": "1.7",
                                         "marginBottom": "6px"}),
                html.Div(conf_note, style={"color": THEME["muted"], "fontSize": "12px",
                                           "fontFamily": "DM Sans, sans-serif"}),
            ], style={"flex": "1"}),
            html.Div(conf_badge, style={
                "color": conf_color, "fontSize": "10px", "fontWeight": "700",
                "letterSpacing": "1px", "fontFamily": "DM Sans, sans-serif",
                "backgroundColor": conf_color + "22", "padding": "4px 10px",
                "borderRadius": "6px", "whiteSpace": "nowrap", "marginLeft": "16px",
            }),
        ]),

        # ── Team header ──
        card(style={"marginBottom": "20px", "borderLeft": f"4px solid {arch_color}"}, children=[
            html.Div(style={"display": "flex", "justifyContent": "space-between", "alignItems": "flex-start"}, children=[
                html.Div([
                    html.H2(team_name, style={"color": THEME["text"], "margin": "0 0 8px 0",
                                               "fontFamily": "DM Sans, sans-serif", "fontSize": "28px", "fontWeight": "700"}),
                    html.Div(style={"display": "flex", "gap": "10px", "alignItems": "center", "flexWrap": "wrap"}, children=[
                        html.Span(arch, style={
                            "backgroundColor": arch_color + "22", "color": arch_color,
                            "fontSize": "13px", "padding": "4px 12px", "borderRadius": "20px",
                            "fontFamily": "DM Sans, sans-serif", "fontWeight": "600",
                        }),
                        html.Span(f"Group {row['group']}", style={
                            "color": THEME["muted"], "fontSize": "13px", "fontFamily": "DM Sans, sans-serif"}),
                        html.Span(f"Rank #{rank} of 48", style={
                            "color": THEME["muted"], "fontSize": "13px", "fontFamily": "DM Sans, sans-serif"}),
                        html.Span(f"Coverage {int(confidence_val*100)}%", style={
                            "color": conf_color, "fontSize": "12px", "fontFamily": "JetBrains Mono, monospace",
                            "backgroundColor": conf_color + "15", "padding": "3px 8px", "borderRadius": "10px",
                        }),
                    ]),
                    html.Div(peer_label, style={"color": THEME["muted"], "fontSize": "11px",
                                                "fontFamily": "DM Sans, sans-serif", "marginTop": "8px"}) if peer_label else None,
                ]),
                html.Div(style={"textAlign": "right"}, children=[
                    html.Div(f"{row['readiness_score']:.1f}", style={
                        "color": THEME["accent"], "fontSize": "48px", "fontWeight": "700",
                        "fontFamily": "JetBrains Mono, monospace", "lineHeight": "1",
                    }),
                    html.Div("Readiness Score (0–100)", style={"color": THEME["muted"], "fontSize": "11px",
                                                               "fontFamily": "DM Sans, sans-serif", "marginTop": "4px"}),
                    html.Div("tactical + player + context", style={"color": THEME["muted"], "fontSize": "10px",
                                                                    "fontFamily": "DM Sans, sans-serif"}),
                ]),
            ]),
        ]),

        # ── Score breakdown + player info ──
        html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "20px", "marginBottom": "20px"}, children=[
            card([
                section_header("How We Built This Score",
                    "Every layer shown explicitly. Nothing hidden."),
                dcc.Graph(figure=wf_fig, config={"displayModeBar": False}),
                html.Div(style={"marginTop": "12px", "paddingTop": "12px",
                                "borderTop": f"1px solid {THEME['border']}"}, children=[
                    html.Div("What each layer means:", style={"color": THEME["muted"], "fontSize": "11px",
                                                               "fontFamily": "DM Sans, sans-serif",
                                                               "marginBottom": "6px", "fontWeight": "600"}),
                    *[html.Div(style={"display": "flex", "gap": "8px", "marginBottom": "4px"}, children=[
                        html.Span(f"{k}:", style={"color": THEME["accent2"], "fontSize": "11px",
                                                   "fontFamily": "DM Sans, sans-serif", "fontWeight": "600",
                                                   "minWidth": "150px", "flexShrink": "0"}),
                        html.Span(v, style={"color": THEME["muted"], "fontSize": "11px",
                                            "fontFamily": "DM Sans, sans-serif", "lineHeight": "1.5"}),
                    ]) for k, v in WATERFALL_EXPLANATIONS.items()],
                ]),
            ]),
            card([
                section_header("Player Quality",
                    "Scored within position archetypes from 3 seasons of club data"),

                # Top player card
                html.Div(style={"backgroundColor": THEME["card2"], "borderRadius": "10px",
                                "padding": "16px", "marginBottom": "12px"}, children=[
                    html.Div(style={"display": "flex", "justifyContent": "space-between",
                                    "alignItems": "flex-start"}, children=[
                        html.Div([
                            html.Div("⭐ Top Player", style={"color": THEME["muted"], "fontSize": "11px",
                                                             "textTransform": "uppercase", "letterSpacing": "1px",
                                                             "fontFamily": "DM Sans, sans-serif", "marginBottom": "6px"}),
                            html.Div(str(top_player), style={"color": THEME["text"], "fontWeight": "600",
                                                              "fontFamily": "DM Sans, sans-serif", "fontSize": "15px"}),
                            html.Div(style={"display": "flex", "gap": "8px", "marginTop": "4px",
                                            "alignItems": "center"}, children=[
                                html.Span(f"Position: {top_player_pos}", style={
                                    "color": THEME["muted"], "fontSize": "12px",
                                    "fontFamily": "DM Sans, sans-serif"}),
                                html.Span(f"Tier {top_player_tier}", style={
                                    "color": tier_colors.get(top_player_tier, THEME["muted"]),
                                    "fontSize": "11px", "fontFamily": "DM Sans, sans-serif",
                                    "backgroundColor": tier_colors.get(top_player_tier, THEME["muted"]) + "22",
                                    "padding": "2px 6px", "borderRadius": "6px",
                                }) if top_player_tier != "—" else None,
                            ]),
                            html.Div(tier_descs.get(top_player_tier, ""), style={
                                "color": THEME["muted"], "fontSize": "11px",
                                "fontFamily": "DM Sans, sans-serif", "marginTop": "2px",
                            }) if top_player_tier in tier_descs else None,
                        ]),
                        html.Div(str(top_score), style={
                            "color": THEME["accent"], "fontWeight": "700",
                            "fontFamily": "JetBrains Mono, monospace", "fontSize": "28px",
                        }),
                    ]),
                ]),

                # Stats grid
                html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "12px",
                                "marginBottom": "16px"}, children=[
                    html.Div(style={"backgroundColor": THEME["card2"], "borderRadius": "10px", "padding": "14px"}, children=[
                        html.Div("Players Scored", style={"color": THEME["muted"], "fontSize": "11px",
                                                           "textTransform": "uppercase", "letterSpacing": "1px",
                                                           "fontFamily": "DM Sans, sans-serif", "marginBottom": "6px"}),
                        html.Div(str(n_players), style={"color": THEME["text"], "fontWeight": "700",
                                                         "fontFamily": "JetBrains Mono, monospace", "fontSize": "22px"}),
                        html.Div(f"Data confidence: {int(confidence_val*100)}%",
                                 style={"color": conf_color, "fontSize": "11px",
                                        "fontFamily": "DM Sans, sans-serif"}),
                    ]),
                    html.Div(style={"backgroundColor": THEME["card2"], "borderRadius": "10px", "padding": "14px"}, children=[
                        html.Div("Champion %", style={"color": THEME["muted"], "fontSize": "11px",
                                                       "textTransform": "uppercase", "letterSpacing": "1px",
                                                       "fontFamily": "DM Sans, sans-serif", "marginBottom": "6px"}),
                        html.Div(f"{row['p_champion']:.1f}%", style={"color": THEME["green"], "fontWeight": "700",
                                                                       "fontFamily": "JetBrains Mono, monospace",
                                                                       "fontSize": "22px"}),
                        html.Div(champ_context, style={"color": THEME["muted"], "fontSize": "11px",
                                                        "fontFamily": "DM Sans, sans-serif"}),
                    ]),
                    html.Div(style={"backgroundColor": THEME["card2"], "borderRadius": "10px", "padding": "14px"}, children=[
                        html.Div("Group Exit %", style={"color": THEME["muted"], "fontSize": "11px",
                                                         "textTransform": "uppercase", "letterSpacing": "1px",
                                                         "fontFamily": "DM Sans, sans-serif", "marginBottom": "6px"}),
                        html.Div(f"{row['p_group_exit']:.1f}%", style={"color": THEME["red"], "fontWeight": "700",
                                                                         "fontFamily": "JetBrains Mono, monospace",
                                                                         "fontSize": "22px"}),
                        html.Div(f"Group {row['group']} draw", style={"color": THEME["muted"], "fontSize": "11px",
                                                                        "fontFamily": "DM Sans, sans-serif"}),
                    ]),
                    html.Div(style={"backgroundColor": THEME["card2"], "borderRadius": "10px", "padding": "14px"}, children=[
                        html.Div("Semi-Final %", style={"color": THEME["muted"], "fontSize": "11px",
                                                         "textTransform": "uppercase", "letterSpacing": "1px",
                                                         "fontFamily": "DM Sans, sans-serif", "marginBottom": "6px"}),
                        html.Div(f"{row.get('p_semi_final', 0):.1f}%", style={
                            "color": THEME["accent2"], "fontWeight": "700",
                            "fontFamily": "JetBrains Mono, monospace", "fontSize": "22px"}),
                        html.Div(f"from 10,000 simulations", style={"color": THEME["muted"], "fontSize": "11px",
                                                                      "fontFamily": "DM Sans, sans-serif"}),
                    ]),
                ]),

                section_header("Outcome Distribution", "How often each stage was reached"),
                *[html.Div(style={"marginBottom": "8px"}, children=[
                    html.Div(style={"display": "flex", "justifyContent": "space-between",
                                    "marginBottom": "4px"}, children=[
                        html.Span(label, style={"color": THEME["muted"], "fontSize": "12px",
                                                 "fontFamily": "DM Sans, sans-serif"}),
                        html.Span(f"{row.get(col, 0):.1f}%", style={"color": THEME["text"], "fontSize": "12px",
                                                                      "fontFamily": "JetBrains Mono, monospace"}),
                    ]),
                    html.Div(style={"backgroundColor": THEME["border"], "borderRadius": "4px", "height": "6px"}, children=[
                        html.Div(style={"backgroundColor": color, "borderRadius": "4px", "height": "6px",
                                        "width": f"{min(row.get(col, 0), 100)}%"}),
                    ]),
                ]) for label, col, color in [
                    ("🏆 Champion",      "p_champion",      THEME["green"]),
                    ("🥈 Runner-Up",     "p_runner_up",     THEME["accent"]),
                    ("4️⃣  Semi-Final",   "p_semi_final",    THEME["accent2"]),
                    ("8️⃣  Quarter-Final","p_quarter_final", "#A78BFA"),
                    ("⬇️  Group Exit",   "p_group_exit",    THEME["red"]),
                ]],
            ]),
        ]),

        # ── Comparison panel ──
        card(style={"marginBottom": "20px"}, children=[
            section_header("How Does This Compare?",
                f"Key components vs selected teams. Compare dropdown auto-populates with same-archetype peers ({arch})."),
            dcc.Graph(figure=compare_fig, config={"displayModeBar": False})
            if compare_fig is not None
            else html.Div("Select teams to compare using the dropdown above.",
                          style={"color": THEME["muted"], "fontFamily": "DM Sans, sans-serif",
                                 "padding": "20px 0", "textAlign": "center", "fontSize": "13px"}),
        ]),

        # ── Player heatmap ──
        card([
            section_header("Player Trait Heatmap",
                f"{n_players} rostered players · 4 trait categories · sorted by final score"),
            # Coverage tier legend
            html.Div(style={"display": "flex", "gap": "16px", "marginBottom": "16px", "flexWrap": "wrap"}, children=[
                html.Div(style={"display": "flex", "alignItems": "center", "gap": "6px"}, children=[
                    html.Span(f"[{tier}]", style={"color": tier_colors[tier], "fontFamily": "JetBrains Mono, monospace",
                                                   "fontSize": "12px", "fontWeight": "700"}),
                    html.Span(desc, style={"color": THEME["muted"], "fontSize": "11px",
                                           "fontFamily": "DM Sans, sans-serif"}),
                ])
                for tier, desc in tier_descs.items()
            ]),
            html.Div(
                dcc.Graph(figure=hm_fig, config={"displayModeBar": False})
                if hm_fig is not None
                else html.Div("No detailed player data available for this team.",
                              style={"color": THEME["muted"], "fontFamily": "DM Sans, sans-serif",
                                     "padding": "20px 0", "textAlign": "center"}),
            ),
        ]),
    ])



# ─────────────────────────────────────────────
#  TAB 4 — MONTE CARLO
# ─────────────────────────────────────────────
def monte_carlo_layout():
    return html.Div([
        card(style={"marginBottom": "20px"}, children=[
            section_header("10,000 Simulations of the Unmeasurable",
                "Each team's Readiness Score (0–100 composite of tactical style, player quality, FIFA ranking, and context) was used as the input. "
                "We ran the tournament 10,000 times. The field is genuinely open — Spain, France, and Argentina sit between 7–8.5% each. "
                "High Press / High Output teams hold 6 of the top 7 spots. Not coincidence — that is the outcome validation finding from the tactical clustering playing out at scale across 10,000 tournaments."),
            dcc.Graph(figure=make_mc_bars(), config={"displayModeBar": False},
                      style={"height": "1200px"}),
        ]),

        # Group view

        card([
            section_header("Group Stage Overview", "Champion probability vs group exit risk per group"),
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
                    html.Div(style={"display": "flex", "justifyContent": "space-between",
                                    "marginBottom": "6px", "paddingBottom": "4px",
                                    "borderBottom": f"1px solid {THEME['border']}"}, children=[
                        html.Span("Team", style={"color": THEME["muted"], "fontSize": "10px",
                                                  "textTransform": "uppercase", "letterSpacing": "0.5px",
                                                  "fontFamily": "DM Sans, sans-serif"}),
                        html.Div(style={"display": "flex", "gap": "12px"}, children=[
                            html.Span("Win %", style={"color": THEME["green"], "fontSize": "10px",
                                                       "textTransform": "uppercase", "letterSpacing": "0.5px",
                                                       "fontFamily": "DM Sans, sans-serif"}),
                            html.Span("Exit %", style={"color": THEME["red"], "fontSize": "10px",
                                                        "textTransform": "uppercase", "letterSpacing": "0.5px",
                                                        "fontFamily": "DM Sans, sans-serif"}),
                        ]),
                    ]),
                    *[html.Div(style={
                        "display": "flex", "justifyContent": "space-between",
                        "alignItems": "center", "padding": "5px 0",
                        "borderBottom": f"1px solid {THEME['border']}22",
                    }, children=[
                        html.Span(row["team"], style={"color": THEME["text"], "fontSize": "11px",
                                                       "fontFamily": "DM Sans, sans-serif"}),
                        html.Div(style={"display": "flex", "gap": "12px"}, children=[
                            html.Span(f"{row['p_champion']:.1f}%", style={
                                "color": THEME["green"], "fontFamily": "JetBrains Mono, monospace",
                                "fontSize": "11px", "fontWeight": "600", "minWidth": "36px",
                                "textAlign": "right",
                            }),
                            html.Span(f"{row['p_group_exit']:.1f}%", style={
                                "color": THEME["red"], "fontFamily": "JetBrains Mono, monospace",
                                "fontSize": "11px", "fontWeight": "600", "minWidth": "36px",
                                "textAlign": "right",
                            }),
                        ]),
                    ]) for _, row in master[master["group"] == group].sort_values("p_champion", ascending=False).iterrows()],
                ])
                for group in sorted(master["group"].unique())
            ]),
            html.Div(style={"display": "flex", "gap": "24px", "marginTop": "14px",
                            "paddingTop": "12px", "borderTop": f"1px solid {THEME['border']}"}, children=[
                html.Span("Win % — Champion probability from 10,000 simulations",
                          style={"color": THEME["muted"], "fontSize": "11px", "fontFamily": "DM Sans, sans-serif"}),
                html.Span("Exit % — Probability of not advancing past the group stage",
                          style={"color": THEME["muted"], "fontSize": "11px", "fontFamily": "DM Sans, sans-serif"}),
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
                " This is our attempt to answer that. Using StatsBomb open data as a foundation, built by ",
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
                             "using 3 seasons of club data (2021/22, 2022/23, 2023/24) with time decay and Bayesian shrinkage.",
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
                "baseline. Only 7 countries exceed 0.8 confidence. Low-coverage nations receive a FIFA-fallback "
                "score blended by their confidence weight — the lower the confidence, the more the score relies "
                "on FIFA ranking rather than measured player data.",
                THEME["yellow"]),
            limit_card(5, "9 Teams Have No Tactical Data",
                "Bosnia and Herzegovina, Curaçao, Haiti, Iraq, Jordan, New Zealand, Norway, Uzbekistan, and Sweden "
                "do not appear in any of the 4 StatsBomb tournament datasets used for clustering (WC 2022, Euro 2024, "
                "Copa América 2024, AFCON 2023). They receive no tactical archetype assignment — their composite "
                "readiness scores rely entirely on player quality and external signals, with a confidence penalty applied.",
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
                     "GMM was run as a cross-validation check against KMeans. 23 of 69 teams received different assignments between the two models, confirming the continuum finding. Teams where GMM confidence = 0.0 are assigned directly to their second archetype; all others use their KMeans assignment."),
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


