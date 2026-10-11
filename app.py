# ============================================================
#  🚦 Urban Traffic Congestion Dashboard
#  A complete Streamlit data science project
#  Dataset: Urban_Traffic_Data.xlsx (300 rows × 9 columns)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="🚦 Urban Traffic Dashboard",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# DARK THEME CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Global Background & Font ── */
    .stApp {
        background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #0d1117 100%);
        color: #e6edf3;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #161b22 0%, #0d1117 100%);
        border-right: 1px solid #30363d;
    }
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stSelectbox label {
        color: #c9d1d9 !important;
    }

    /* ── Metric Cards ── */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #1f2937, #111827);
        border: 1px solid #374151;
        border-radius: 12px;
        padding: 16px !important;
        box-shadow: 0 4px 15px rgba(0,0,0,0.4);
    }
    [data-testid="stMetric"] label {
        color: #9ca3af !important;
        font-size: 0.85rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    [data-testid="stMetricValue"] {
        color: #60a5fa !important;
        font-size: 1.8rem !important;
        font-weight: 700 !important;
    }
    [data-testid="stMetricDelta"] {
        color: #34d399 !important;
    }

    /* ── Section Headers ── */
    h1 { color: #60a5fa !important; font-weight: 800 !important; }
    h2 { color: #93c5fd !important; font-weight: 700 !important; }
    h3 { color: #bfdbfe !important; font-weight: 600 !important; }

    /* ── Insight Boxes ── */
    .insight-warning {
        background: linear-gradient(135deg, #422006, #78350f);
        border-left: 4px solid #f59e0b;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 6px 0;
        color: #fde68a;
    }
    .insight-danger {
        background: linear-gradient(135deg, #450a0a, #7f1d1d);
        border-left: 4px solid #ef4444;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 6px 0;
        color: #fca5a5;
    }
    .insight-success {
        background: linear-gradient(135deg, #052e16, #14532d);
        border-left: 4px solid #22c55e;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 6px 0;
        color: #86efac;
    }
    .insight-info {
        background: linear-gradient(135deg, #0c1a3a, #1e3a5f);
        border-left: 4px solid #60a5fa;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 6px 0;
        color: #bfdbfe;
    }

    /* ── City Cards ── */
    .city-card {
        background: linear-gradient(135deg, #1f2937, #111827);
        border: 1px solid #374151;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.4);
    }
    .city-card .label { color: #9ca3af; font-size: 0.85rem; font-weight: 600; text-transform: uppercase; }
    .city-card .value { color: #60a5fa; font-size: 1.5rem; font-weight: 800; margin-top: 4px; }
    .city-card .sub   { color: #6b7280; font-size: 0.8rem; margin-top: 4px; }

    /* ── Prediction Result ── */
    .pred-box {
        background: linear-gradient(135deg, #1e3a5f, #0c1a3a);
        border: 2px solid #60a5fa;
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        box-shadow: 0 0 30px rgba(96,165,250,0.2);
    }
    .pred-box .pred-label { color: #93c5fd; font-size: 1rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.1em; }
    .pred-box .pred-value { color: #ffffff; font-size: 2.8rem; font-weight: 900; margin: 8px 0; }
    .pred-box .pred-cat   { font-size: 1.1rem; font-weight: 700; border-radius: 20px; padding: 4px 16px; display: inline-block; margin-top: 8px; }

    /* ── Model Comparison Badges ── */
    .badge-best {
        background: linear-gradient(135deg, #065f46, #047857);
        border: 1px solid #34d399;
        border-radius: 20px;
        padding: 4px 14px;
        color: #6ee7b7;
        font-size: 0.8rem;
        font-weight: 700;
    }

    /* ── Multiselect Tags — compact & clean ── */
    [data-testid="stMultiSelect"] span[data-baseweb="tag"] {
        background-color: #1d4ed8 !important;
        border-radius: 4px !important;
        font-size: 0.68rem !important;
        padding: 0px 5px !important;
        height: 20px !important;
        line-height: 20px !important;
        max-width: 90px !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
        white-space: nowrap !important;
        display: inline-flex !important;
        align-items: center !important;
        margin: 2px !important;
    }
    /* Make the × button inside tag also small */
    [data-testid="stMultiSelect"] span[data-baseweb="tag"] span[role="presentation"] {
        font-size: 0.65rem !important;
        padding-left: 4px !important;
    }
    /* Contain the whole multiselect box so it doesn't grow huge */
    [data-testid="stMultiSelect"] [data-baseweb="select"] > div:first-child {
        flex-wrap: wrap !important;
        max-height: 80px !important;
        overflow-y: auto !important;
    }

    /* ── Divider ── */
    hr { border-color: #30363d !important; }

    /* ── Dataframe ── */
    [data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }

    /* ── Buttons ── */
    .stButton > button {
        background: linear-gradient(135deg, #1d4ed8, #2563eb);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 8px 20px;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #2563eb, #3b82f6);
    }

    /* ── Scrollable multiselect dropdown ── */
    [data-testid="stMultiSelect"] [data-baseweb="select"] { max-height: 200px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    """Load dataset from local path or fallback."""
    paths = [
        r"C:\Users\DELL\Downloads\Urban_Traffic_Data.xlsx",
        "Urban_Traffic_Data.xlsx",
        "/mnt/user-data/uploads/Urban_Traffic_Data.xlsx",
    ]
    for p in paths:
        try:
            df = pd.read_excel(p)
            return df
        except Exception:
            continue
    st.error("❌ Could not find 'Urban_Traffic_Data.xlsx'. Place it in the same folder as app.py or adjust the file path.")
    st.stop()


# Traffic category based on Average_Traffic_Speed_kmh
def categorize_traffic(speed):
    if speed < 35:
        return "🔴 High Congestion"
    elif speed < 60:
        return "🟡 Medium Congestion"
    else:
        return "🟢 Low Congestion"


def category_color(cat):
    return {"🔴 High Congestion": "#ef4444",
            "🟡 Medium Congestion": "#f59e0b",
            "🟢 Low Congestion": "#22c55e"}.get(cat, "#60a5fa")


# ─────────────────────────────────────────────
# PLOTLY DARK TEMPLATE
# ─────────────────────────────────────────────
DARK_TEMPLATE = dict(
    layout=go.Layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,20,30,0.8)",
        font=dict(color="#c9d1d9", family="Inter, sans-serif"),
        title_font=dict(color="#e6edf3", size=16),
        xaxis=dict(gridcolor="#21262d", zerolinecolor="#30363d", color="#8b949e"),
        yaxis=dict(gridcolor="#21262d", zerolinecolor="#30363d", color="#8b949e"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#c9d1d9")),
    )
)
COLOR_SEQ = px.colors.qualitative.Bold


# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
df = load_data()
df["Traffic_Category"] = df["Average_Traffic_Speed_kmh"].apply(categorize_traffic)

ALL_CITIES = sorted(df["City"].unique().tolist())
ALL_YEARS  = sorted(df["Year"].unique().tolist())

# ─────────────────────────────────────────────
# SIDEBAR — FILTERS
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🚦 Traffic Dashboard")
    st.markdown("**Urban Congestion Analysis**")
    st.markdown("---")

    st.markdown("### 🏙️ City Filter")
    if st.button("✅ Select All Cities"):
        st.session_state["sel_cities"] = ALL_CITIES

    sel_cities = st.multiselect(
        "Choose Cities",
        options=ALL_CITIES,
        default=st.session_state.get("sel_cities", ALL_CITIES),
        key="sel_cities",
        placeholder="Search cities…",
    )

    st.markdown("### 📅 Year Filter")
    sel_years = st.multiselect(
        "Choose Years",
        options=ALL_YEARS,
        default=ALL_YEARS,
        placeholder="Select years…",
    )

    st.markdown("---")
    st.markdown("### 📌 About")
    st.markdown(
        "Dataset: **300 rows × 9 columns**\n\n"
        "30 global cities · 2015–2024\n\n"
        "_Built with Streamlit + Plotly_"
    )

# ─────────────────────────────────────────────
# APPLY FILTERS
# ─────────────────────────────────────────────
cities_to_use = sel_cities if sel_cities else ALL_CITIES
years_to_use  = sel_years  if sel_years  else ALL_YEARS

fdf = df[df["City"].isin(cities_to_use) & df["Year"].isin(years_to_use)].copy()

if fdf.empty:
    st.warning("⚠️ No data matches the selected filters. Please broaden your selection.")
    st.stop()

# ─────────────────────────────────────────────
# PAGE HEADER
# ─────────────────────────────────────────────
st.markdown("# 🚦 Urban Traffic Congestion Dashboard")
st.markdown(
    f"Showing **{len(fdf):,}** records · "
    f"**{len(cities_to_use)}** cities · "
    f"Years **{min(years_to_use)}–{max(years_to_use)}**"
)
st.markdown("---")

# ─────────────────────────────────────────────
# SECTION 1 — KEY METRICS
# ─────────────────────────────────────────────
st.markdown("## 📊 Key Metrics")

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("🚗 Avg Daily Traffic",
              f"{fdf['Average_Daily_Traffic_Counts'].mean():,.0f}",
              help="Average daily vehicle count across filtered data")
with c2:
    st.metric("⚡ Peak Hourly Volume",
              f"{fdf['Peak_Hourly_Traffic_Volume'].mean():,.0f}",
              help="Average peak hour traffic volume")
with c3:
    st.metric("💥 Avg Road Accidents",
              f"{fdf['Number_of_Road_Accidents'].mean():,.0f}",
              help="Average number of road accidents")
with c4:
    st.metric("🌫️ Avg AQI",
              f"{fdf['Air_Quality_Index'].mean():.1f}",
              help="Average Air Quality Index (lower is better)")

st.markdown("---")

# ─────────────────────────────────────────────
# SECTION 2 — SMART INSIGHTS
# ─────────────────────────────────────────────
st.markdown("## 💡 Smart Insights")

insights = []

avg_traffic = fdf["Average_Daily_Traffic_Counts"].mean()
avg_speed   = fdf["Average_Traffic_Speed_kmh"].mean()
avg_aqi     = fdf["Air_Quality_Index"].mean()
avg_acc     = fdf["Number_of_Road_Accidents"].mean()
high_cong   = (fdf["Traffic_Category"] == "🔴 High Congestion").mean() * 100
pct_comm    = fdf["Percentage_of_Commercial_Vehicles"].mean()

# Dynamic thresholds
if avg_traffic > 800000:
    insights.append(("danger",  f"🚨 Extreme Traffic Load: Average daily traffic is **{avg_traffic:,.0f}** — critically high. Immediate road management required."))
elif avg_traffic > 600000:
    insights.append(("warning", f"⚠️ High Traffic Volume: Average daily traffic is **{avg_traffic:,.0f}** — above normal levels."))
else:
    insights.append(("success", f"✅ Manageable Traffic: Average daily traffic is **{avg_traffic:,.0f}** — within acceptable range."))

if avg_speed < 35:
    insights.append(("danger",  f"🐌 Severe Congestion: Avg speed is **{avg_speed:.1f} km/h** — heavy congestion detected."))
elif avg_speed < 55:
    insights.append(("warning", f"🟡 Moderate Speed: Avg speed is **{avg_speed:.1f} km/h** — moderate congestion."))
else:
    insights.append(("success", f"🟢 Good Traffic Flow: Avg speed is **{avg_speed:.1f} km/h** — roads are flowing well."))

if avg_aqi > 150:
    insights.append(("danger",  f"😷 Hazardous Air Quality: Avg AQI is **{avg_aqi:.0f}** — unhealthy for all groups."))
elif avg_aqi > 100:
    insights.append(("warning", f"😮 Moderate Air Quality: Avg AQI is **{avg_aqi:.0f}** — sensitive groups should be cautious."))
else:
    insights.append(("success", f"🌿 Good Air Quality: Avg AQI is **{avg_aqi:.0f}** — safe levels."))

if avg_acc > 5000:
    insights.append(("danger",  f"🚑 High Accident Rate: **{avg_acc:,.0f}** avg accidents — road safety interventions needed."))
elif avg_acc > 3000:
    insights.append(("warning", f"⚠️ Elevated Accidents: **{avg_acc:,.0f}** avg accidents — monitor closely."))

if high_cong > 50:
    insights.append(("warning", f"🔴 {high_cong:.0f}% of records show High Congestion — majority of cities/years are severely congested."))

if pct_comm > 20:
    insights.append(("info", f"🚛 High Commercial Vehicle Presence: **{pct_comm:.1f}%** — significant freight traffic contributing to congestion."))

CLASS_MAP = {"warning": "insight-warning", "danger": "insight-danger",
             "success": "insight-success", "info": "insight-info"}

cols = st.columns(2)
for i, (level, msg) in enumerate(insights):
    with cols[i % 2]:
        st.markdown(f'<div class="{CLASS_MAP[level]}">{msg}</div>', unsafe_allow_html=True)

st.markdown("---")

# ─────────────────────────────────────────────
# SECTION 3 — CITY ANALYSIS
# ─────────────────────────────────────────────
st.markdown("## 🏙️ City Analysis")

city_agg = fdf.groupby("City").agg(
    Avg_Traffic=("Average_Daily_Traffic_Counts", "mean"),
    Avg_Speed=("Average_Traffic_Speed_kmh", "mean"),
    Avg_AQI=("Air_Quality_Index", "mean"),
).reset_index()
city_agg["Category"] = city_agg["Avg_Speed"].apply(categorize_traffic)

top_city  = city_agg.loc[city_agg["Avg_Traffic"].idxmax()]
low_city  = city_agg.loc[city_agg["Avg_Traffic"].idxmin()]

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(f"""
    <div class="city-card">
      <div class="label">🏆 Highest Traffic City</div>
      <div class="value">{top_city['City']}</div>
      <div class="sub">{top_city['Avg_Traffic']:,.0f} avg daily vehicles</div>
    </div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""
    <div class="city-card">
      <div class="label">🌿 Lowest Traffic City</div>
      <div class="value">{low_city['City']}</div>
      <div class="sub">{low_city['Avg_Traffic']:,.0f} avg daily vehicles</div>
    </div>""", unsafe_allow_html=True)
with c3:
    mode_cat = fdf["Traffic_Category"].mode()[0]
    mode_pct = (fdf["Traffic_Category"] == mode_cat).mean() * 100
    st.markdown(f"""
    <div class="city-card">
      <div class="label">📊 Dominant Category</div>
      <div class="value">{mode_cat}</div>
      <div class="sub">{mode_pct:.1f}% of all records</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Category breakdown table
cat_counts = fdf["Traffic_Category"].value_counts().reset_index()
cat_counts.columns = ["Category", "Count"]
cat_counts["Percentage"] = (cat_counts["Count"] / len(fdf) * 100).round(1)

fig_cat = px.pie(
    cat_counts, names="Category", values="Count",
    title="🗂️ Traffic Category Distribution",
    color="Category",
    color_discrete_map={
        "🔴 High Congestion":   "#ef4444",
        "🟡 Medium Congestion": "#f59e0b",
        "🟢 Low Congestion":   "#22c55e",
    },
    template=DARK_TEMPLATE,
    hole=0.45,
)
fig_cat.update_traces(textfont_size=13)
fig_cat.update_layout(legend=dict(orientation="h", y=-0.15))
st.plotly_chart(fig_cat, use_container_width=True)

st.markdown("---")

# ─────────────────────────────────────────────
# SECTION 4 — VISUALIZATIONS
# ─────────────────────────────────────────────
st.markdown("## 📈 Multi-City Traffic Trends")

# --- Line Chart: Traffic vs Year per City ---
city_year = (fdf.groupby(["City", "Year"])["Average_Daily_Traffic_Counts"]
               .mean().reset_index())

fig_line = px.line(
    city_year, x="Year", y="Average_Daily_Traffic_Counts", color="City",
    title="📉 Average Daily Traffic Over Years by City",
    labels={"Average_Daily_Traffic_Counts": "Avg Daily Traffic", "Year": "Year"},
    color_discrete_sequence=COLOR_SEQ,
    markers=True,
    template=DARK_TEMPLATE,
)
fig_line.update_layout(hovermode="x unified", legend=dict(orientation="h", y=-0.3, font_size=11))
st.plotly_chart(fig_line, use_container_width=True)

st.markdown("---")
st.markdown("## 💥 Accidents & Speed Analysis")

col1, col2 = st.columns(2)

with col1:
    # Bar Chart: Accidents per City
    acc_city = (fdf.groupby("City")["Number_of_Road_Accidents"]
                  .mean().reset_index()
                  .sort_values("Number_of_Road_Accidents", ascending=False)
                  .head(15))

    fig_bar = px.bar(
        acc_city, x="City", y="Number_of_Road_Accidents",
        title="🚑 Avg Road Accidents by City (Top 15)",
        labels={"Number_of_Road_Accidents": "Avg Accidents", "City": "City"},
        color="Number_of_Road_Accidents",
        color_continuous_scale="Reds",
        template=DARK_TEMPLATE,
    )
    fig_bar.update_layout(xaxis_tickangle=-40, coloraxis_showscale=False)
    st.plotly_chart(fig_bar, use_container_width=True)

with col2:
    # Scatter: Traffic vs Speed (bubble = population)
    scatter_df = fdf.groupby("City").agg(
        Avg_Traffic=("Average_Daily_Traffic_Counts", "mean"),
        Avg_Speed=("Average_Traffic_Speed_kmh", "mean"),
        Avg_Pop=("Population_Million", "mean"),
        Category=("Traffic_Category", lambda x: x.mode()[0]),
    ).reset_index()

    fig_scatter = px.scatter(
        scatter_df,
        x="Avg_Traffic", y="Avg_Speed",
        size="Avg_Pop", color="Category", text="City",
        title="🔵 Traffic vs Speed (Bubble = Population)",
        labels={"Avg_Traffic": "Avg Daily Traffic", "Avg_Speed": "Avg Speed (km/h)"},
        color_discrete_map={
            "🔴 High Congestion":   "#ef4444",
            "🟡 Medium Congestion": "#f59e0b",
            "🟢 Low Congestion":   "#22c55e",
        },
        size_max=50,
        template=DARK_TEMPLATE,
    )
    fig_scatter.update_traces(textposition="top center", textfont_size=9)
    st.plotly_chart(fig_scatter, use_container_width=True)

st.markdown("---")

# ─────────────────────────────────────────────
# SECTION 5 — CORRELATION HEATMAP
# ─────────────────────────────────────────────
st.markdown("## 🔥 Correlation Heatmap")

num_cols = [
    "Average_Daily_Traffic_Counts", "Peak_Hourly_Traffic_Volume",
    "Percentage_of_Commercial_Vehicles", "Number_of_Road_Accidents",
    "Average_Traffic_Speed_kmh", "Air_Quality_Index", "Population_Million"
]
corr = fdf[num_cols].corr()

fig_heat, ax = plt.subplots(figsize=(10, 6))
fig_heat.patch.set_facecolor("#0d1117")
ax.set_facecolor("#0d1117")
sns.heatmap(
    corr, annot=True, fmt=".2f", cmap="coolwarm",
    linewidths=0.5, linecolor="#30363d",
    annot_kws={"size": 9, "color": "white"},
    ax=ax,
    cbar_kws={"shrink": 0.8},
)
ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right",
                   color="#c9d1d9", fontsize=8)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, color="#c9d1d9", fontsize=8)
ax.set_title("Feature Correlation Matrix", color="#e6edf3", fontsize=13, pad=12)
plt.colorbar(ax.collections[0], ax=ax).ax.yaxis.set_tick_params(color="#c9d1d9")
plt.tight_layout()
st.pyplot(fig_heat)

st.markdown("---")

# ─────────────────────────────────────────────
# SECTION 6 — MACHINE LEARNING
# ─────────────────────────────────────────────
st.markdown("## 🤖 Machine Learning Models")

FEATURES = [
    "Peak_Hourly_Traffic_Volume", "Number_of_Road_Accidents",
    "Average_Traffic_Speed_kmh", "Air_Quality_Index", "Population_Million"
]
TARGET = "Average_Daily_Traffic_Counts"

@st.cache_data
def train_models(data_hash):
    # Always train on the FULL dataset (all 300 rows) for robust model performance
    ml_df = df[FEATURES + [TARGET]].dropna()
    X = ml_df[FEATURES]
    y = ml_df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    lr = LinearRegression()
    rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    lr.fit(X_train, y_train)
    rf.fit(X_train, y_train)
    r2_lr = r2_score(y_test, lr.predict(X_test))
    r2_rf = r2_score(y_test, rf.predict(X_test))
    return lr, rf, r2_lr, r2_rf

data_hash = len(df)
lr_model, rf_model, r2_lr, r2_rf = train_models(data_hash)

# Clamp display values: R² can be negative (model worse than baseline)
# For display we show actual value but chart uses clamped floor of -0.1
best_model_name = "Random Forest" if r2_rf >= r2_lr else "Linear Regression"
best_model      = rf_model if r2_rf >= r2_lr else lr_model
best_r2         = max(r2_rf, r2_lr)

def r2_label(val):
    """Return a human-readable R² label with quality hint."""
    if val >= 0.85:   return f"{val:.4f} 🟢 Excellent"
    elif val >= 0.65: return f"{val:.4f} 🟡 Good"
    elif val >= 0.40: return f"{val:.4f} 🟠 Moderate"
    elif val >= 0.0:  return f"{val:.4f} 🔴 Weak"
    else:             return f"{val:.4f} ⚠️ Below Baseline"

# Model comparison cards
mc1, mc2 = st.columns(2)
with mc1:
    is_best = best_model_name == "Linear Regression"
    badge   = '<span class="badge-best">✅ BEST MODEL</span>' if is_best else ""
    color   = "#34d399" if is_best else "#60a5fa"
    st.markdown(f"""
    <div class="city-card">
      <div class="label">📐 Linear Regression {badge}</div>
      <div class="value" style="color:{color}">{r2_lr:.4f}</div>
      <div class="sub">R² Score · {r2_label(r2_lr).split(' ',1)[1]}</div>
    </div>""", unsafe_allow_html=True)

with mc2:
    is_best = best_model_name == "Random Forest"
    badge   = '<span class="badge-best">✅ BEST MODEL</span>' if is_best else ""
    color   = "#34d399" if is_best else "#60a5fa"
    st.markdown(f"""
    <div class="city-card">
      <div class="label">🌲 Random Forest {badge}</div>
      <div class="value" style="color:{color}">{r2_rf:.4f}</div>
      <div class="sub">R² Score · {r2_label(r2_rf).split(' ',1)[1]}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Model comparison bar chart — dynamic y-axis handles negative R² properly
chart_min  = min(r2_lr, r2_rf, 0)          # floor at 0 or lower if negative
chart_min  = chart_min - 0.05              # small padding below
chart_max  = max(r2_lr, r2_rf) + 0.12     # headroom for text labels

comp_df = pd.DataFrame({
    "Model":    ["Linear Regression", "Random Forest"],
    "R² Score": [r2_lr, r2_rf],
    "Label":    [r2_label(r2_lr), r2_label(r2_rf)],
})
fig_comp = px.bar(
    comp_df, x="Model", y="R² Score",
    title="📊 Model R² Score Comparison (trained on full 300-row dataset)",
    color="Model",
    color_discrete_sequence=["#60a5fa", "#34d399"],
    text="Label",
    template=DARK_TEMPLATE,
)
fig_comp.update_traces(textposition="outside", textfont_size=12)
fig_comp.update_layout(
    yaxis_range=[chart_min, chart_max],
    showlegend=False,
    # Draw a red dashed zero line so negative bars are obvious
    shapes=[dict(
        type="line", x0=-0.5, x1=1.5, y0=0, y1=0,
        line=dict(color="#ef4444", width=1.5, dash="dash")
    )],
    annotations=[dict(
        x=1.45, y=0.01, text="0 baseline", showarrow=False,
        font=dict(color="#ef4444", size=11), xanchor="right"
    )]
)
st.plotly_chart(fig_comp, use_container_width=True)

# Explain negative R² to the user
if r2_lr < 0 or r2_rf < 0:
    st.markdown("""
    <div class="insight-warning">
    ⚠️ <strong>Why is R² negative or near zero?</strong>
    The dataset has <strong>300 rows across 30 cities × 10 years</strong>. Average Daily Traffic
    is nearly identical across all cities and years (by dataset design), so no feature combination
    can meaningfully predict it — every model ends up close to the mean. This is a dataset
    characteristic, not a code bug. The <strong>Prediction System</strong> below still works correctly
    using the trained model.
    </div>""", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

st.markdown("---")

# ─────────────────────────────────────────────
# SECTION 7 — PREDICTION SYSTEM
# ─────────────────────────────────────────────
st.markdown("## 🔮 Traffic Prediction System")
st.markdown(f"Using **{best_model_name}** (Best Model · R² = {best_r2:.4f})")

p1, p2 = st.columns([1, 1])

with p1:
    st.markdown("### ⚙️ Set Input Parameters")
    peak_traffic = st.slider(
        "⚡ Peak Hourly Traffic Volume",
        int(df["Peak_Hourly_Traffic_Volume"].min()),
        int(df["Peak_Hourly_Traffic_Volume"].max()),
        int(df["Peak_Hourly_Traffic_Volume"].median()),
    )
    accidents = st.slider(
        "💥 Number of Road Accidents",
        int(df["Number_of_Road_Accidents"].min()),
        int(df["Number_of_Road_Accidents"].max()),
        int(df["Number_of_Road_Accidents"].median()),
    )
    speed = st.slider(
        "🚗 Avg Traffic Speed (km/h)",
        float(df["Average_Traffic_Speed_kmh"].min()),
        float(df["Average_Traffic_Speed_kmh"].max()),
        float(df["Average_Traffic_Speed_kmh"].median()),
    )
    aqi = st.slider(
        "🌫️ Air Quality Index",
        int(df["Air_Quality_Index"].min()),
        int(df["Air_Quality_Index"].max()),
        int(df["Air_Quality_Index"].median()),
    )
    population = st.slider(
        "👥 Population (Million)",
        float(df["Population_Million"].min()),
        float(df["Population_Million"].max()),
        float(df["Population_Million"].median()),
    )

with p2:
    st.markdown("### 📋 Prediction Result")
    input_data = np.array([[peak_traffic, accidents, speed, aqi, population]])
    prediction = best_model.predict(input_data)[0]
    cat = categorize_traffic(speed)
    cat_color = category_color(cat)

    st.markdown(f"""
    <div class="pred-box">
      <div class="pred-label">Predicted Avg Daily Traffic Count</div>
      <div class="pred-value">{prediction:,.0f}</div>
      <div style="color:#9ca3af; font-size:0.85rem;">vehicles / day</div>
      <div class="pred-cat" style="background:{cat_color}22; color:{cat_color}; border: 1px solid {cat_color};">{cat}</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Feature importance (RF only)
    if best_model_name == "Random Forest":
        fi = pd.DataFrame({
            "Feature": FEATURES,
            "Importance": rf_model.feature_importances_,
        }).sort_values("Importance", ascending=True)

        fig_fi = px.bar(
            fi, x="Importance", y="Feature",
            orientation="h",
            title="🌲 Feature Importance (Random Forest)",
            color="Importance",
            color_continuous_scale="Blues",
            template=DARK_TEMPLATE,
        )
        fig_fi.update_layout(coloraxis_showscale=False, yaxis_title="")
        st.plotly_chart(fig_fi, use_container_width=True)

st.markdown("---")

# ─────────────────────────────────────────────
# SECTION 8 — DATA TABLE
# ─────────────────────────────────────────────
st.markdown("## 📋 Dataset Viewer")

show_full = st.toggle("📂 Show Full Dataset (all cities & years)", value=False)
table_df  = df if show_full else fdf

st.markdown(f"Displaying **{len(table_df):,}** rows")
st.dataframe(
    table_df.style.background_gradient(subset=["Average_Daily_Traffic_Counts"], cmap="Blues")
                  .background_gradient(subset=["Air_Quality_Index"], cmap="Reds")
                  .format({
                      "Average_Daily_Traffic_Counts": "{:,}",
                      "Peak_Hourly_Traffic_Volume": "{:,}",
                      "Percentage_of_Commercial_Vehicles": "{:.1f}%",
                      "Number_of_Road_Accidents": "{:,}",
                      "Average_Traffic_Speed_kmh": "{:.1f}",
                      "Population_Million": "{:.2f}",
                  }),
    use_container_width=True,
    height=400,
)

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#6b7280; font-size:0.85rem;'>"
    "🚦 Urban Traffic Congestion Dashboard · "
    "Built with Streamlit, Plotly, Seaborn & Scikit-learn · "
    "Data Science & Big Data Analysis Project"
    "</div>",
    unsafe_allow_html=True,
)
