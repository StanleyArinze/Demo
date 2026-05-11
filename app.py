import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import glob
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Beijing Air Quality Prediction",
    page_icon="🌫️",
    layout="wide",
)

st.title("🌫️ Beijing Air Quality Prediction System")
st.markdown("**CMP7005 Programming for Data Analysis** | PM2.5 Concentration Predictor")
st.divider()

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
FEATURES = ["PM10", "SO2", "NO2", "CO", "O3",
            "TEMP", "PRES", "DEWP", "RAIN", "WSPM"]

def get_season(month):
    if month in [12, 1, 2]:   return "Winter"
    elif month in [3, 4, 5]:  return "Spring"
    elif month in [6, 7, 8]:  return "Summer"
    else:                      return "Autumn"

def get_aqi_category(pm25):
    if pm25 <= 12:    return "Good", "#00e400"
    elif pm25 <= 35:  return "Moderate", "#ffff00"
    elif pm25 <= 55:  return "Unhealthy for Sensitive Groups", "#ff7e00"
    elif pm25 <= 150: return "Unhealthy", "#ff0000"
    elif pm25 <= 250: return "Very Unhealthy", "#8f3f97"
    else:             return "Hazardous", "#7e0023"

@st.cache_data(show_spinner="Loading and preprocessing data…")
def load_data(uploaded_files):
    dfs = []
    for f in uploaded_files:
        dfs.append(pd.read_csv(f))
    df = pd.concat(dfs, ignore_index=True)

    # Handle missing values
    for col in df.select_dtypes(include=np.number).columns:
        df[col] = df[col].fillna(df[col].median())
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].fillna(df[col].mode()[0])

    # Datetime & season
    df["datetime"] = pd.to_datetime(df[["year", "month", "day", "hour"]])
    df["season"] = df["month"].apply(get_season)

    # Station type
    urban = ["Dongsi", "Guanyuan"]
    df["station_type"] = df["station"].apply(
        lambda x: "Urban" if x in urban else "Suburban"
    )
    return df

@st.cache_resource(show_spinner="Training Random Forest model…")
def train_model(df):
    X = df[FEATURES]
    y = df["PM2.5"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_s, y_train)

    y_pred = model.predict(X_test_s)
    metrics = {
        "MSE":  mean_squared_error(y_test, y_pred),
        "RMSE": mean_squared_error(y_test, y_pred) ** 0.5,
        "R²":   r2_score(y_test, y_pred),
    }
    return model, scaler, y_test, y_pred, metrics

# ─────────────────────────────────────────────
# SIDEBAR — DATA UPLOAD
# ─────────────────────────────────────────────
st.sidebar.header("📂 Upload Data")
st.sidebar.markdown("Upload one or more Beijing air quality CSV files.")
uploaded = st.sidebar.file_uploader(
    "Choose CSV file(s)",
    type="csv",
    accept_multiple_files=True,
)

# ─────────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────────
if not uploaded:
    st.info(
        "👈 Upload at least one Beijing air quality CSV file from the sidebar to get started.\n\n"
        "**Expected columns:** `year`, `month`, `day`, `hour`, `PM2.5`, `PM10`, `SO2`, `NO2`, "
        "`CO`, `O3`, `TEMP`, `PRES`, `DEWP`, `RAIN`, `WSPM`, `station`"
    )
    st.stop()

# Load & train
df = load_data(uploaded)
model, scaler, y_test, y_pred, metrics = train_model(df)

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Data Overview",
    "📈 EDA & Visualisations",
    "🤖 Model Performance",
    "🔮 Predict PM2.5",
    "📋 Feature Importance",
])

# ── TAB 1: DATA OVERVIEW ──────────────────────
with tab1:
    st.subheader("Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows",    f"{df.shape[0]:,}")
    col2.metric("Columns", df.shape[1])
    col3.metric("Stations", df["station"].nunique() if "station" in df.columns else "N/A")
    col4.metric("Years",   df["year"].nunique() if "year" in df.columns else "N/A")

    st.markdown("**Sample Data**")
    st.dataframe(df.head(20), use_container_width=True)

    st.markdown("**Descriptive Statistics**")
    st.dataframe(df.describe(), use_container_width=True)

# ── TAB 2: EDA ───────────────────────────────
with tab2:
    st.subheader("Exploratory Data Analysis")

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("#### PM2.5 Distribution")
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.histplot(df["PM2.5"], bins=50, kde=True, ax=ax)
        ax.set_title("Distribution of PM2.5")
        st.pyplot(fig)
        plt.close()

        st.markdown("#### PM2.5 by Season")
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.boxplot(x="season", y="PM2.5", data=df, ax=ax,
                    order=["Spring","Summer","Autumn","Winter"])
        ax.set_title("PM2.5 Across Seasons")
        st.pyplot(fig)
        plt.close()

    with col_right:
        st.markdown("#### Temperature Distribution")
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.histplot(df["TEMP"], bins=50, kde=True, ax=ax, color="orange")
        ax.set_title("Distribution of Temperature")
        st.pyplot(fig)
        plt.close()

        st.markdown("#### PM2.5 by Station Type")
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.boxplot(x="station_type", y="PM2.5", data=df, ax=ax)
        ax.set_title("PM2.5: Urban vs Suburban")
        st.pyplot(fig)
        plt.close()

    st.markdown("#### PM2.5 Trend Over Time")
    fig, ax = plt.subplots(figsize=(14, 4))
    df.groupby("datetime")["PM2.5"].mean().plot(ax=ax)
    ax.set_title("PM2.5 Trend Over Time")
    ax.set_ylabel("Average PM2.5")
    st.pyplot(fig)
    plt.close()

    st.markdown("#### Correlation Heatmap")
    numeric_df = df.select_dtypes(include=np.number)
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Correlation Matrix")
    st.pyplot(fig)
    plt.close()

    if "station" in df.columns:
        st.markdown("#### PM2.5 by Monitoring Station")
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.boxplot(x="station", y="PM2.5", data=df, ax=ax)
        ax.set_title("PM2.5 by Station")
        plt.xticks(rotation=45)
        st.pyplot(fig)
        plt.close()

# ── TAB 3: MODEL PERFORMANCE ─────────────────
with tab3:
    st.subheader("Random Forest Model Performance")

    m1, m2, m3 = st.columns(3)
    m1.metric("MSE",  f"{metrics['MSE']:.2f}")
    m2.metric("RMSE", f"{metrics['RMSE']:.2f}")
    m3.metric("R² Score", f"{metrics['R²']:.4f}")

    st.markdown("#### Actual vs Predicted PM2.5")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_test, y_pred, alpha=0.3, s=10)
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", linewidth=1, label="Perfect prediction")
    ax.set_xlabel("Actual PM2.5")
    ax.set_ylabel("Predicted PM2.5")
    ax.set_title("Actual vs Predicted PM2.5")
    ax.legend()
    st.pyplot(fig)
    plt.close()

    st.markdown("#### Residual Distribution")
    residuals = np.array(y_test) - np.array(y_pred)
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(residuals, bins=50, kde=True, ax=ax)
    ax.axvline(0, color="red", linestyle="--")
    ax.set_title("Residuals (Actual − Predicted)")
    st.pyplot(fig)
    plt.close()

# ── TAB 4: PREDICT ───────────────────────────
with tab4:
    st.subheader("🔮 PM2.5 Prediction")
    st.markdown("Adjust the sliders below to input current pollutant and weather conditions.")

    col_a, col_b = st.columns(2)

    with col_a:
        pm10  = st.slider("PM10 (µg/m³)",   0.0, 999.0, 80.0,  step=1.0)
        so2   = st.slider("SO2 (µg/m³)",    0.0, 300.0, 20.0,  step=0.5)
        no2   = st.slider("NO2 (µg/m³)",    0.0, 300.0, 50.0,  step=0.5)
        co    = st.slider("CO (µg/m³)",     0.0, 10000.0, 800.0, step=10.0)
        o3    = st.slider("O3 (µg/m³)",     0.0, 500.0, 60.0,  step=1.0)

    with col_b:
        temp  = st.slider("Temperature (°C)",  -30.0, 45.0, 15.0, step=0.5)
        pres  = st.slider("Pressure (hPa)",    980.0, 1040.0, 1013.0, step=0.5)
        dewp  = st.slider("Dew Point (°C)",    -40.0, 30.0,  0.0,  step=0.5)
        rain  = st.slider("Rainfall (mm)",     0.0,  100.0,  0.0,  step=0.1)
        wspm  = st.slider("Wind Speed (m/s)",  0.0,   20.0,  2.0,  step=0.1)

    if st.button("Predict PM2.5", type="primary"):
        input_data = np.array([[pm10, so2, no2, co, o3,
                                 temp, pres, dewp, rain, wspm]])
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        category, colour = get_aqi_category(prediction)

        st.markdown("---")
        st.markdown(f"### Predicted PM2.5: **{prediction:.1f} µg/m³**")
        st.markdown(
            f"<div style='background-color:{colour};padding:12px;"
            f"border-radius:8px;color:#000;font-weight:bold;font-size:18px;'>"
            f"AQI Category: {category}</div>",
            unsafe_allow_html=True,
        )
        st.markdown("---")

        aqi_table = pd.DataFrame({
            "PM2.5 Range (µg/m³)": ["0–12", "12.1–35.4", "35.5–55.4",
                                     "55.5–150.4", "150.5–250.4", "250.5+"],
            "AQI Category": ["Good", "Moderate", "Unhealthy for Sensitive Groups",
                              "Unhealthy", "Very Unhealthy", "Hazardous"],
        })
        st.dataframe(aqi_table, use_container_width=True, hide_index=True)

# ── TAB 5: FEATURE IMPORTANCE ────────────────
with tab5:
    st.subheader("Feature Importance")
    importance = pd.Series(model.feature_importances_, index=FEATURES).sort_values()

    fig, ax = plt.subplots(figsize=(8, 5))
    importance.plot(kind="barh", ax=ax, color="steelblue")
    ax.set_title("Feature Importance — Random Forest")
    ax.set_xlabel("Importance Score")
    st.pyplot(fig)
    plt.close()

    st.dataframe(
        importance.reset_index()
              .rename(columns={"index": "Feature", 0: "Importance"})
              .sort_values("Importance", ascending=False),
        use_container_width=True,
        hide_index=True,
    )

# ─────────────────────────────────────────────
st.divider()
st.caption("CMP7005 Programming for Data Analysis · Beijing Air Quality Prediction System")
