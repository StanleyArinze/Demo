"""
Beijing Air Quality Prediction System
CMP7005 - Streamlit Web Application
Student: Stanley Arinze Ugwuoke | ID: st20338478
"""
 
import streamlit as st
import pandas as pd
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
 
# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Beijing Air Quality",
    page_icon="🌫️",
    layout="wide",
)
 
# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2rem;
        font-weight: 700;
        color: #1a73e8;
    }
    .metric-card {
        background: #f0f4ff;
        border-radius: 10px;
        padding: 1rem 1.5rem;
        text-align: center;
    }
    .section-header {
        font-size: 1.2rem;
        font-weight: 600;
        color: #333;
        margin-top: 1rem;
    }
    [data-testid="stSidebar"] {
        background-color: #0d1b2a;
    }
    [data-testid="stSidebar"] * {
        color: #e0e8f5 !important;
    }
    [data-testid="stSidebar"] .stRadio label {
        font-size: 0.95rem;
    }
    .stAlert {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)
 
 
# ─── Data Loading ────────────────────────────────────────────────────────────
@st.cache_data
def load_data(uploaded_files):
    dfs = []
    for f in uploaded_files:
        df = pd.read_csv(f)
        dfs.append(df)
    air_df = pd.concat(dfs, ignore_index=True)
    return air_df
 
 
@st.cache_data
def preprocess(air_df: pd.DataFrame) -> pd.DataFrame:
    df = air_df.copy()
 
    # Handle missing values
    for col in df.select_dtypes(include=np.number).columns:
        df[col] = df[col].fillna(df[col].median())
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].fillna(df[col].mode()[0])
 
    # Datetime
    if all(c in df.columns for c in ["year", "month", "day", "hour"]):
        df["datetime"] = pd.to_datetime(df[["year", "month", "day", "hour"]])
 
    # Season
    def get_season(m):
        if m in [12, 1, 2]:   return "Winter"
        elif m in [3, 4, 5]:  return "Spring"
        elif m in [6, 7, 8]:  return "Summer"
        else:                  return "Autumn"
 
    if "month" in df.columns:
        df["season"] = df["month"].apply(get_season)
 
    # Station type
    if "station" in df.columns:
        urban = ["Dongsi", "Guanyuan"]
        df["station_type"] = df["station"].apply(
            lambda x: "Urban" if x in urban else "Suburban"
        )
 
    return df
 
 
@st.cache_resource
def train_model(df: pd.DataFrame):
    features = [c for c in ["PM10","SO2","NO2","CO","O3","TEMP","PRES","DEWP","RAIN","WSPM"]
                if c in df.columns]
    if "PM2.5" not in df.columns or len(features) < 3:
        return None, None, None, None, None
 
    X = df[features].dropna()
    y = df.loc[X.index, "PM2.5"]
 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_train)
    X_te_sc = scaler.transform(X_test)
 
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_tr_sc, y_train)
 
    y_pred = model.predict(X_te_sc)
    metrics = {
        "MSE":  round(mean_squared_error(y_test, y_pred), 3),
        "RMSE": round(mean_squared_error(y_test, y_pred) ** 0.5, 3),
        "R²":   round(r2_score(y_test, y_pred), 4),
    }
    return model, scaler, features, metrics, (y_test, y_pred)
 
 
# ─── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌫️ Beijing Air Quality")
    st.markdown("*CMP7005 · Stanley Ugwuoke*")
    st.markdown("---")
 
    st.markdown("### 📂 Upload Dataset(s)")
    uploaded = st.file_uploader(
        "Upload one or more CSV files",
        type=["csv"],
        accept_multiple_files=True,
        help="Upload the Beijing air quality CSV files (e.g. Dongsi, Guanyuan, Shunyi, Huairou)"
    )
 
    st.markdown("---")
    st.markdown("### 🗂️ Navigation")
    page = st.radio(
        "",
        ["🏠 Home",
         "📋 Dataset View",
         "📊 Visualisations",
         "🤖 Model Outputs",
         "🔮 Predict PM2.5"],
        label_visibility="collapsed"
    )
 
    st.markdown("---")
    st.caption("Built with Streamlit · RandomForest Regressor")
 
 
# ─── Guards ──────────────────────────────────────────────────────────────────
if not uploaded:
    st.markdown('<p class="main-title">🌫️ Beijing Air Quality Prediction System</p>', unsafe_allow_html=True)
    st.info("👈  Upload your Beijing air quality CSV files in the sidebar to get started.")
    st.markdown("""
    **Expected columns include:**
    `year`, `month`, `day`, `hour`, `PM2.5`, `PM10`, `SO2`, `NO2`, `CO`, `O3`,
    `TEMP`, `PRES`, `DEWP`, `RAIN`, `WSPM`, `station`
    """)
    st.stop()
 
# Load & preprocess
raw_df = load_data(uploaded)
df = preprocess(raw_df)
 
# ─── HOME ────────────────────────────────────────────────────────────────────
if page == "🏠 Home":
    st.markdown('<p class="main-title">🌫️ Beijing Air Quality Prediction System</p>', unsafe_allow_html=True)
    st.markdown("Analysing pollution across monitoring stations — **Guanyuan · Dongsi · Shunyi · Huairou**")
    st.markdown("---")
 
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records",   f"{len(df):,}")
    c2.metric("Stations",        df["station"].nunique() if "station" in df.columns else "—")
    c3.metric("Avg PM2.5",       f"{df['PM2.5'].mean():.1f} µg/m³" if "PM2.5" in df.columns else "—")
    c4.metric("Date Range",
              f"{df['datetime'].dt.year.min()}–{df['datetime'].dt.year.max()}"
              if "datetime" in df.columns else "—")
 
    st.markdown("#### Quick PM2.5 Snapshot")
    if "datetime" in df.columns and "PM2.5" in df.columns:
        fig, ax = plt.subplots(figsize=(12, 3))
        df.set_index("datetime")["PM2.5"].resample("M").mean().plot(ax=ax, color="#1a73e8")
        ax.set_xlabel(""); ax.set_ylabel("µg/m³"); ax.set_title("")
        ax.spines[["top","right"]].set_visible(False)
        st.pyplot(fig); plt.close()
 
 
# ─── DATASET VIEW ────────────────────────────────────────────────────────────
elif page == "📋 Dataset View":
    st.header("📋 Dataset View")
 
    tab1, tab2, tab3 = st.tabs(["Preview", "Statistics", "Missing Values"])
 
    with tab1:
        st.subheader("Raw Data Preview")
        n = st.slider("Rows to display", 5, 100, 20)
        if "station" in df.columns:
            stations = ["All"] + sorted(df["station"].unique().tolist())
            sel = st.selectbox("Filter by station", stations)
            view = df if sel == "All" else df[df["station"] == sel]
        else:
            view = df
        st.dataframe(view.head(n), use_container_width=True)
        st.caption(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
 
    with tab2:
        st.subheader("Descriptive Statistics")
        st.dataframe(df.describe().T.style.format("{:.2f}"), use_container_width=True)
 
    with tab3:
        st.subheader("Missing Value Summary")
        mv = df.isnull().sum().reset_index()
        mv.columns = ["Column", "Missing"]
        mv["% Missing"] = (mv["Missing"] / len(df) * 100).round(2)
        mv = mv[mv["Missing"] > 0]
        if mv.empty:
            st.success("✅ No missing values found after preprocessing.")
        else:
            st.dataframe(mv, use_container_width=True)
 
 
# ─── VISUALISATIONS ──────────────────────────────────────────────────────────
elif page == "📊 Visualisations":
    st.header("📊 Visualisations")
 
    viz = st.selectbox("Choose a chart", [
        "PM2.5 Distribution",
        "Temperature Distribution",
        "PM2.5 Trend Over Time",
        "PM2.5 by Season",
        "PM2.5 by Station",
        "PM2.5 by Station Type",
        "PM2.5 vs Temperature",
        "Correlation Heatmap",
    ])
 
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.set_style("whitegrid")
 
    if viz == "PM2.5 Distribution" and "PM2.5" in df.columns:
        sns.histplot(df["PM2.5"], bins=50, kde=True, ax=ax, color="#1a73e8")
        ax.set_title("Distribution of PM2.5"); ax.set_xlabel("PM2.5 (µg/m³)")
 
    elif viz == "Temperature Distribution" and "TEMP" in df.columns:
        sns.histplot(df["TEMP"], bins=50, kde=True, ax=ax, color="#e84040")
        ax.set_title("Distribution of Temperature"); ax.set_xlabel("Temperature (°C)")
 
    elif viz == "PM2.5 Trend Over Time" and "datetime" in df.columns:
        df.set_index("datetime")["PM2.5"].resample("M").mean().plot(ax=ax, color="#1a73e8")
        ax.set_title("Monthly Average PM2.5 Over Time"); ax.set_ylabel("PM2.5 (µg/m³)")
 
    elif viz == "PM2.5 by Season" and "season" in df.columns:
        order = ["Spring","Summer","Autumn","Winter"]
        sns.boxplot(x="season", y="PM2.5", data=df, order=order,
                    palette="Set2", ax=ax)
        ax.set_title("PM2.5 Concentration by Season")
 
    elif viz == "PM2.5 by Station" and "station" in df.columns:
        sns.boxplot(x="station", y="PM2.5", data=df, palette="Set3", ax=ax)
        ax.set_title("PM2.5 by Monitoring Station")
        plt.xticks(rotation=30)
 
    elif viz == "PM2.5 by Station Type" and "station_type" in df.columns:
        sns.boxplot(x="station_type", y="PM2.5", data=df, palette=["#1a73e8","#e84040"], ax=ax)
        ax.set_title("PM2.5: Urban vs Suburban Stations")
 
    elif viz == "PM2.5 vs Temperature" and "TEMP" in df.columns:
        sample = df.sample(min(5000, len(df)), random_state=1)
        sns.scatterplot(x="TEMP", y="PM2.5", data=sample, alpha=0.3, ax=ax, color="#1a73e8")
        ax.set_title("PM2.5 vs Temperature")
 
    elif viz == "Correlation Heatmap":
        plt.close()
        fig, ax = plt.subplots(figsize=(12, 8))
        num = df.select_dtypes(include=np.number)
        sns.heatmap(num.corr(), annot=True, fmt=".2f", cmap="coolwarm",
                    linewidths=0.5, ax=ax)
        ax.set_title("Correlation Matrix")
 
    else:
        st.warning("Required column not found in the dataset.")
        plt.close()
        st.stop()
 
    ax.spines[["top","right"]].set_visible(False) if viz != "Correlation Heatmap" else None
    st.pyplot(fig); plt.close()
 
 
# ─── MODEL OUTPUTS ───────────────────────────────────────────────────────────
elif page == "🤖 Model Outputs":
    st.header("🤖 Random Forest — Model Outputs")
 
    with st.spinner("Training model… this may take a moment."):
        model, scaler, features, metrics, test_data = train_model(df)
 
    if model is None:
        st.error("Could not train model. Please check that the dataset contains PM2.5 and pollutant columns.")
        st.stop()
 
    # Metrics
    st.subheader("Performance Metrics")
    c1, c2, c3 = st.columns(3)
    c1.metric("MSE",  metrics["MSE"])
    c2.metric("RMSE", metrics["RMSE"])
    c3.metric("R² Score", metrics["R²"])
 
    y_test, y_pred = test_data
 
    tab1, tab2 = st.tabs(["Actual vs Predicted", "Feature Importance"])
 
    with tab1:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(y_test, y_pred, alpha=0.3, color="#1a73e8", s=10)
        mn, mx = float(y_test.min()), float(y_test.max())
        ax.plot([mn, mx], [mn, mx], "r--", lw=1.5, label="Perfect fit")
        ax.set_xlabel("Actual PM2.5"); ax.set_ylabel("Predicted PM2.5")
        ax.set_title("Actual vs Predicted PM2.5"); ax.legend()
        ax.spines[["top","right"]].set_visible(False)
        st.pyplot(fig); plt.close()
 
    with tab2:
        importance = pd.Series(model.feature_importances_, index=features).sort_values()
        fig, ax = plt.subplots(figsize=(8, 5))
        importance.plot(kind="barh", ax=ax, color="#1a73e8")
        ax.set_title("Feature Importance"); ax.set_xlabel("Importance Score")
        ax.spines[["top","right"]].set_visible(False)
        st.pyplot(fig); plt.close()
 
    st.markdown("#### Interpretation")
    top3 = pd.Series(model.feature_importances_, index=features).nlargest(3).index.tolist()
    st.info(f"**Top predictors:** {', '.join(top3)} — confirming strong relationships between particulate and gaseous pollutants.")
 
 
# ─── PREDICT PM2.5 ───────────────────────────────────────────────────────────
elif page == "🔮 Predict PM2.5":
    st.header("🔮 Predict PM2.5 Concentration")
 
    with st.spinner("Loading model…"):
        model, scaler, features, metrics, _ = train_model(df)
 
    if model is None:
        st.error("Model could not be trained. Check dataset columns.")
        st.stop()
 
    st.markdown("Enter pollutant and meteorological values to predict PM2.5 level:")
 
    defaults = dict(PM10=80, SO2=15, NO2=40, CO=1000, O3=60,
                    TEMP=12, PRES=1013, DEWP=2, RAIN=0, WSPM=2)
 
    cols = st.columns(2)
    user_input = {}
    for i, feat in enumerate(features):
        with cols[i % 2]:
            user_input[feat] = st.number_input(
                feat, value=float(defaults.get(feat, 0.0)), step=0.1
            )
 
    if st.button("🔍 Predict", type="primary"):
        X_in = np.array([[user_input[f] for f in features]])
        X_sc = scaler.transform(X_in)
        pred = model.predict(X_sc)[0]
 
        # AQI category
        if pred <= 12:      aqi, col = "Good 🟢", "green"
        elif pred <= 35.4:  aqi, col = "Moderate 🟡", "orange"
        elif pred <= 55.4:  aqi, col = "Unhealthy for Sensitive Groups 🟠", "orange"
        elif pred <= 150.4: aqi, col = "Unhealthy 🔴", "red"
        elif pred <= 250.4: aqi, col = "Very Unhealthy 🟣", "purple"
        else:               aqi, col = "Hazardous ⚫", "maroon"
 
        st.markdown(f"""
        <div style='background:#f0f4ff;border-radius:12px;padding:1.5rem;text-align:center;margin-top:1rem'>
            <h2 style='color:#1a73e8;margin:0'>Predicted PM2.5</h2>
            <h1 style='font-size:3rem;margin:0.5rem 0'>{pred:.1f} µg/m³</h1>
            <h3 style='color:{col};margin:0'>AQI Category: {aqi}</h3>
        </div>
        """, unsafe_allow_html=True)
 
