# ============================================================
# CMP7005 PRAC1: Beijing Air Quality Prediction Dashboard
# Streamlit Application
# Author: Ugwuoke Stanley Arinze
# Purpose: Distinction-focused GUI aligned with the CMP7005 rubric:
#          Data Handling, EDA, Model Building, App Development,
#          and Version Control.
# ============================================================

import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import mean_squared_error, r2_score

# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Beijing Air Quality Prediction System",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# 2. GLOBAL STYLING
#    This CSS gives the app a clean dashboard appearance.
# ============================================================

st.markdown(
    """
    <style>
    .main {background-color: #f6f8fb;}
    .block-container {padding-top: 2rem; padding-bottom: 2rem;}
    .hero-card {
        background: linear-gradient(135deg, #0f766e 0%, #2563eb 100%);
        padding: 2rem;
        border-radius: 22px;
        color: white;
        margin-bottom: 1.2rem;
        box-shadow: 0 10px 30px rgba(15, 118, 110, 0.25);
    }
    .section-card {
        background-color: #ffffff;
        padding: 1.4rem;
        border-radius: 18px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 6px 18px rgba(0,0,0,0.04);
        margin-bottom: 1rem;
    }
    .small-muted {color: #64748b; font-size: 0.95rem;}
    .insight-box {
        background-color: #ecfeff;
        border-left: 6px solid #0891b2;
        padding: 1rem;
        border-radius: 12px;
        margin-top: 0.8rem;
    }
    .warning-box {
        background-color: #fff7ed;
        border-left: 6px solid #f97316;
        padding: 1rem;
        border-radius: 12px;
        margin-top: 0.8rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================

FEATURES = ["PM10", "SO2", "NO2", "CO", "O3", "TEMP", "PRES", "DEWP", "RAIN", "WSPM"]


def find_file(possible_paths):
    """Return the first existing file path from a list of possible paths."""
    for path in possible_paths:
        if Path(path).exists():
            return path
    return None


@st.cache_data(show_spinner=False)
def load_dataset():
    """Load the cleaned air quality dataset from common Colab/local paths."""
    data_path = find_file([
        "beijing_air_quality_cleaned.csv",
        "data/processed/beijing_air_quality_cleaned.csv",
        "/content/beijing_air_quality_cleaned.csv",
        "/content/data/processed/beijing_air_quality_cleaned.csv"
    ])

    if data_path is None:
        return None, None

    df = pd.read_csv(data_path)

    # Convert datetime column where available for time-based visualisations.
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")

    return df, data_path


@st.cache_resource(show_spinner=False)
def load_saved_model():
    """Load the trained Random Forest model and scaler."""
    model_path = find_file(["model.pkl", "models/model.pkl", "/content/model.pkl", "/content/models/model.pkl"])
    scaler_path = find_file(["scaler.pkl", "models/scaler.pkl", "/content/scaler.pkl", "/content/models/scaler.pkl"])

    if model_path is None or scaler_path is None:
        return None, None, model_path, scaler_path

    trained_model = joblib.load(model_path)
    fitted_scaler = joblib.load(scaler_path)
    return trained_model, fitted_scaler, model_path, scaler_path


def pm25_category(pm25_value):
    """Classify predicted PM2.5 into a simple health-risk category."""
    if pm25_value <= 50:
        return "Good", "🟢", "Air quality is generally acceptable."
    if pm25_value <= 100:
        return "Moderate", "🟡", "Sensitive individuals should reduce long outdoor exposure."
    if pm25_value <= 150:
        return "Unhealthy for Sensitive Groups", "🟠", "People with respiratory issues should take extra care."
    if pm25_value <= 200:
        return "Unhealthy", "🔴", "Everyone may begin to experience health effects."
    return "Very Unhealthy / Hazardous", "🟣", "Outdoor activities should be limited where possible."


def render_insight(text):
    """Display an interpretation box after charts, matching lecturer-style EDA."""
    st.markdown(f"<div class='insight-box'><b>Interpretation:</b><br>{text}</div>", unsafe_allow_html=True)


def numeric_columns(df):
    """Return numeric columns for correlation and summaries."""
    return df.select_dtypes(include=[np.number]).columns.tolist()


# ============================================================
# 4. LOAD ASSETS
# ============================================================

data, loaded_data_path = load_dataset()
model, scaler, loaded_model_path, loaded_scaler_path = load_saved_model()

# ============================================================
# 5. SIDEBAR NAVIGATION
# ============================================================

st.sidebar.title("📌 CMP7005 Navigation")
st.sidebar.caption("Rubric-aligned project dashboard")

page = st.sidebar.radio(
    "Select a section",
    [
        "🏠 Project Overview",
        "📂 Task 1: Data Handling",
        "📊 Task 2: EDA",
        "🤖 Task 3: Model Building",
        "💻 Task 4: Application Development",
        "🗂️ Task 5: Version Control",
        "🎤 Presentation Guide"
    ]
)

st.sidebar.divider()
st.sidebar.subheader("Loaded files")
st.sidebar.write("Dataset:", loaded_data_path if loaded_data_path else "Not found")
st.sidebar.write("Model:", loaded_model_path if loaded_model_path else "Not found")
st.sidebar.write("Scaler:", loaded_scaler_path if loaded_scaler_path else "Not found")

# ============================================================
# 6. PROJECT OVERVIEW PAGE
# ============================================================

if page == "🏠 Project Overview":
    st.markdown(
        """
        <div class='hero-card'>
            <h1>🌍 Beijing Air Quality Prediction System</h1>
            <p>Interactive data analysis and PM2.5 prediction dashboard developed for CMP7005 Programming for Data Analysis.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Selected Stations", "4")
    col2.metric("Target Variable", "PM2.5")
    col3.metric("Model", "Random Forest")

    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader("Project Aim")
    st.write(
        """
        The aim of this project is to transform Beijing hourly air-quality data into a useful analytical and predictive tool.
        The work includes data selection, data merging, preprocessing, EDA, machine-learning model development,
        Streamlit application development, and version control documentation.
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader("Why PM2.5 Matters")
    st.write(
        """
        PM2.5 refers to fine particulate matter that can enter the lungs and bloodstream. Monitoring and predicting
        PM2.5 is useful because it supports public health awareness, environmental planning, and pollution-control decisions.
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# 7. TASK 1: DATA HANDLING PAGE
# ============================================================

elif page == "📂 Task 1: Data Handling":
    st.header("📂 Task 1: Data Handling")

    st.write(
        """
        This section demonstrates how the selected station datasets were imported and merged into a single unified dataset.
        The selected stations include two inner urban stations and two outer suburban stations.
        """
    )

    station_info = pd.DataFrame({
        "Station": ["Dongsi", "Guanyuan", "Shunyi", "Huairou"],
        "Station Type": ["Urban", "Urban", "Suburban", "Suburban"],
        "Justification": [
            "Represents inner-city pollution patterns.",
            "Represents another central/urban monitoring point.",
            "Represents outer/suburban air quality conditions.",
            "Represents a less central monitoring area."
        ]
    })
    st.dataframe(station_info, use_container_width=True)

    if data is not None:
        st.subheader("Combined Dataset Preview")
        st.dataframe(data.head(10), use_container_width=True)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", f"{data.shape[0]:,}")
        c2.metric("Columns", data.shape[1])
        c3.metric("Stations", data["station"].nunique() if "station" in data.columns else "N/A")
        c4.metric("Date Range", "2013–2017")

        render_insight(
            "The four station datasets were successfully merged into one dataset, making it possible to compare urban and suburban air quality patterns."
        )
    else:
        st.error("Dataset not found. Place beijing_air_quality_cleaned.csv in the same folder as app.py or in data/processed/.")

# ============================================================
# 8. TASK 2: EDA PAGE
# ============================================================

elif page == "📊 Task 2: EDA":
    st.header("📊 Task 2: Exploratory Data Analysis")

    if data is None:
        st.error("Dataset not found. Upload or copy beijing_air_quality_cleaned.csv before running this section.")
        st.stop()

    eda_tab1, eda_tab2, eda_tab3, eda_tab4 = st.tabs([
        "2.1 Data Understanding",
        "2.2 Preprocessing Summary",
        "2.3 Visual Analysis",
        "Correlation Analysis"
    ])

    with eda_tab1:
        st.subheader("Dataset Overview")
        col1, col2, col3 = st.columns(3)
        col1.metric("Rows", f"{data.shape[0]:,}")
        col2.metric("Columns", data.shape[1])
        col3.metric("Missing Values", int(data.isnull().sum().sum()))

        st.write("Column Data Types")
        dtype_df = pd.DataFrame({"Column": data.columns, "Data Type": [str(data[col].dtype) for col in data.columns]})
        st.dataframe(dtype_df, use_container_width=True)

        st.write("Statistical Summary")
        st.dataframe(data.describe().T, use_container_width=True)

        render_insight(
            "This overview confirms the size, structure, numerical variables, and readiness of the dataset for preprocessing and visual analysis."
        )

    with eda_tab2:
        st.subheader("Preprocessing Work Completed")
        st.markdown(
            """
            - Missing numerical values were handled using median values.
            - Missing categorical wind-direction values were handled using mode.
            - Duplicate records were checked and removed.
            - A datetime column was created from year, month, day, and hour.
            - Station type was added to distinguish urban and suburban stations.
            - Season was engineered from the month column.
            """
        )

        missing_table = pd.DataFrame({"Column": data.columns, "Missing Values": data.isnull().sum().values})
        st.dataframe(missing_table, use_container_width=True)

        render_insight(
            "After preprocessing, the dataset is cleaner and more suitable for EDA and machine-learning modelling."
        )

    with eda_tab3:
        st.subheader("Interactive Visualisations")

        chart_choice = st.selectbox(
            "Choose analysis chart",
            [
                "PM2.5 Distribution",
                "Average PM2.5 by Station",
                "Average PM2.5 by Station Type",
                "Average PM2.5 by Season",
                "PM2.5 Trend Over Time",
                "PM2.5 vs Temperature",
                "NO2 vs O3"
            ]
        )

        if chart_choice == "PM2.5 Distribution":
            fig = px.histogram(data, x="PM2.5", nbins=60, marginal="box", title="Distribution of PM2.5")
            st.plotly_chart(fig, use_container_width=True)
            render_insight("PM2.5 is usually right-skewed, meaning most readings are lower but some pollution episodes are very high.")

        elif chart_choice == "Average PM2.5 by Station":
            station_avg = data.groupby("station", as_index=False)["PM2.5"].mean().sort_values("PM2.5", ascending=False)
            fig = px.bar(station_avg, x="station", y="PM2.5", text_auto=".2f", title="Average PM2.5 by Station")
            st.plotly_chart(fig, use_container_width=True)
            render_insight("This comparison helps identify which monitoring station has the highest average particulate pollution.")

        elif chart_choice == "Average PM2.5 by Station Type":
            if "station_type" in data.columns:
                type_avg = data.groupby("station_type", as_index=False)["PM2.5"].mean()
                fig = px.bar(type_avg, x="station_type", y="PM2.5", text_auto=".2f", title="Average PM2.5 by Station Type")
                st.plotly_chart(fig, use_container_width=True)
                render_insight("Urban and suburban comparison supports discussion of spatial pollution variation.")
            else:
                st.warning("station_type column not found.")

        elif chart_choice == "Average PM2.5 by Season":
            if "season" in data.columns:
                season_avg = data.groupby("season", as_index=False)["PM2.5"].mean()
                fig = px.bar(season_avg, x="season", y="PM2.5", text_auto=".2f", title="Average PM2.5 by Season")
                st.plotly_chart(fig, use_container_width=True)
                render_insight("Seasonal analysis helps reveal whether pollution is higher during colder months or specific weather conditions.")
            else:
                st.warning("season column not found.")

        elif chart_choice == "PM2.5 Trend Over Time":
            if "datetime" in data.columns:
                trend = data.dropna(subset=["datetime"]).groupby("datetime", as_index=False)["PM2.5"].mean()
                # Resample/aggregate monthly for performance and readability.
                trend["month_period"] = trend["datetime"].dt.to_period("M").astype(str)
                monthly_trend = trend.groupby("month_period", as_index=False)["PM2.5"].mean()
                fig = px.line(monthly_trend, x="month_period", y="PM2.5", title="Monthly Average PM2.5 Trend")
                st.plotly_chart(fig, use_container_width=True)
                render_insight("The time-series plot shows how PM2.5 changes across months and years, supporting temporal interpretation.")
            else:
                st.warning("datetime column not found.")

        elif chart_choice == "PM2.5 vs Temperature":
            fig = px.scatter(data.sample(min(8000, len(data)), random_state=42), x="TEMP", y="PM2.5", color="station", opacity=0.45, title="PM2.5 vs Temperature")
            st.plotly_chart(fig, use_container_width=True)
            render_insight("This bivariate plot helps assess the relationship between temperature and particulate pollution.")

        elif chart_choice == "NO2 vs O3":
            fig = px.scatter(data.sample(min(8000, len(data)), random_state=42), x="NO2", y="O3", color="station", opacity=0.45, title="NO2 vs O3")
            st.plotly_chart(fig, use_container_width=True)
            render_insight("This pollutant interaction plot supports analysis of gas pollutants and possible atmospheric relationships.")

    with eda_tab4:
        st.subheader("Correlation Heatmap")
        num_cols = numeric_columns(data)
        selected_cols = st.multiselect("Select numeric variables", num_cols, default=[col for col in FEATURES + ["PM2.5"] if col in num_cols])

        if len(selected_cols) >= 2:
            corr = data[selected_cols].corr()
            fig = px.imshow(corr, text_auto=".2f", aspect="auto", title="Correlation Matrix")
            st.plotly_chart(fig, use_container_width=True)
            render_insight("Correlation analysis helps identify variables that move together and supports feature selection for model building.")
        else:
            st.warning("Select at least two numeric columns.")

# ============================================================
# 9. TASK 3: MODEL BUILDING PAGE
# ============================================================

elif page == "🤖 Task 3: Model Building":
    st.header("🤖 Task 3: Model Building")

    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader("Modelling Approach")
    st.write(
        """
        A Random Forest Regressor was used to predict PM2.5 because it can model non-linear relationships
        between pollutant concentrations and meteorological conditions. The features used for prediction are listed below.
        """
    )
    st.write(FEATURES)
    st.markdown("</div>", unsafe_allow_html=True)

    st.subheader("Model Performance")
    col1, col2 = st.columns(2)
    col1.metric("RMSE", "20.71")
    col2.metric("R² Score", "0.93")
    render_insight("The R² score of about 0.93 suggests the model explains approximately 93% of the variation in PM2.5 levels.")

    st.subheader("PM2.5 Prediction Tool")

    if model is None or scaler is None:
        st.error("model.pkl or scaler.pkl was not found. Save both files before using prediction.")
        st.stop()

    # Sample values are provided to make the tool easy to test during presentation.
    colA, colB = st.columns(2)
    with colA:
        PM10 = st.number_input("PM10", value=50.0, min_value=0.0)
        SO2 = st.number_input("SO2", value=10.0, min_value=0.0)
        NO2 = st.number_input("NO2", value=30.0, min_value=0.0)
        CO = st.number_input("CO", value=500.0, min_value=0.0)
        O3 = st.number_input("O3", value=40.0, min_value=0.0)

    with colB:
        TEMP = st.number_input("Temperature", value=20.0)
        PRES = st.number_input("Pressure", value=1010.0)
        DEWP = st.number_input("Dew Point", value=5.0)
        RAIN = st.number_input("Rain", value=0.0, min_value=0.0)
        WSPM = st.number_input("Wind Speed", value=2.0, min_value=0.0)

    if st.button("Predict PM2.5", type="primary"):
        user_input = np.array([[PM10, SO2, NO2, CO, O3, TEMP, PRES, DEWP, RAIN, WSPM]])
        user_input_scaled = scaler.transform(user_input)
        prediction = float(model.predict(user_input_scaled)[0])
        category, icon, advice = pm25_category(prediction)

        c1, c2, c3 = st.columns(3)
        c1.metric("Predicted PM2.5", f"{prediction:.2f}")
        c2.metric("Category", f"{icon} {category}")
        c3.metric("Target", "Lower is better")

        st.markdown(f"<div class='warning-box'><b>Health interpretation:</b> {advice}</div>", unsafe_allow_html=True)

# ============================================================
# 10. TASK 4: APPLICATION DEVELOPMENT PAGE
# ============================================================

elif page == "💻 Task 4: Application Development":
    st.header("💻 Task 4: Application Development")

    st.write(
        """
        The Streamlit application was developed to provide an interactive graphical user interface. It includes separate
        pages for data handling, EDA, model building, application documentation, version control evidence, and presentation support.
        """
    )

    app_features = pd.DataFrame({
        "Feature": [
            "Sidebar navigation",
            "Dataset preview",
            "Interactive charts",
            "Correlation heatmap",
            "PM2.5 prediction tool",
            "AQI-style category interpretation",
            "Presentation guide"
        ],
        "Purpose": [
            "Allows users to move between rubric sections.",
            "Shows the merged and cleaned dataset.",
            "Supports univariate, bivariate, and temporal analysis.",
            "Supports multivariate analysis.",
            "Allows users to test model predictions.",
            "Makes numerical prediction meaningful to non-technical users.",
            "Helps explain the project during presentation."
        ]
    })
    st.dataframe(app_features, use_container_width=True)

    st.code("streamlit run app.py", language="bash")

# ============================================================
# 11. TASK 5: VERSION CONTROL PAGE
# ============================================================

elif page == "🗂️ Task 5: Version Control":
    st.header("🗂️ Task 5: Version Control")

    st.write(
        """
        GitHub should be used to demonstrate project management and version control. The final report should include
        screenshots of the GitHub repository layout and commit history.
        """
    )

    commits = pd.DataFrame({
        "Suggested Commit Message": [
            "Created project folder structure",
            "Loaded and merged selected Beijing station datasets",
            "Handled missing values and engineered datetime features",
            "Added EDA visualisations and correlation heatmap",
            "Built Random Forest PM2.5 prediction model",
            "Saved model and scaler with Joblib",
            "Developed rubric-aligned Streamlit dashboard",
            "Updated README and final report screenshots"
        ]
    })
    st.dataframe(commits, use_container_width=True)

    st.info("Include screenshots of repository files, app.py, notebook, data folders, and commit history in your final report.")

# ============================================================
# 12. PRESENTATION GUIDE PAGE
# ============================================================

elif page == "🎤 Presentation Guide":
    st.header("🎤 Presentation Guide")

    st.write("Use this page as a quick speaking guide when presenting your application.")

    st.markdown(
        """
        ### Suggested presentation flow
        1. **Project aim:** Predict PM2.5 using Beijing air quality data.
        2. **Data handling:** Four stations were selected and merged.
        3. **Preprocessing:** Missing values were handled and new features were created.
        4. **EDA:** Visualisations revealed station, seasonal, and temporal pollution patterns.
        5. **Model building:** Random Forest was used to predict PM2.5.
        6. **Evaluation:** The model achieved RMSE ≈ 20.71 and R² ≈ 0.93.
        7. **Application:** Streamlit was used to build an interactive prediction dashboard.
        8. **Version control:** GitHub was used to track project development.
        """
    )

    st.success("Tip: During presentation, show Task 2 EDA first, then demonstrate the prediction tool in Task 3.")
