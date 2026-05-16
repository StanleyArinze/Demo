
# ============================================================
# CMP7005 PRAC1 - Beijing Air Quality Prediction Dashboard
# Streamlit Cloud Application
# Student: Ugwuoke Stanley Arinze
#
# This app follows the assessment workflow:
# 1. Dataset Overview
# 2. Beautiful Visualisations / EDA
# 3. Model Outputs / PM2.5 Prediction
# 4. Project Summary
# ============================================================

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Beijing Air Quality Dashboard",
    page_icon="🌍",
    layout="wide"
)


# ============================================================
# CUSTOM CSS FOR PROFESSIONAL DESIGN
# ============================================================

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #FDF6E3 0%, #F7F9FC 50%, #EAF4F4 100%);
    }

    .main-title {
        font-size: 44px;
        font-weight: 900;
        color: #1F2937;
        margin-bottom: 0px;
    }

    .subtitle {
        font-size: 18px;
        color: #4B5563;
        margin-bottom: 25px;
    }

    .section-card {
        background-color: white;
        padding: 22px;
        border-radius: 18px;
        border: 1px solid #E5E7EB;
        box-shadow: 0px 4px 14px rgba(0,0,0,0.06);
        margin-bottom: 18px;
    }

    .insight-box {
        background-color: #FFF7ED;
        padding: 18px;
        border-radius: 14px;
        border: 1px solid #FDBA74;
        margin-top: 14px;
        margin-bottom: 14px;
    }

    .success-box {
        background-color: #ECFDF5;
        padding: 18px;
        border-radius: 14px;
        border: 1px solid #6EE7B7;
        margin-top: 14px;
        margin-bottom: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD DATASET
# ============================================================

@st.cache_data
def load_data():
    """
    Load cleaned air quality dataset from GitHub repository.
    """
    file_path = "beijing_air_quality_cleaned.csv"

    if not os.path.exists(file_path):
        return pd.DataFrame()

    df = pd.read_csv(file_path)

    # Convert datetime column if available
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")

    # Create Month column if missing
    if "Month" not in df.columns:
        if "month" in df.columns:
            df["Month"] = df["month"]
        elif "datetime" in df.columns:
            df["Month"] = df["datetime"].dt.month

    # Create Hour column if missing
    if "Hour" not in df.columns:
        if "hour" in df.columns:
            df["Hour"] = df["hour"]
        elif "datetime" in df.columns:
            df["Hour"] = df["datetime"].dt.hour

    # Create station category if missing
    if "category" not in df.columns:
        if "station_type" in df.columns:
            df["category"] = df["station_type"]
        elif "station" in df.columns:
            urban_stations = ["Dongsi", "Guanyuan"]
            df["category"] = df["station"].apply(
                lambda x: "Inner (Urban)" if x in urban_stations else "Outer (Suburban)"
            )

    # Create season if missing
    if "season" not in df.columns and "Month" in df.columns:
        def get_season(month):
            if month in [12, 1, 2]:
                return "Winter"
            elif month in [3, 4, 5]:
                return "Spring"
            elif month in [6, 7, 8]:
                return "Summer"
            else:
                return "Autumn"

        df["season"] = df["Month"].apply(get_season)

    return df


# ============================================================
# LOAD MODEL AND SCALER
# ============================================================

@st.cache_resource
def load_model_files():
    """
    Load trained machine learning model and scaler.
    """
    if os.path.exists("model.pkl") and os.path.exists("scaler.pkl"):
        model = joblib.load("model.pkl")
        scaler = joblib.load("scaler.pkl")
        return model, scaler

    return None, None


df = load_data()
model, scaler = load_model_files()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_air_quality_category(pm25):
    """
    Convert predicted PM2.5 value into an understandable air quality category.
    """
    if pm25 < 35:
        return "Good 🟢", "Air quality is good."
    elif pm25 < 75:
        return "Moderate 🟡", "Air quality is moderate. Sensitive groups should take care."
    else:
        return "Unhealthy 🔴", "Air quality is unhealthy. Outdoor exposure should be reduced."


def apply_plotly_style(fig, title):
    """
    Apply consistent professional style to Plotly charts.
    """
    fig.update_layout(
        title=dict(
            text=title,
            x=0.02,
            font=dict(size=23, color="#1F2937")
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="white",
        font=dict(color="#2D3436"),
        margin=dict(l=40, r=40, t=75, b=45),
        height=520,
        legend=dict(
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#E5E7EB",
            borderwidth=1
        )
    )

    return fig


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

st.sidebar.title("📌 Navigation")

page = st.sidebar.radio(
    "Go to:",
    [
        "🏠 Home",
        "📂 Dataset Overview",
        "📊 Beautiful Visualisations",
        "🤖 Model Outputs",
        "📌 Project Summary"
    ]
)

st.sidebar.markdown("---")

if not df.empty:
    st.sidebar.success("Dataset loaded successfully")
else:
    st.sidebar.error("Dataset not found")


# ============================================================
# HOME PAGE
# ============================================================

if page == "🏠 Home":

    st.markdown(
        '<div class="main-title">🌍 Beijing Air Quality Dashboard</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">CMP7005 Programming for Data Analysis | Data Handling, EDA, Model Building and Streamlit Application</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="section-card">
        This dashboard presents a complete air quality data science workflow.
        It analyses Beijing air pollution data, explores pollutant patterns, and predicts PM2.5 levels
        using a trained machine learning model.
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.info("📂 **Dataset Overview**\n\nView dataset structure, station filters, missing values and summary statistics.")

    with col2:
        st.warning("📊 **Beautiful EDA Visualisations**\n\nExplore pollutant distributions, station comparison, weather relationships and correlations.")

    with col3:
        st.success("🤖 **Model Outputs**\n\nPredict PM2.5 level and interpret the air quality category.")

    st.subheader("Why PM2.5 Matters")

    st.write(
        """
        PM2.5 refers to fine particulate matter that can enter the lungs and bloodstream.
        Analysing and predicting PM2.5 is important for environmental monitoring,
        public health awareness, and air quality decision-making.
        """
    )


# ============================================================
# DATASET OVERVIEW PAGE
# ============================================================

elif page == "📂 Dataset Overview":

    st.title("📂 Dataset Overview")

    if df.empty:
        st.error("Dataset not found. Please upload `Task2_Cleaned_Data.csv` to your GitHub repository.")

    else:
        st.markdown(
            """
            <div class="section-card">
            This section provides a clear overview of the cleaned Beijing air quality dataset.
            It includes dataset size, station filtering, missing value checks and statistical summary.
            </div>
            """,
            unsafe_allow_html=True
        )

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Rows", f"{df.shape[0]:,}")
        col2.metric("Columns", df.shape[1])

        if "station" in df.columns:
            col3.metric("Stations", df["station"].nunique())
        else:
            col3.metric("Stations", "N/A")

        if "PM2.5" in df.columns:
            col4.metric("Average PM2.5", f"{df['PM2.5'].mean():.2f}")
        else:
            col4.metric("Average PM2.5", "N/A")

        filtered_df = df.copy()

        st.subheader("Filter Dataset")

        if "station" in df.columns:
            selected_stations = st.multiselect(
                "Select station(s):",
                sorted(df["station"].dropna().unique()),
                default=list(sorted(df["station"].dropna().unique()))
            )

            if selected_stations:
                filtered_df = filtered_df[filtered_df["station"].isin(selected_stations)]

        st.subheader("Dataset Preview")
        st.dataframe(filtered_df.head(100), use_container_width=True)

        tab1, tab2, tab3 = st.tabs(
            ["Missing Values", "Statistical Summary", "Column Information"]
        )

        with tab1:
            missing_df = filtered_df.isnull().sum().reset_index()
            missing_df.columns = ["Column", "Missing Values"]
            st.dataframe(missing_df, use_container_width=True)

        with tab2:
            st.dataframe(filtered_df.describe(), use_container_width=True)

        with tab3:
            column_info = pd.DataFrame(
                {
                    "Column": filtered_df.columns,
                    "Data Type": filtered_df.dtypes.astype(str).values
                }
            )
            st.dataframe(column_info, use_container_width=True)

        st.markdown(
            """
            <div class="insight-box">
            <b>Interpretation:</b> This dataset overview confirms that the cleaned dataset is ready
            for exploratory analysis and machine learning.
            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# BEAUTIFUL VISUALISATIONS PAGE
# ============================================================

elif page == "📊 Beautiful Visualisations":

    st.title("📊 Beautiful Interactive Visualisations")

    if df.empty:
        st.error("Dataset not found. Please upload `Task2_Cleaned_Data.csv` to your GitHub repository.")

    else:
        st.markdown(
            """
            <div class="section-card">
            This section uses interactive Plotly charts to explore pollution patterns.
            The charts are designed to look professional and support strong interpretation in your presentation.
            </div>
            """,
            unsafe_allow_html=True
        )

        pollutant_options = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3"]
        available_pollutants = [col for col in pollutant_options if col in df.columns]

        if not available_pollutants:
            st.error("No pollutant columns found in the dataset.")

        else:
            selected_pollutant = st.selectbox(
                "Select pollutant for analysis:",
                available_pollutants
            )
visual_tab1, visual_tab2, visual_tab3, visual_tab4, visual_tab5, visual_tab6, visual_tab7, visual_tab8 = st.tabs(
    [
        "Distribution",
        "Station Comparison",
        "Weather Relationship",
        "Seasonal Pattern",
        "Trend Over Time",
        "Temperature Distribution",
        "Correlation Heatmap",
        "Actual vs Predicted"
    ]
)

            # --------------------------------------------------------
            # TAB 1: DISTRIBUTION
            # --------------------------------------------------------

        with visual_tab1:

                color_column = "category" if "category" in df.columns else None

                fig = px.histogram(
                    df,
                    x=selected_pollutant,
                    color=color_column,
                    marginal="box",
                    nbins=60,
                    opacity=0.75,
                    color_discrete_sequence=px.colors.qualitative.Set2
                )

                fig = apply_plotly_style(
                    fig,
                    f"Distribution of {selected_pollutant} Concentration"
                )

                st.plotly_chart(fig, use_container_width=True)

                st.markdown(
                    """
                    <div class="insight-box">
                    <b>Interpretation:</b> This distribution shows how the pollutant values are spread.
                    The marginal boxplot helps identify outliers and extreme pollution episodes.
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # --------------------------------------------------------
            # TAB 2: STATION COMPARISON
            # --------------------------------------------------------

            with visual_tab2:

                if "station" not in df.columns:
                    st.warning("Station column not found.")

                else:
                    fig = px.box(
                        df,
                        x="station",
                        y=selected_pollutant,
                        color="station",
                        points=False,
                        color_discrete_sequence=px.colors.qualitative.Bold
                    )

                    fig = apply_plotly_style(
                        fig,
                        f"{selected_pollutant} Distribution Across Stations"
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    station_avg = (
                        df.groupby("station")[selected_pollutant]
                        .mean()
                        .sort_values(ascending=False)
                        .reset_index()
                    )

                    fig2 = px.bar(
                        station_avg,
                        x="station",
                        y=selected_pollutant,
                        color=selected_pollutant,
                        color_continuous_scale="YlOrRd",
                        text_auto=".2f"
                    )

                    fig2 = apply_plotly_style(
                        fig2,
                        f"Average {selected_pollutant} by Station"
                    )

                    st.plotly_chart(fig2, use_container_width=True)

                    st.markdown(
                        """
                        <div class="insight-box">
                        <b>Interpretation:</b> Station-level comparison shows whether pollution levels
                        are higher in specific monitoring locations.
                        </div>
                        """,
                        unsafe_allow_html=True
                    )


            # --------------------------------------------------------
            # TAB 3: WEATHER RELATIONSHIP
            # --------------------------------------------------------

            with visual_tab3:

                if "TEMP" in df.columns:

                    sample_df = df.sample(min(8000, len(df)), random_state=42)

                    fig = px.density_heatmap(
                        sample_df,
                        x="TEMP",
                        y=selected_pollutant,
                        nbinsx=45,
                        nbinsy=45,
                        color_continuous_scale="YlOrRd"
                    )

                    fig = apply_plotly_style(
                        fig,
                        f"Density Heatmap: Temperature vs {selected_pollutant}"
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    st.markdown(
                        """
                        <div class="insight-box">
                        <b>Interpretation:</b> The density heatmap is more effective than a normal
                        scatterplot for large datasets because it shows where observations are most concentrated.
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                else:
                    st.warning("TEMP column not found.")

                if "WSPM" in df.columns:

                    sample_df2 = df.sample(min(8000, len(df)), random_state=24)

                    color_column = "category" if "category" in sample_df2.columns else None

                    fig2 = px.scatter(
                        sample_df2,
                        x="WSPM",
                        y=selected_pollutant,
                        color=color_column,
                        opacity=0.55,
                        color_discrete_sequence=px.colors.qualitative.Set1
                    )

                    fig2 = apply_plotly_style(
                        fig2,
                        f"Wind Speed vs {selected_pollutant}"
                    )

                    st.plotly_chart(fig2, use_container_width=True)

                    st.markdown(
                        """
                        <div class="insight-box">
                        <b>Interpretation:</b> Wind speed is important because stronger wind can disperse
                        pollutants and reduce pollutant concentration around monitoring stations.
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                else:
                    st.warning("WSPM column not found.")
            # --------------------------------------------------------
            # TAB 4: SEASONAL PATTERN
            # --------------------------------------------------------

            with visual_tab4:

                if "season" not in df.columns:
                    st.warning("Season column not found.")

                else:
                    season_order = ["Spring", "Summer", "Autumn", "Winter"]

                    season_avg = (
                        df.groupby("season")[selected_pollutant]
                        .mean()
                        .reindex([s for s in season_order if s in df["season"].unique()])
                        .reset_index()
                    )

                    fig = px.bar(
                        season_avg,
                        x="season",
                        y=selected_pollutant,
                        color="season",
                        text_auto=".2f",
                        color_discrete_sequence=["#60A5FA", "#34D399", "#F59E0B", "#EF4444"]
                    )

                    fig = apply_plotly_style(
                        fig,
                        f"Average {selected_pollutant} by Season"
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    st.markdown(
                        """
                        <div class="insight-box">
                        <b>Interpretation:</b> Seasonal analysis helps identify whether pollution becomes
                        worse during particular periods of the year.
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                if "Month" in df.columns:

                    monthly_avg = (
                        df.groupby("Month")[selected_pollutant]
                        .mean()
                        .reset_index()
                    )

                    fig2 = px.line(
                        monthly_avg,
                        x="Month",
                        y=selected_pollutant,
                        markers=True,
                        line_shape="spline"
                    )

                    fig2.update_traces(
                        line=dict(width=4),
                        marker=dict(size=10)
                    )

                    fig2 = apply_plotly_style(
                        fig2,
                        f"Monthly Trend of {selected_pollutant}"
                    )

                    st.plotly_chart(fig2, use_container_width=True)
            # --------------------------------------------------------
            # TAB 5: TREND OVER TIME
            # --------------------------------------------------------

            with visual_tab5:

                st.subheader(f"{selected_pollutant} Trend Over Time")

                if "datetime" in df.columns:

                    trend_df = (
                        df.groupby("datetime")[selected_pollutant]
                        .mean()
                        .reset_index()
                    )

                    fig = px.line(
                        trend_df,
                        x="datetime",
                        y=selected_pollutant,
                        title=f"{selected_pollutant} Trend Over Time",
                        markers=False
                    )

                    fig.update_traces(line=dict(width=3))

                    fig = apply_plotly_style(
                        fig,
                        f"{selected_pollutant} Trend Over Time"
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    st.markdown(
                        """
                        <div class="insight-box">
                        <b>Interpretation:</b> This time-series chart shows how pollution levels change over time.
                        It helps identify periods of higher and lower air pollution.
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                elif "Month" in df.columns:

                    trend_df = (
                        df.groupby("Month")[selected_pollutant]
                        .mean()
                        .reset_index()
                    )

                    fig = px.line(
                        trend_df,
                        x="Month",
                        y=selected_pollutant,
                        markers=True,
                        line_shape="spline"
                    )

                    fig.update_traces(line=dict(width=4), marker=dict(size=10))

                    fig = apply_plotly_style(
                        fig,
                        f"Monthly Trend of {selected_pollutant}"
                    )

                    st.plotly_chart(fig, use_container_width=True)

                else:
                    st.warning("No datetime or month column found for trend analysis.")
            # --------------------------------------------------------
            # TAB 6: TEMPERATURE DISTRIBUTION
            # --------------------------------------------------------

            with visual_tab6:

                st.subheader("Temperature Distribution")

                if "TEMP" in df.columns:

                    color_column = "season" if "season" in df.columns else None

                    fig = px.histogram(
                        df,
                        x="TEMP",
                        color=color_column,
                        nbins=50,
                        marginal="box",
                        opacity=0.75,
                        color_discrete_sequence=px.colors.qualitative.Set2
                    )

                    fig = apply_plotly_style(
                        fig,
                        "Temperature Distribution"
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    st.markdown(
                        """
                        <div class="insight-box">
                        <b>Interpretation:</b> This chart shows the spread of temperature values in the dataset.
                        It also helps explain seasonal and weather-related effects on air pollution.
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                else:
                    st.warning("TEMP column not found in the dataset.")
            # --------------------------------------------------------
            # TAB 7: CORRELATION HEATMAP
            # --------------------------------------------------------

            with visual_tab7:

                numeric_df = df.select_dtypes(include=["float64", "int64"])

                if numeric_df.empty:
                    st.warning("No numerical columns found for correlation analysis.")

                else:
                    corr = numeric_df.corr()

                    fig = px.imshow(
                        corr,
                        text_auto=".2f",
                        color_continuous_scale="RdBu_r",
                        zmin=-1,
                        zmax=1,
                        aspect="auto"
                    )

                    fig = apply_plotly_style(
                        fig,
                        "Correlation Heatmap of Numerical Variables"
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    st.markdown(
                        """
                        <div class="insight-box">
                        <b>Interpretation:</b> The correlation heatmap shows the strength and direction
                        of relationships between numerical variables. Strong positive correlations between
                        pollutants suggest that some pollutants increase together.
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
            # --------------------------------------------------------
            # TAB 8: ACTUAL VS PREDICTED
            # --------------------------------------------------------

            with visual_tab8:

                st.subheader("Actual vs Predicted PM2.5")

                required_features = ["PM10", "SO2", "NO2", "CO", "O3", "TEMP", "PRES", "DEWP", "RAIN", "WSPM"]

                if model is None or scaler is None:
                    st.warning("Model or scaler file not found.")

                elif "PM2.5" not in df.columns:
                    st.warning("PM2.5 column not found in dataset.")

                elif not all(col in df.columns for col in required_features):
                    st.warning("Some required model features are missing from the dataset.")

                else:
                    model_df = df[required_features + ["PM2.5"]].dropna()

                    if len(model_df) > 3000:
                        model_df = model_df.sample(3000, random_state=42)

                    X_actual = model_df[required_features]
                    y_actual = model_df["PM2.5"]

                    try:
                        X_scaled = scaler.transform(X_actual)
                        y_predicted = model.predict(X_scaled)

                        actual_pred_df = pd.DataFrame(
                            {
                                "Actual PM2.5": y_actual,
                                "Predicted PM2.5": y_predicted
                            }
                        )

                        fig = px.scatter(
                            actual_pred_df,
                            x="Actual PM2.5",
                            y="Predicted PM2.5",
                            opacity=0.6,
                            title="Actual vs Predicted PM2.5"
                        )

                        fig.add_shape(
                            type="line",
                            x0=actual_pred_df["Actual PM2.5"].min(),
                            y0=actual_pred_df["Actual PM2.5"].min(),
                            x1=actual_pred_df["Actual PM2.5"].max(),
                            y1=actual_pred_df["Actual PM2.5"].max(),
                            line=dict(width=3, dash="dash")
                        )

                        fig = apply_plotly_style(
                            fig,
                            "Actual vs Predicted PM2.5"
                        )

                        st.plotly_chart(fig, use_container_width=True)

                        st.markdown(
                            """
                            <div class="insight-box">
                            <b>Interpretation:</b> Points closer to the diagonal line show better prediction accuracy.
                            This graph demonstrates how well the trained model predicts PM2.5 values compared with actual observations.
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                    except Exception as e:
                        st.error(f"Could not create Actual vs Predicted graph: {e}")


# ============================================================
# MODEL OUTPUTS PAGE
# ============================================================

elif page == "🤖 Model Outputs":

    st.title("🤖 PM2.5 Prediction Model")

    st.markdown(
        """
        <div class="section-card">
        This section uses the trained model to predict PM2.5 concentration from pollutant
        and meteorological variables.
        </div>
        """,
        unsafe_allow_html=True
    )

    if model is None or scaler is None:
        st.error("Model files not found. Please upload `model.pkl` and `scaler.pkl` to GitHub.")

    else:
        st.success("Model and scaler loaded successfully.")

        st.subheader("Enter Input Values")

        with st.form("prediction_form"):

            col1, col2 = st.columns(2)

            with col1:
                pm10 = st.number_input("PM10", min_value=0.0, value=80.0)
                so2 = st.number_input("SO2", min_value=0.0, value=15.0)
                no2 = st.number_input("NO2", min_value=0.0, value=40.0)
                co = st.number_input("CO", min_value=0.0, value=1000.0)
                o3 = st.number_input("O3", min_value=0.0, value=50.0)

            with col2:
                temp = st.number_input("Temperature", value=10.0)
                pres = st.number_input("Pressure", value=1015.0)
                dewp = st.number_input("Dew Point", value=0.0)
                rain = st.number_input("Rainfall", min_value=0.0, value=0.0)
                wspm = st.number_input("Wind Speed", min_value=0.0, value=2.0)

            submit = st.form_submit_button("Predict PM2.5 Level")

        if submit:

            input_data = pd.DataFrame(
                [
                    {
                        "PM10": pm10,
                        "SO2": so2,
                        "NO2": no2,
                        "CO": co,
                        "O3": o3,
                        "TEMP": temp,
                        "PRES": pres,
                        "DEWP": dewp,
                        "RAIN": rain,
                        "WSPM": wspm
                    }
                ]
            )

            try:
                input_scaled = scaler.transform(input_data)
                prediction = model.predict(input_scaled)

                predicted_pm25 = float(prediction[0])
                category_label, advice = get_air_quality_category(predicted_pm25)

                col1, col2 = st.columns(2)

                col1.metric("Predicted PM2.5", f"{predicted_pm25:.2f} μg/m³")
                col2.metric("Air Quality Category", category_label)

                if predicted_pm25 < 35:
                    st.success(advice)
                elif predicted_pm25 < 75:
                    st.warning(advice)
                else:
                    st.error(advice)

            except Exception as e:
                st.error(f"Prediction error: {e}")
                st.info(
                    """
                    This usually means your model was trained with different input columns.
                    The prediction input must match the same features used during training.
                    """
                )


# ============================================================
# PROJECT SUMMARY PAGE
# ============================================================

elif page == "📌 Project Summary":

    st.title("📌 Project Summary")

    st.markdown(
        """
        <div class="success-box">
        This project successfully connects data cleaning, exploratory data analysis,
        machine learning, and Streamlit Cloud deployment into one complete application.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.subheader("What the App Demonstrates")

    st.write(
        """
        - Cleaned Beijing air quality dataset
        - Interactive dataset overview
        - Beautiful Plotly visualisations
        - Station, seasonal, weather and correlation analysis
        - PM2.5 prediction model output
        - Air quality category interpretation
        """
    )

    st.subheader("Presentation Point")

    st.info(
        """
       The visualisation component provides an interactive and professional way to explore
    air pollution patterns across stations, seasons, weather variables and pollutant relationships.
    This improves the clarity of the analysis and supports evidence-based interpretation.
        """
    )
