# ============================================================
# CMP7005 PRAC1: Beijing Air Quality Prediction System
# Streamlit Cloud Application
# Student: Ugwuoke Stanley Arinze
#
# Purpose:
# This app presents the full workflow required by the rubric:
# Task 1: Data Handling
# Task 2: Exploratory Data Analysis
# Task 3: Model Building
# Task 4: Application Development
# Task 5: Version Control
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Beijing Air Quality Prediction App",
    page_icon="🌍",
    layout="wide"
)


# ============================================================
# CUSTOM PAGE STYLE
# ============================================================

st.markdown(
    """
    <style>
    .main {
        background-color: #f7f9fc;
    }

    .title-text {
        font-size: 42px;
        font-weight: 800;
        color: #1f2937;
        margin-bottom: 10px;
    }

    .subtitle-text {
        font-size: 18px;
        color: #4b5563;
        margin-bottom: 25px;
    }

    .info-box {
        background-color: #ffffff;
        padding: 22px;
        border-radius: 16px;
        border: 1px solid #e5e7eb;
        margin-bottom: 18px;
        box-shadow: 0px 2px 8px rgba(0,0,0,0.04);
    }

    .small-note {
        font-size: 14px;
        color: #6b7280;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD DATA, MODEL AND SCALER
# ============================================================

@st.cache_data
def load_dataset():
    """
    Loads the cleaned Beijing air quality dataset.

    The dataset should be placed in the same GitHub repository
    as this app.py file.
    """
    file_path = "beijing_air_quality_cleaned.csv"

    if os.path.exists(file_path):
        df = pd.read_csv(file_path)

        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")

        return df

    return None


@st.cache_resource
def load_machine_learning_files():
    """
    Loads the trained Random Forest model and StandardScaler.
    These files must be included in the GitHub repository.
    """
    model_path = "model.pkl"
    scaler_path = "scaler.pkl"

    if os.path.exists(model_path) and os.path.exists(scaler_path):
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler

    return None, None


data = load_dataset()
model, scaler = load_machine_learning_files()


# ============================================================
# PM2.5 CATEGORY FUNCTION
# ============================================================

def get_pm25_category(pm25_value):
    """
    Converts predicted PM2.5 value into a simple air quality category.
    This makes the model output easier for users to understand.
    """

    if pm25_value <= 50:
        return "Good 🟢", "Air quality is acceptable."
    elif pm25_value <= 100:
        return "Moderate 🟡", "Sensitive individuals should reduce prolonged outdoor activity."
    elif pm25_value <= 150:
        return "Unhealthy for Sensitive Groups 🟠", "People with respiratory conditions should be careful."
    elif pm25_value <= 200:
        return "Unhealthy 🔴", "Everyone may begin to experience health effects."
    else:
        return "Very Unhealthy / Hazardous 🟣", "Outdoor activity should be limited where possible."


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

st.sidebar.title("📌 Navigation")

page = st.sidebar.radio(
    "Select Section",
    [
        "🏠 Home",
        "📂 Task 1: Data Handling",
        "📊 Task 2: EDA",
        "🤖 Task 3: Model Building",
        "💻 Task 4: Application Development",
        "🗂️ Task 5: Version Control",
        "📌 Conclusion"
    ]
)


# ============================================================
# HOME PAGE
# ============================================================

if page == "🏠 Home":

    st.markdown('<div class="title-text">🌍 Beijing Air Quality Prediction System</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle-text">CMP7005 Programming for Data Analysis — From Data to Application Development</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="info-box">
        This Streamlit application presents a complete data analysis and machine learning workflow
        for predicting PM2.5 air pollution levels in Beijing. The project uses selected urban and
        suburban monitoring stations, performs exploratory data analysis, builds a prediction model,
        and presents the results through an interactive web application.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.subheader("🎯 Project Aim")

    st.write(
        """
        The aim of this project is to develop a data-driven air quality prediction system using Python.
        The system analyses pollutant and meteorological variables and predicts PM2.5 concentration
        using a machine learning model.
        """
    )

    st.subheader("✅ Project Workflow")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.info("📂 Data Handling\n\nImport, combine and organise selected station datasets.")

    with col2:
        st.info("📊 EDA\n\nExplore data patterns, relationships, missing values and visual trends.")

    with col3:
        st.info("🤖 Model Building\n\nTrain and evaluate a Random Forest model for PM2.5 prediction.")

    col4, col5 = st.columns(2)

    with col4:
        st.success("💻 Application Development\n\nBuild an interactive Streamlit GUI.")

    with col5:
        st.success("🗂️ Version Control\n\nUse GitHub for project management and commits.")


# ============================================================
# TASK 1: DATA HANDLING
# ============================================================

elif page == "📂 Task 1: Data Handling":

    st.header("📂 Task 1: Data Handling")

    st.markdown(
        """
        <div class="info-box">
        This section explains how the data was selected, imported, merged and prepared
        for analysis. Four monitoring stations were used: two urban and two suburban.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.subheader("Selected Monitoring Stations")

    station_selection = pd.DataFrame(
        {
            "Station": ["Dongsi", "Guanyuan", "Shunyi", "Huairou"],
            "Station Type": ["Urban", "Urban", "Suburban", "Suburban"],
            "Justification": [
                "Represents inner-city air quality conditions.",
                "Represents another urban monitoring location.",
                "Represents suburban air quality conditions.",
                "Represents outer/suburban air quality conditions."
            ]
        }
    )

    st.dataframe(station_selection, use_container_width=True)

    st.subheader("Dataset Loading Status")

    if data is not None:
        st.success("Cleaned dataset loaded successfully.")

        col1, col2, col3 = st.columns(3)

        col1.metric("Number of Rows", f"{data.shape[0]:,}")
        col2.metric("Number of Columns", data.shape[1])

        if "station" in data.columns:
            col3.metric("Number of Stations", data["station"].nunique())
        else:
            col3.metric("Number of Stations", "N/A")

        st.subheader("Dataset Preview")
        st.dataframe(data.head(10), use_container_width=True)

    else:
        st.error(
            """
            Dataset not found. Please upload `beijing_air_quality_cleaned.csv`
            to the same GitHub repository as app.py.
            """
        )


# ============================================================
# TASK 2: EXPLORATORY DATA ANALYSIS
# ============================================================

elif page == "📊 Task 2: EDA":

    st.header("📊 Task 2: Exploratory Data Analysis")

    if data is None:
        st.error("Dataset not found. Please add beijing_air_quality_cleaned.csv to your GitHub repository.")

    else:
        st.markdown(
            """
            <div class="info-box">
            This section presents data understanding, preprocessing summary, statistical analysis
            and visualisation. It helps identify pollution patterns across stations, seasons and time.
            </div>
            """,
            unsafe_allow_html=True
        )

        eda_tab1, eda_tab2, eda_tab3, eda_tab4 = st.tabs(
            [
                "2.1 Data Understanding",
                "2.2 Preprocessing",
                "2.3 Visualisation",
                "2.4 Key Insights"
            ]
        )

        # ------------------------------------------------------------
        # DATA UNDERSTANDING TAB
        # ------------------------------------------------------------

        with eda_tab1:

            st.subheader("Dataset Structure")

            col1, col2 = st.columns(2)

            with col1:
                st.write("Dataset shape:")
                st.code(f"{data.shape[0]} rows and {data.shape[1]} columns")

            with col2:
                st.write("Column names:")
                st.write(list(data.columns))

            st.subheader("Missing Values")

            missing_values = data.isnull().sum().reset_index()
            missing_values.columns = ["Column", "Missing Values"]

            st.dataframe(missing_values, use_container_width=True)

            st.subheader("Statistical Summary")

            st.dataframe(data.describe(), use_container_width=True)

        # ------------------------------------------------------------
        # PREPROCESSING TAB
        # ------------------------------------------------------------

        with eda_tab2:

            st.subheader("Preprocessing Steps Completed")

            st.write(
                """
                The following preprocessing steps were performed before analysis and modelling:
                """
            )

            preprocessing_steps = pd.DataFrame(
                {
                    "Step": [
                        "Datetime creation",
                        "Station type classification",
                        "Missing value handling",
                        "Duplicate removal",
                        "Season feature creation",
                        "Cleaned dataset export"
                    ],
                    "Purpose": [
                        "Combined year, month, day and hour into a usable time column.",
                        "Grouped stations into urban and suburban categories.",
                        "Filled numerical values using median and categorical values using mode.",
                        "Removed repeated records to improve data quality.",
                        "Created seasonal groups for temporal pollution analysis.",
                        "Saved cleaned data for modelling and Streamlit deployment."
                    ]
                }
            )

            st.dataframe(preprocessing_steps, use_container_width=True)

            st.info(
                """
                These preprocessing steps improved data quality and prepared the dataset for
                reliable exploratory analysis and machine learning.
                """
            )

        # ------------------------------------------------------------
        # VISUALISATION TAB
        # ------------------------------------------------------------

        with eda_tab3:

            st.subheader("Interactive Visualisation Section")

            chart_option = st.selectbox(
                "Choose a visualisation:",
                [
                    "PM2.5 Distribution",
                    "Average PM2.5 by Station",
                    "Average PM2.5 by Station Type",
                    "Average PM2.5 by Season",
                    "PM2.5 Trend Over Time",
                    "Correlation Heatmap"
                ]
            )

            if chart_option == "PM2.5 Distribution":

                st.write("This chart shows the distribution of PM2.5 values.")

                fig, ax = plt.subplots(figsize=(10, 5))
                sns.histplot(data["PM2.5"], bins=50, kde=True, ax=ax)
                ax.set_title("Distribution of PM2.5 Concentration")
                ax.set_xlabel("PM2.5")
                ax.set_ylabel("Frequency")
                st.pyplot(fig)

                st.info(
                    """
                    Interpretation: The PM2.5 distribution is usually right-skewed,
                    meaning most values are moderate but some pollution episodes are very high.
                    """
                )

            elif chart_option == "Average PM2.5 by Station":

                st.write("This chart compares average PM2.5 concentration across the selected stations.")

                station_avg = data.groupby("station")["PM2.5"].mean().sort_values(ascending=False)

                st.bar_chart(station_avg)

                st.info(
                    """
                    Interpretation: Differences between stations suggest that location influences
                    pollution exposure and air quality patterns.
                    """
                )

            elif chart_option == "Average PM2.5 by Station Type":

                if "station_type" in data.columns:
                    type_avg = data.groupby("station_type")["PM2.5"].mean()
                    st.bar_chart(type_avg)

                    st.info(
                        """
                        Interpretation: This comparison helps identify whether urban stations
                        experience higher PM2.5 concentration than suburban stations.
                        """
                    )
                else:
                    st.warning("station_type column not found in the dataset.")

            elif chart_option == "Average PM2.5 by Season":

                if "season" in data.columns:
                    season_order = ["Spring", "Summer", "Autumn", "Winter"]
                    season_avg = data.groupby("season")["PM2.5"].mean()
                    season_avg = season_avg.reindex([s for s in season_order if s in season_avg.index])

                    st.bar_chart(season_avg)

                    st.info(
                        """
                        Interpretation: Seasonal variation helps explain how weather conditions
                        and human activities influence pollution levels.
                        """
                    )
                else:
                    st.warning("season column not found in the dataset.")

            elif chart_option == "PM2.5 Trend Over Time":

                if "datetime" in data.columns:
                    trend = data.groupby("datetime")["PM2.5"].mean()

                    st.line_chart(trend)

                    st.info(
                        """
                        Interpretation: The time-series trend helps identify periods of high and low
                        pollution across the dataset.
                        """
                    )
                else:
                    st.warning("datetime column not found in the dataset.")

            elif chart_option == "Correlation Heatmap":

                st.write("This heatmap shows relationships between numerical variables.")

                numeric_data = data.select_dtypes(include=["float64", "int64"])

                fig, ax = plt.subplots(figsize=(12, 8))
                sns.heatmap(numeric_data.corr(), annot=False, cmap="coolwarm", ax=ax)
                ax.set_title("Correlation Heatmap of Numerical Variables")
                st.pyplot(fig)

                st.info(
                    """
                    Interpretation: Strong correlations between pollutant variables suggest that
                    some pollutants may increase or decrease together under similar environmental conditions.
                    """
                )

        # ------------------------------------------------------------
        # KEY INSIGHTS TAB
        # ------------------------------------------------------------

        with eda_tab4:

            st.subheader("Main EDA Insights")

            st.write(
                """
                From the exploratory analysis, the following insights were identified:
                """
            )

            st.success(
                """
                1. PM2.5 values show variation across monitoring stations.
                """
            )

            st.success(
                """
                2. Seasonal patterns suggest that weather and time of year influence pollution levels.
                """
            )

            st.success(
                """
                3. Pollutant variables such as PM10, NO2, SO2 and CO are useful for understanding PM2.5 behaviour.
                """
            )

            st.success(
                """
                4. Meteorological variables such as temperature, pressure, dew point, rainfall and wind speed
                help explain environmental effects on air pollution.
                """
            )


# ============================================================
# TASK 3: MODEL BUILDING
# ============================================================

elif page == "🤖 Task 3: Model Building":

    st.header("🤖 Task 3: Model Building")

    st.markdown(
        """
        <div class="info-box">
        A Random Forest Regressor was used to predict PM2.5 concentration.
        The model uses air pollutant and meteorological features as input variables.
        </div>
        """,
        unsafe_allow_html=True
    )

    model_tab1, model_tab2, model_tab3 = st.tabs(
        [
            "Model Description",
            "Model Performance",
            "Prediction Tool"
        ]
    )

    with model_tab1:

        st.subheader("Selected Features")

        features = ["PM10", "SO2", "NO2", "CO", "O3", "TEMP", "PRES", "DEWP", "RAIN", "WSPM"]

        feature_table = pd.DataFrame(
            {
                "Feature": features,
                "Description": [
                    "Particulate matter smaller than 10 micrometres",
                    "Sulphur dioxide",
                    "Nitrogen dioxide",
                    "Carbon monoxide",
                    "Ozone",
                    "Temperature",
                    "Atmospheric pressure",
                    "Dew point",
                    "Rainfall",
                    "Wind speed"
                ]
            }
        )

        st.dataframe(feature_table, use_container_width=True)

        st.write(
            """
            Random Forest was selected because it can model non-linear relationships
            between pollutants, weather variables and PM2.5 concentration.
            """
        )

    with model_tab2:

        st.subheader("Model Evaluation Metrics")

        col1, col2 = st.columns(2)

        col1.metric("RMSE", "20.71")
        col2.metric("R² Score", "0.93")

        st.success(
            """
            The R² score of 0.93 indicates that the model explains approximately 93%
            of the variation in PM2.5 concentration.
            """
        )

        st.info(
            """
            RMSE shows the average prediction error. A lower RMSE means the model predictions
            are closer to the actual PM2.5 values.
            """
        )

    with model_tab3:

        st.subheader("PM2.5 Prediction Tool")

        if model is None or scaler is None:
            st.error("model.pkl or scaler.pkl not found. Please upload both files to the GitHub repository.")

        else:
            st.write("Enter pollutant and meteorological values below:")

            col1, col2 = st.columns(2)

            with col1:
                PM10 = st.number_input("PM10", min_value=0.0, value=50.0)
                SO2 = st.number_input("SO2", min_value=0.0, value=10.0)
                NO2 = st.number_input("NO2", min_value=0.0, value=30.0)
                CO = st.number_input("CO", min_value=0.0, value=500.0)
                O3 = st.number_input("O3", min_value=0.0, value=40.0)

            with col2:
                TEMP = st.number_input("Temperature", value=20.0)
                PRES = st.number_input("Pressure", value=1010.0)
                DEWP = st.number_input("Dew Point", value=5.0)
                RAIN = st.number_input("Rain", min_value=0.0, value=0.0)
                WSPM = st.number_input("Wind Speed", min_value=0.0, value=2.0)

            if st.button("Predict PM2.5"):

                input_data = np.array([[PM10, SO2, NO2, CO, O3, TEMP, PRES, DEWP, RAIN, WSPM]])
                input_scaled = scaler.transform(input_data)
                prediction = model.predict(input_scaled)

                pm25_value = float(prediction[0])
                category, advice = get_pm25_category(pm25_value)

                st.subheader("Prediction Result")

                col1, col2 = st.columns(2)

                col1.metric("Predicted PM2.5", f"{pm25_value:.2f}")
                col2.metric("Air Quality Category", category)

                st.warning(advice)


# ============================================================
# TASK 4: APPLICATION DEVELOPMENT
# ============================================================

elif page == "💻 Task 4: Application Development":

    st.header("💻 Task 4: Application Development")

    st.write(
        """
        This application was developed using Streamlit to provide a clear graphical user interface.
        The application is divided into multiple pages that follow the assignment rubric.
        """
    )

    app_features = pd.DataFrame(
        {
            "Application Feature": [
                "Sidebar navigation",
                "Data handling section",
                "EDA section",
                "Visualisation section",
                "Model building section",
                "PM2.5 prediction tool",
                "AQI-style interpretation",
                "Version control documentation"
            ],
            "Purpose": [
                "Allows users to move between project sections.",
                "Shows data selection and merging process.",
                "Explains data understanding and preprocessing.",
                "Displays charts and analytical insights.",
                "Explains the machine learning method.",
                "Allows users to input values and obtain predictions.",
                "Makes predictions easier to understand.",
                "Shows GitHub usage and project organisation."
            ]
        }
    )

    st.dataframe(app_features, use_container_width=True)

    st.success(
        """
        The application demonstrates how Python can be used to move from raw data
        to a working interactive machine learning system.
        """
    )


# ============================================================
# TASK 5: VERSION CONTROL
# ============================================================

elif page == "🗂️ Task 5: Version Control":

    st.header("🗂️ Task 5: Version Control")

    st.write(
        """
        GitHub was used to organise the project files and track development progress.
        Clear commit messages help demonstrate how the project developed over time.
        """
    )

    st.subheader("Recommended GitHub Repository Structure")

    st.code(
        """
CMP7005_Air_Quality_App/
│── app.py
│── requirements.txt
│── model.pkl
│── scaler.pkl
│── beijing_air_quality_cleaned.csv
│── notebook/
│   └── CMP7005_PRAC1_Notebook.ipynb
│── figures/
│   └── visualisation_screenshots.png
│── README.md
        """
    )

    st.subheader("Suggested Commit Messages")

    commit_table = pd.DataFrame(
        {
            "Commit Message": [
                "Created project folder structure",
                "Loaded and merged Beijing air quality datasets",
                "Handled missing values and created datetime features",
                "Completed exploratory data analysis visualisations",
                "Built Random Forest PM2.5 prediction model",
                "Saved model and scaler files",
                "Created Streamlit application interface",
                "Updated README and final project documentation"
            ]
        }
    )

    st.dataframe(commit_table, use_container_width=True)

    st.info(
        """
        Include screenshots of your GitHub repository layout and commit history in your final report.
        """
    )


# ============================================================
# CONCLUSION
# ============================================================

elif page == "📌 Conclusion":

    st.header("📌 Conclusion")

    st.write(
        """
        This project successfully developed a complete air quality prediction system.
        It combined data handling, preprocessing, exploratory data analysis,
        machine learning and application development into one interactive system.
        """
    )

    st.success(
        """
        The final Streamlit application demonstrates how data analysis can be transformed
        into a practical decision-support tool for PM2.5 air pollution prediction.
        """
    )

    st.write(
        """
        The project also supports professional software development practices by using
        GitHub version control and organised project documentation.
        """
    )
