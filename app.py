from streamlit.elements.lib.layout_utils import TextAlignment
import streamlit as st
import pandas as pd

from main import predict_irrigation_need


st.set_page_config(page_title="Irrigation Need Prediction", layout="wide")

def inject_custom_styles():
    st.markdown("""
        <style>
        /* Main background */
        .stApp {
            background-color: #f9fdf9;
        }

        /* Expander styling */
        .streamlit-expanderHeader {
            background-color: #f0f5ec !important;
            border-radius: 8px !important;
            border: 1px solid #d5e8d4 !important;
            color: #2c5530 !important;
            font-weight: 600 !important;
        }
        .streamlit-expanderContent {
            border: 1px solid #d5e8d4 !important;
            border-top: none !important;
            border-bottom-left-radius: 8px !important;
            border-bottom-right-radius: 8px !important;
            background-color: #ffffff;
        }

        /* Button styling */
        .stButton > button {
            background-color: #2c5530 !important;
            color: white !important;
            border-radius: 8px !important;
            border: none !important;
            padding: 0.5rem 1rem !important;
            font-weight: bold !important;
            transition: all 0.3s ease !important;
        }
        .stButton > button:hover {
            background-color: #1e3b21 !important;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1) !important;
        }

        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background-color: #1a331d !important;
            color: #ffffff !important;
        }
        [data-testid="stSidebar"] p, [data-testid="stSidebar"] label, [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
            color: #e8f5e9 !important;
        }

        /* Header and text colors */
        h1, h2, h3, h4, h5, h6 {
            color: #1a331d !important;
        }
        
        /* Metric styling */
        [data-testid="stMetricValue"] {
            color: #4caf50 !important;
        }
        </style>
    """, unsafe_allow_html=True)

inject_custom_styles()

MODELS = {
    "XGBoost": "XG",
    "Neural Network": "NN",
    "Linear Regression": "LR",
}

SOIL_TYPE_OPTIONS = ["Loamy", "Clay", "Sandy", "Silt"]
CROP_TYPE_OPTIONS = ["Sugarcane", "Wheat", "Rice", "Potato", "Cotton", "Maize"]
CROP_GROWTH_STAGE_OPTIONS = ["Sowing", "Vegetative", "Flowering", "Harvest"]
SEASON_OPTIONS = ["Zaid", "Kharif", "Rabi"]
IRRIGATION_TYPE_OPTIONS = ["Drip", "Rainfed", "Sprinkler", "Canal"]
WATER_SOURCE_OPTIONS = ["Rainwater", "River", "Reservoir", "Groundwater"]
MULCHING_OPTIONS = ["No", "Yes"]
REGION_OPTIONS = ["East", "South", "North", "West", "Central"]

REQUIRED_COLUMNS = [
    "Soil_Type",
    "Soil_pH",
    "Soil_Moisture",
    "Organic_Carbon",
    "Electrical_Conductivity",
    "Temperature_C",
    "Humidity",
    "Rainfall_mm",
    "Sunlight_Hours",
    "Wind_Speed_kmh",
    "Crop_Type",
    "Crop_Growth_Stage",
    "Season",
    "Irrigation_Type",
    "Water_Source",
    "Field_Area_hectare",
    "Mulching_Used",
    "Previous_Irrigation_mm",
    "Region",
]

# Real feature importance scores from model training
TOP_10_FEATURES_STATIC = {
    "Feature": [
        "Soil Moisture",
        "Crop Growth Stage: Flowering",
        "Crop Growth Stage: Harvest",
        "Crop Growth Stage: Sowing",
        "Crop Growth Stage: Vegetative",
        "Mulching Used",
        "Wind Speed (km/h)",
        "Temperature (°C)",
        "Rainfall (mm)",
        "Irrigation Type: Canal",
    ],
    "Importance": [
        0.455482,
        0.321888,
        0.312770,
        0.312283,
        0.302851,
        0.300043,
        0.258170,
        0.252867,
        0.111478,
        0.035745,
    ],
}

MODEL_COMPARISON_STATIC = pd.DataFrame(
    {
        "Model": ["XGBoost", "Random Forest", "Neural Network", "Linear Regression"],
        "Test Accuracy": [0.9843, 0.9842, 0.9806, 0.8412],
    }
)

def format_data(value):
    if isinstance(value, str):
        return value.strip()
    return value


def normalize_prediction(prediction):
    if isinstance(prediction, list):
        return prediction
    return [prediction]


def get_prediction_label(prediction):
    if isinstance(prediction, list):
        return prediction[0] if prediction else None
    return prediction


def show_prediction_alert(label):
    if label == "High":
        st.markdown("""
        <div style="background-color: #ffebee; border-left: 5px solid #f44336; padding: 20px; border-radius: 5px; animation: pulse 2s infinite;">
            <h3 style="color: #c62828; margin: 0;">🚨 High Irrigation Need</h3>
            <p style="color: #b71c1c; margin-top: 5px;">Immediate action required. Please irrigate the field as soon as possible to prevent crop stress.</p>
        </div>
        <style>
        @keyframes pulse {
            0% { box-shadow: 0 0 0 0 rgba(244, 67, 54, 0.4); }
            70% { box-shadow: 0 0 0 10px rgba(244, 67, 54, 0); }
            100% { box-shadow: 0 0 0 0 rgba(244, 67, 54, 0); }
        }
        </style>
        """, unsafe_allow_html=True)
    elif label == "Medium":
        st.markdown("""
        <div style="background-color: #fff8e1; border-left: 5px solid #ffc107; padding: 20px; border-radius: 5px;">
            <h3 style="color: #f57f17; margin: 0;">⚠️ Medium Irrigation Need</h3>
            <p style="color: #f57f17; margin-top: 5px;">Monitor soil moisture closely. Prepare for irrigation in the near future.</p>
        </div>
        """, unsafe_allow_html=True)
    elif label == "Low":
        st.markdown("""
        <div style="background-color: #e8f5e9; border-left: 5px solid #4caf50; padding: 20px; border-radius: 5px;">
            <h3 style="color: #2e7d32; margin: 0;">✅ Low Irrigation Need</h3>
            <p style="color: #2e7d32; margin-top: 5px;">Soil moisture is adequate. No immediate irrigation is required.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="background-color: #e3f2fd; border-left: 5px solid #2196f3; padding: 20px; border-radius: 5px;">
            <h3 style="color: #1565c0; margin: 0;">ℹ️ Predicted Irrigation Need: {label}</h3>
        </div>
        """, unsafe_allow_html=True)


def collect_manual_inputs():
    # Build the farmer manual input form.
    with st.expander("🪴 Soil Properties", expanded=True):
        soil_col_1, soil_col_2, soil_col_3 = st.columns(3)
        with soil_col_1:
            soil_type = st.selectbox("Soil Type", SOIL_TYPE_OPTIONS)
            soil_ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0, step=0.1)
        with soil_col_2:
            soil_moisture = st.number_input("Soil Moisture (%)", min_value=0.0, max_value=100.0, step=0.1)
            organic_carbon = st.number_input("Organic Carbon (%)", min_value=0.0, max_value=100.0, step=0.1)
        with soil_col_3:
            electrical_conductivity = st.number_input("Electrical Conductivity (dS/m)", min_value=0.0, step=0.1)
            field_area = st.number_input("Field Area (hectares)", min_value=0.0, step=0.1)

    with st.expander("🌤️ Weather Conditions", expanded=True):
        weather_col_1, weather_col_2, weather_col_3 = st.columns(3)
        with weather_col_1:
            temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=60.0, step=0.1)
            humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, step=0.1)
        with weather_col_2:
            rainfall = st.number_input("Rainfall (mm)", min_value=0.0, step=0.1)
            sunlight_hours = st.number_input("Sunlight Hours", min_value=0.0, max_value=24.0, step=0.1)
        with weather_col_3:
            wind_speed = st.number_input("Wind Speed (km/h)", min_value=0.0, step=0.1)
            previous_irrigation = st.number_input("Previous Irrigation (mm)", min_value=0.0, step=0.1)

    with st.expander("🌾 Crop Details", expanded=True):
        crop_col_1, crop_col_2, crop_col_3 = st.columns(3)
        with crop_col_1:
            crop_type = st.selectbox("Crop Type", CROP_TYPE_OPTIONS)
            crop_growth_stage = st.selectbox("Crop Growth Stage", CROP_GROWTH_STAGE_OPTIONS)
        with crop_col_2:
            season = st.selectbox("Season", SEASON_OPTIONS)
            irrigation_type = st.selectbox("Irrigation Type", IRRIGATION_TYPE_OPTIONS)
        with crop_col_3:
            water_source = st.selectbox("Water Source", WATER_SOURCE_OPTIONS)
            region = st.selectbox("Region", REGION_OPTIONS)

    with st.expander("🚜 Farm Practice", expanded=True):
        farm_col_1, = st.columns(1)
        with farm_col_1:
            mulching_used = st.selectbox("Mulching Used", MULCHING_OPTIONS)

    return {
        "Soil_Type": format_data(soil_type),
        "Soil_pH": format_data(soil_ph),
        "Soil_Moisture": format_data(soil_moisture),
        "Organic_Carbon": format_data(organic_carbon),
        "Electrical_Conductivity": format_data(electrical_conductivity),
        "Temperature_C": format_data(temperature),
        "Humidity": format_data(humidity),
        "Rainfall_mm": format_data(rainfall),
        "Sunlight_Hours": format_data(sunlight_hours),
        "Wind_Speed_kmh": format_data(wind_speed),
        "Crop_Type": format_data(crop_type),
        "Crop_Growth_Stage": format_data(crop_growth_stage),
        "Season": format_data(season),
        "Irrigation_Type": format_data(irrigation_type),
        "Water_Source": format_data(water_source),
        "Field_Area_hectare": format_data(field_area),
        "Mulching_Used": format_data(mulching_used),
        "Previous_Irrigation_mm": format_data(previous_irrigation),
        "Region": format_data(region),
    }


st.image("assets/banner.png", use_container_width=True)
st.title("🌱 Irrigation Need Prediction")
st.write("Predict irrigation need using a manual form or a CSV file upload.")

with st.sidebar:
    st.header("🌾 About")
    st.write(
        "This project helps farmers estimate irrigation needs from soil, weather, and crop conditions. "
        "Use the manual tab for one field or upload a CSV for batch predictions."
    )
    st.metric("Current Model Accuracy", "98.4%")
    model_choice = st.selectbox("Choose Model", list(MODELS.keys()))
    MODEL = MODELS[model_choice]

manual_tab, batch_tab, dashboard_tab = st.tabs([
    "Manual Prediction",
    "Batch Prediction (CSV)",
    "Dashboard",
])

with manual_tab:
    st.subheader("Manual Prediction")
    st.caption("Fill in the farm conditions below and run a single prediction.")

    manual_inputs = collect_manual_inputs()

    if st.button("Predict Irrigation Need"):

        prediction = predict_irrigation_need(
            MODEL,
            manual_inputs["Soil_Type"],
            manual_inputs["Soil_pH"],
            manual_inputs["Soil_Moisture"],
            manual_inputs["Organic_Carbon"],
            manual_inputs["Electrical_Conductivity"],
            manual_inputs["Temperature_C"],
            manual_inputs["Humidity"],
            manual_inputs["Rainfall_mm"],
            manual_inputs["Sunlight_Hours"],
            manual_inputs["Wind_Speed_kmh"],
            manual_inputs["Crop_Type"],
            manual_inputs["Crop_Growth_Stage"],
            manual_inputs["Season"],
            manual_inputs["Irrigation_Type"],
            manual_inputs["Water_Source"],
            manual_inputs["Field_Area_hectare"],
            manual_inputs["Mulching_Used"],
            manual_inputs["Previous_Irrigation_mm"],
            manual_inputs["Region"],
        )

        if isinstance(prediction, str):
            st.error(prediction)
        else:
            show_prediction_alert(get_prediction_label(normalize_prediction(prediction)))

with batch_tab:
    st.subheader("Batch Prediction (CSV)")
    st.caption("Upload a CSV file with the required columns to generate predictions for many rows.")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("Data Preview")
            st.dataframe(df, use_container_width=True)

            if st.button("Predict from CSV"):
                missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
                if missing_columns:
                    st.error(f"Missing required columns: {', '.join(missing_columns)}")
                else:

                    prediction_csv = predict_irrigation_need(
                        MODEL,
                        df["Soil_Type"],
                        df["Soil_pH"],
                        df["Soil_Moisture"],
                        df["Organic_Carbon"],
                        df["Electrical_Conductivity"],
                        df["Temperature_C"],
                        df["Humidity"],
                        df["Rainfall_mm"],
                        df["Sunlight_Hours"],
                        df["Wind_Speed_kmh"],
                        df["Crop_Type"],
                        df["Crop_Growth_Stage"],
                        df["Season"],
                        df["Irrigation_Type"],
                        df["Water_Source"],
                        df["Field_Area_hectare"],
                        df["Mulching_Used"],
                        df["Previous_Irrigation_mm"],
                        df["Region"],
                    )

                    if isinstance(prediction_csv, str):
                        st.error(prediction_csv)
                    else:
                        results_df = df.copy()
                        results_df["Irrigation_Need"] = prediction_csv

                        st.success("Predicted Irrigation Needs")
                        st.dataframe(results_df, use_container_width=True)

                        csv_data = results_df.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            label="Download Predictions CSV",
                            data=csv_data,
                            file_name="predictions.csv",
                            mime="text/csv",
                        )
        except Exception as exc:
            st.error(f"Could not read the CSV file: {exc}")
    else:
        st.info("Upload a CSV file to preview and predict irrigation need in bulk.")

with dashboard_tab:
    st.subheader("Model Generation Snapshot")
    st.write("A compact view of the training notebook results and the strongest predictors.")

    summary_col_1, summary_col_2, summary_col_3, summary_col_4 = st.columns(4)
    with summary_col_1:
        st.metric("Best Model", "XGBoost")
    with summary_col_2:
        st.metric("Top Test Accuracy", "98.43%")
    with summary_col_3:
        st.metric("Input Features", "19")
    with summary_col_4:
        st.metric("Target Classes", "3")

    st.markdown("#### Model Comparison")
    model_chart = MODEL_COMPARISON_STATIC.set_index("Model").sort_values("Test Accuracy", ascending=True)
    st.bar_chart(model_chart)

    st.caption("Test accuracy values were taken from the training notebook outputs.")

    st.divider()

    st.markdown("#### Top 10 Feature Importance")
    st.caption("Features ranked by their importance according to the correlation matrix.")

    top_10_df = pd.DataFrame(TOP_10_FEATURES_STATIC)
    chart_data = top_10_df.set_index("Feature").sort_values("Importance", ascending=True)
    st.bar_chart(chart_data)

    ranking_table = top_10_df.copy()
    ranking_table.insert(0, "Rank", range(1, len(ranking_table) + 1))
    st.dataframe(ranking_table, hide_index=True,width='content')