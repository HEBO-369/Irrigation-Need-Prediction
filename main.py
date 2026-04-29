import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

def format_data( X ):
    return X if isinstance(X, (list, tuple, np.ndarray, pd.Series)) else [X]

def predict_irrigation_need(MODEL,
    Soil_Type, Soil_pH, Soil_Moisture, Organic_Carbon, 
    Electrical_Conductivity, Temperature_C, Humidity, 
    Rainfall_mm, Sunlight_Hours, Wind_Speed_kmh, 
    Crop_Type, Crop_Growth_Stage, Season, 
    Irrigation_Type, Water_Source, Field_Area_hectare, 
    Mulching_Used, Previous_Irrigation_mm, Region
):
    user_input_dict = {
        'Soil_Type': format_data(Soil_Type),
        'Soil_pH': format_data(Soil_pH),
        'Soil_Moisture': format_data(Soil_Moisture),
        'Organic_Carbon': format_data(Organic_Carbon),
        'Electrical_Conductivity': format_data(Electrical_Conductivity),
        'Temperature_C': format_data(Temperature_C),
        'Humidity': format_data(Humidity),
        'Rainfall_mm': format_data(Rainfall_mm),
        'Sunlight_Hours': format_data(Sunlight_Hours),
        'Wind_Speed_kmh': format_data(Wind_Speed_kmh),
        'Crop_Type': format_data(Crop_Type),
        'Crop_Growth_Stage': format_data(Crop_Growth_Stage),
        'Season': format_data(Season),
        'Irrigation_Type': format_data(Irrigation_Type),
        'Water_Source': format_data(Water_Source),
        'Field_Area_hectare': format_data(Field_Area_hectare),
        'Mulching_Used': format_data(Mulching_Used),
        'Previous_Irrigation_mm': format_data(Previous_Irrigation_mm),
        'Region': format_data(Region)
    }
    
    user_df = pd.DataFrame(user_input_dict)
    try:
        model = joblib.load(f'{MODEL}_model.pkl')
        scaler = joblib.load('scaler.pkl')
        training_columns = joblib.load('training_columns.pkl')
    except FileNotFoundError as e:
        return "Check the model and scaler files."
    
    if 'Mulching_Used' in user_df.columns:
        user_df['Mulching_Used'] = user_df['Mulching_Used'].map({'Yes': 1, 'No': 0})
    
    cols_to_one_hot = [col for col in user_df.columns if user_df[col].dtype == 'object']
    user_df_encoded = pd.get_dummies(user_df, columns=cols_to_one_hot)
    
    user_df_aligned = user_df_encoded.reindex(columns=training_columns, fill_value=0)
    
    user_scaled = scaler.transform(user_df_aligned)
    
    prediction = model.predict(user_scaled)

    irrigation_map = {0: 'Low', 1: 'Medium', 2: 'High'}
    prediction_text = [irrigation_map.get(value, str(value)) for value in prediction]

    return prediction_text

if __name__ == "__main__":


# This Example BRO !!
    result = predict_irrigation_need("XG",
        Soil_Type='Loamy',
        Soil_pH=6.5,
        Soil_Moisture=35.5,
        Organic_Carbon=1.2,
        Electrical_Conductivity=2.5,
        Temperature_C=22.0,
        Humidity=65.0,
        Rainfall_mm=120.0,
        Sunlight_Hours=8.0,
        Wind_Speed_kmh=15.0,
        Crop_Type='Wheat',
        Crop_Growth_Stage='Vegetative',
        Season='Kharif',
        Irrigation_Type='Drip',
        Water_Source='River',
        Field_Area_hectare=2.5,
        Mulching_Used='Yes',
        Previous_Irrigation_mm=50.0,
        Region='North'
    )
    
    print(f"Irrigation Need Prediction: {result}")