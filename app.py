import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import os
import random
import plotly.tools as tls
import seaborn as sns
from sklearn.inspection import PartialDependenceDisplay, partial_dependence 
from sklearn.inspection import permutation_importance

# --- Model Loading (Cached for performance) ---
# @st.cache_resource ensures the heavy model loading happens only once, 
# even if the user interacts with the app many times.
MODEL_FILE = 'reduced_accident_model.joblib'
DATA_FILE = 'accident_data.pkl'

@st.cache_resource
def load_model():
    """Loads the serialized model from the joblib file."""
    if not os.path.exists(MODEL_FILE):
        # In a real app, this should only happen if the file is missing
        st.error(f"FATAL ERROR: The model file '{MODEL_FILE}' was not found. Please ensure it is in the same directory.")
        st.stop()
    try:
        model = joblib.load(MODEL_FILE)
        st.success("Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

# Load the model object
xgbregressor = load_model()

@st.cache_data
def load_data():
    data = joblib.load(DATA_FILE)
    return data
    
accident_data = load_data()
X = accident_data.copy()
accident_risk = X.pop('accident_risk')



LIGHTING_CATEGORIES = ['night', 'dim', 'daylight']
LIGHTING_DTYPE = pd.CategoricalDtype(categories=LIGHTING_CATEGORIES, ordered=True)
X['lighting'] = X['lighting'].astype(LIGHTING_DTYPE)

WEATHER_CATEGORIES = ['clear', 'rainy', 'foggy'] 
WEATHER_DTYPE = pd.CategoricalDtype(categories=WEATHER_CATEGORIES, ordered=True)
X['weather'] = X['weather'].astype(WEATHER_DTYPE)

st.set_page_config(
    page_title="Road Accident Predictor", 
    layout="centered"
)

st.title("Road Accident Predictor")


st.markdown("#### Factors affecting road accident risk. This plot shows the effects of 5 " 
 "factors, in order of magnitude, on predicted risk")
fig, axes = plt.subplots(
    nrows=2, 
    ncols=3, 
    figsize=(24, 12) # Sets the overall size
)
axes = axes.flatten()
fig.delaxes(axes[-1])

@st.cache_data
def generate_curvature_pdp(_model, data):
    return PartialDependenceDisplay.from_estimator(
        estimator=xgbregressor, 
        X=data, 
        features=['curvature'], 
        )

curvature_display = generate_curvature_pdp(xgbregressor, X)  
curvature_display.plot(ax=axes[0]) 
curvature_display.axes_[0, 0].set_xlabel("Road Curvature", fontsize=16)
curvature_display.axes_[0, 0].set_ylabel("Partial Dependence", fontsize=16)
curvature_display.axes_[0, 0].tick_params(axis='both', which='major', labelsize=16)

@st.cache_data
def generate_speed_limit_pdp(_model, data):
    return PartialDependenceDisplay.from_estimator(
        estimator=xgbregressor, 
        X=data, 
        features=['speed_limit'], 
        )

speed_limit_display = generate_speed_limit_pdp(xgbregressor, X)  
speed_limit_display.plot(ax=axes[1]) 
speed_limit_display.axes_[0, 0].set_xlabel("Speed Limit", fontsize=16)
speed_limit_display.axes_[0, 0].set_ylabel("Partial Dependence", fontsize=16)
speed_limit_display.axes_[0, 0].tick_params(axis='both', which='major', labelsize=16)

@st.cache_data
def generate_lighting_pdp(_model, data):
    return PartialDependenceDisplay.from_estimator(
        estimator=xgbregressor, 
        X=data, 
        features=['lighting'], 
        categorical_features = ['lighting']
        )

lighting_display = generate_lighting_pdp(xgbregressor, X)  
lighting_display.plot(ax=axes[2]) 
lighting_display.axes_[0, 0].set_xlabel("Lighting", fontsize=16)
lighting_display.axes_[0, 0].set_ylabel("Partial Dependence", fontsize=16)
lighting_display.axes_[0, 0].tick_params(axis='both', which='major', labelsize=16)

@st.cache_data
def generate_weather_pdp(_model, data):
    return PartialDependenceDisplay.from_estimator(
        estimator=xgbregressor, 
        X=data, 
        features=['weather'], 
        categorical_features = ['weather']
        )

weather_display = generate_weather_pdp(xgbregressor, X)  
weather_display.plot(ax=axes[3]) 
weather_display.axes_[0, 0].set_xlabel("Weather", fontsize=16)
weather_display.axes_[0, 0].set_ylabel("Partial Dependence", fontsize=16)
weather_display.axes_[0, 0].tick_params(axis='both', which='major', labelsize=16)

@st.cache_data
def generate_reported_accidents_pdp(_model, data):
    return PartialDependenceDisplay.from_estimator(
        estimator=xgbregressor, 
        X=data, 
        features=['num_reported_accidents'], 
        )

reported_accidents_display = generate_reported_accidents_pdp(xgbregressor, X)  
reported_accidents_display.plot(ax=axes[4]) 
reported_accidents_display.axes_[0, 0].set_xlabel("Reported Accidents", fontsize=16)
reported_accidents_display.axes_[0, 0].set_ylabel("Partial Dependence", fontsize=16)
reported_accidents_display.axes_[0, 0].tick_params(axis='both', which='major', labelsize=16)

fig.tight_layout()
st.pyplot(fig)

speed_choices = [25, 35, 45, 60, 70]

st.markdown("### Feature Inputs")
curvature = st.slider("Curvature", min_value=0.0, max_value=1.0, value=0.0, step=0.01)
speed_limit = st.radio("Speed Limit", speed_choices, horizontal=True)
lighting = st.radio("Lighting", LIGHTING_CATEGORIES, horizontal=True)
weather = st.radio("Weather", WEATHER_CATEGORIES, horizontal=True)
num_reported_accidents = st.radio("Number of Reported Accidents", range(7), horizontal=True)

# Prediction Button
if st.button("Calculate Probability", key="predict_button", type="primary"):
    # 1. Prepare input data for the model
    # The model expects input in the shape (1, num_features)
    input_values = {
        'curvature': [curvature],
        'speed_limit': [speed_limit],
        'lighting': [lighting],
        'weather': [weather],
        'num_reported_accidents': [num_reported_accidents]
    }
    X_predict = pd.DataFrame(input_values)

    X_predict['lighting'] = X_predict['lighting'].astype(LIGHTING_DTYPE)
    X_predict['weather'] = X_predict['weather'].astype(WEATHER_DTYPE)

    # 2. Make the prediction
    try:
        prediction = xgbregressor.predict(X_predict)[0]*100
        # Ensure the final output is strictly [0, 1] as requested
        # This is a safe guard, but proper training (binary:logistic) is the main method
        #probability = np.clip(prediction, 0.0, 1.0) 
        
        st.markdown("---")
        st.markdown("### Result")
        
        # Use HTML/CSS for a clean, visual display of the probability
        st.markdown(
            f"""
            <div style="background-color: #e8f5e9; padding: 20px; border-radius: 10px; border-left: 5px solid #4CAF50;">
                <p style="font-size: 18px; margin: 0;">Predicted Probability of Accident:</p>
                <h1 style="color: #4CAF50; margin: 0; font-size: 48px;">{prediction:.1f} %</h1>
            </div>
            """, 
            unsafe_allow_html=True
        )

    except Exception as e:
        st.error(f"Prediction failed. Check your input data shape/model compatibility: {e}")

st.markdown("---")
st.caption("Deployment ready. Ensure your `xgboost_model.joblib` file is in the same folder.")
