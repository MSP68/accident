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

st.set_page_config(
    page_title="Road Accident Predictor", 
    layout="centered"
)

st.title("Road Accident Predictor")


st.markdown("#### Factors affecting road accident risk. This plot shows the effects of 5 " 
 "factors, in order of magnitude, on predicted risk")
fig, axes = plt.subplots(
    nrows=1, 
    ncols=5, 
    figsize=(24, 4) # Sets the overall size
)

@st.cache_data
def generate_pdp(_model, data, feature):
    return PartialDependenceDisplay.from_estimator(
        estimator=_model, 
        X=data, 
        features=[feature], 
        )

pdp_display = generate_pdp(xgbregressor, X, 'curvature')  
pdp_display.plot(ax=axes[0]) 

barplot_features = accident_data.columns.to_list()
barplot_features.remove('curvature')
barplot_features.remove('accident_risk')

for i, feature in enumerate(barplot_features):
    sns.countplot(
        data=X, 
        x=feature, 
        ax=axes[i + 1],
    )
fig.tight_layout()
st.pyplot(fig)


col1, col2 = st.columns(2)

speed_choices = [25, 35, 45, 60, 70]
lighting_choices = ["night", "dim", "daylight"]
weather_choices = ["clear", "rainy", "foggy"]

with col1:
    st.markdown("### Feature Inputs")
    curvature = st.slider("Curvature", min_value=0.0, max_value=1.0, value=0.0, step=0.01)
    speed_limit = st.radio("Speed Limit", speed_choices)
    lighting = st.radio("Lighting", lighting_choices)
    weather = st.radio("Weather", weather_choices)
    num_reported_accidents = st.radio("Number of Reported Accidents", range(7))

    

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
    X_predict['lighting'] = X_predict['lighting'].astype('category')
    X_predict['weather'] = X_predict['weather'].astype('category')

    # 2. Make the prediction
    try:
        prediction = xgbregressor.predict(X_predict)
        # Ensure the final output is strictly [0, 1] as requested
        # This is a safe guard, but proper training (binary:logistic) is the main method
        #probability = np.clip(prediction, 0.0, 1.0) 
        
        st.markdown("---")
        st.markdown("### Result")
        
        # Use HTML/CSS for a clean, visual display of the probability
        st.markdown(
            f"""
            <div style="background-color: #e8f5e9; padding: 20px; border-radius: 10px; border-left: 5px solid #4CAF50;">
                <p style="font-size: 18px; margin: 0;">Predicted Probability (0-1):</p>
                <h1 style="color: #4CAF50; margin: 0; font-size: 48px;">{prediction}</h1>
                <p style="font-size: 14px; color: #555;">Likelihood: <b>{prediction*100}%</b></p>
            </div>
            """, 
            unsafe_allow_html=True
        )

    except Exception as e:
        st.error(f"Prediction failed. Check your input data shape/model compatibility: {e}")

st.markdown("---")
st.caption("Deployment ready. Ensure your `xgboost_model.joblib` file is in the same folder.")
