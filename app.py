import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import os
import random
import plotly.graph_objects as go
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
model = load_model()

@st.cache_data
def load_data():
    data = joblib.load(DATA_FILE)
    return data
    
accident_data = load_data()

accident_risk = accident_data.pop('accident_risk')
# --- Streamlit UI and Prediction Logic ---

st.set_page_config(
    page_title="Road Accident Predictor", 
    layout="centered"
)

st.title("Road Accident Predictor")

second_feature_options = accident_data.columns.to_list()
second_feature_options.remove('curvature')
second_feature_options = [None] + second_feature_options

st.header("Factors affecting road accident risk")

selected_second_feature = st.selectbox(
        "2nd Feature:",
        options=second_feature_options,
    )

    
@st.cache_data
def calculate_pdp_data(second_feature):
    """
    Calculates the Partial Dependence data and caches the result.
    The cache key depends on the model, the filtered data used, and the feature list.
    """
    # Streamlit displays the spinner while this function is running (or calculating the cache)
    with st.spinner(f"Running computationally intensive PDP calculation..."):
        features = ['curvature']
        if second_feature:
            features += [second_feature]
        # partial_dependence performs the slow calculation
        pdp_results = partial_dependence(
            estimator=model,
            X=accident_data,
            features=features, # e.g., ['curvature']
            kind='average',
            grid_resolution=50 # Determines the smoothness and calculation cost
        )
    
    return pdp_results
 
pdp_results = calculate_pdp_data(selected_second_feature)
if selected_second_feature:
    
    pdp_values = pdp_results.average[0]
    curvature_grid = pdp_results.grid_values[0]

    # 2. Create a DataFrame for Plotly
    pdp_df = pd.DataFrame({
            'Curvature': curvature_grid,
            'Partial Dependence (Risk)': pdp_values
        })

    # 3. Create the interactive Plotly line chart
    fig = px.line(
            pdp_df,
            x='Curvature',
            y='Partial Dependence (Risk)',
            title='PARTIAL DEPENDENCE PLOT',
        )

    # Style and update layout
    fig.update_traces(mode='lines+markers', line=dict(color='#8B0000', width=3))
    fig.update_layout(
            xaxis_title='Curvature',
            yaxis_title='Partial Dependence (Predicted Risk)',
            margin=dict(l=40, r=40, t=60, b=40),
            hovermode="x unified",
            template="plotly_white"
        )


else:
    pdp_results = calculate_pdp_data(selected_second_feature)
    pdp_values = pdp_results.average[0]
    curvature_grid = pdp_results.grid_values[0]

    # 2. Create a DataFrame for Plotly
    pdp_df = pd.DataFrame({
            'Curvature': curvature_grid,
            'Partial Dependence (Risk)': pdp_values
        })

    # 3. Create the interactive Plotly line chart
    fig = px.line(
            pdp_df,
            x='Curvature',
            y='Partial Dependence (Risk)',
            title='PARTIAL DEPENDENCE PLOT',
        )

    # Style and update layout
    fig.update_traces(mode='lines+markers', line=dict(color='#8B0000', width=3))
    fig.update_layout(
            xaxis_title='Curvature',
            yaxis_title='Partial Dependence (Predicted Risk)',
            margin=dict(l=40, r=40, t=60, b=40),
            hovermode="x unified",
            template="plotly_white"
        )
        
        # 4. Display the plot
st.plotly_chart(fig, width='stretch')




# Input features (Adjust these sliders to match your model's 
# number of features and typical input range)

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

with col2:
    st.markdown("### Model Status")
    # Display a visual check that the model is loaded
    st.info(f"Model loaded: {type(model).__name__}")
    

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
        prediction = model.predict(X_predict)
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
