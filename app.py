import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.inspection import PartialDependenceDisplay

MODEL_FILE = 'reduced_accident_model.joblib'
DATA_FILE = 'accident_data.pkl'

@st.cache_resource
def load_model():
    """Loads the model from the joblib file."""
    if not os.path.exists(MODEL_FILE):
        st.error(f"ERROR: Model file '{MODEL_FILE}' not found.")
        st.stop()
    try:
        model = joblib.load(MODEL_FILE)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

# Load the model
xgbregressor = load_model()

# avoid reload when page is refreshed
@st.cache_data
def load_data():
    data = joblib.load(DATA_FILE)
    return data
    
# load data    
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

st.markdown("#### This model uses the XGBoost regressor model trained on 500k data points to " 
    "make accident risk predictions based on five factors\n\n"
    "#### The effects of these factors, in decreasing order of importance, are shown in the following plots")
    
# create grid
fig, axes = plt.subplots(
    nrows=2, 
    ncols=3, 
    figsize=(24, 12)
)
axes = axes.flatten()

# we have one extra 
fig.delaxes(axes[-1])

@st.cache_data
def generate_pdp(feature, categorical_feature):
    return PartialDependenceDisplay.from_estimator(
        estimator=xgbregressor, 
        X=X, 
        features=[feature], 
        categorical_features=[categorical_feature],
        )
        
def plot_pdp(feature, label, axis, categorical_feature=None):
    display = generate_pdp(feature, categorical_feature)
    display.plot(ax=axis)
    display.axes_[0, 0].set_xlabel(label, fontsize=16)
    display.axes_[0, 0].set_ylabel("Partial Dependence", fontsize=16)
    display.axes_[0, 0].tick_params(axis='both', which='major', labelsize=16)
    return display
    
# to prevent recalculating each time
plot_pdp('curvature', 'Road Curvature', axes[0])
plot_pdp('speed_limit', 'Speed Limit', axes[1])
plot_pdp('lighting', 'Lighting', axes[2], categorical_feature='lighting')
plot_pdp('weather', 'Weather', axes[3], categorical_feature='weather')
plot_pdp('num_reported_accidents', 'Reported Accidents', axes[4])

fig.tight_layout()
st.pyplot(fig)

st.markdown("#### Adjust the inputs to see how they affect predicted risk")
    
SPEED_CHOICES = sorted(X['speed_limit'].unique())

with st.form('prediction_form'):
    st.markdown("#### Inputs")
    curvature = st.slider("ROAD CURVATURE", min_value=0.0, max_value=1.0, value=0.0, step=0.01)
    speed_limit = st.radio("SPEED LIMIT", SPEED_CHOICES, horizontal=True)
    lighting = st.radio("LIGHTING", LIGHTING_CATEGORIES, horizontal=True)
    weather = st.radio("WEATHER", WEATHER_CATEGORIES, horizontal=True)
    num_reported_accidents = st.radio("NUMBER OF REPORTED ACCIDENTS", range(7), horizontal=True)
    
    submitted = st.form_submit_button("Calculate Probability", type="primary")

if submitted:
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

