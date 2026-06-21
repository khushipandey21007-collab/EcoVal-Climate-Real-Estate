import streamlit as st
import pickle
import numpy as np

# 1. Setup the Page Vibe
st.set_page_config(page_title="EcoVal Real Estate", page_icon="🌍", layout="centered")

st.title("🌍 EcoVal: Climate-Resilient Pricing")
st.write("Predict property values in Delhi based on standard metrics, climate risk, and green energy additions.")

st.markdown("---")

# 2. Load the Saved AI Model
@st.cache_resource
def load_model():
    with open('../models/ecoval_model.pkl', 'rb') as file:
        return pickle.load(file)

model = load_model()

# 3. Create the User Input Form
col1, col2 = st.columns(2)

with col1:
    area = st.number_input("Square Footage (Area)", min_value=500, max_value=10000, value=1500)
    beds = st.number_input("Bedrooms", min_value=1, max_value=10, value=3)
    baths = st.number_input("Bathrooms", min_value=1, max_value=10, value=2)

with col2:
    st.markdown("### 🌿 ESG & Risk Factors")
    locality = st.selectbox("Select Locality", ["Standard Area", "Okhla", "Jamia Nagar", "Mayur Vihar", "Kashmere Gate", "New Friends Colony"])
    has_solar = st.checkbox("Property has Solar Panels installed")

# 4. Process Inputs exactly how we trained the model
flood_zones = ['Okhla', 'Jamia Nagar', 'Mayur Vihar', 'Kashmere Gate', 'New Friends Colony']
flood_penalty = 0.05 if locality in flood_zones else 0.0
green_bump = 0.08 if has_solar else 0.0

# 5. The Prediction Engine
st.markdown("---")
if st.button("Predict Property Value", type="primary"):
    
    # Group inputs in the exact order the model expects
    features = np.array([[area, beds, baths, flood_penalty, green_bump]])
    
    prediction = model.predict(features)[0]
    
    st.success("AI Analysis Complete!")
    st.metric(label="Estimated Property Value", value=f"₹ {prediction:,.2f}")
    
    if flood_penalty > 0:
        st.warning(f"⚠️ Value includes a 5% penalty due to high flood risk in {locality}.")
    if green_bump > 0:
        st.info("🌿 Value includes an 8% premium for green energy sustainability.")