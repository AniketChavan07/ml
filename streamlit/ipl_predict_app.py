import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# === Page Config ===
st.set_page_config(page_title="🏏 IPL Ad Price Predictor", page_icon="🏏", layout="centered")

st.title("🏏 IPL Ad Inventory Price Predictor")
st.markdown("### Predict the Advertisement Price (in Lakhs) based on match details")

# === Load model ===
try:
    model_data = joblib.load("ipl_model.pkl")
    model = model_data["model"]
    label_encoders = model_data["label_encoders"]
    features = model_data["features"]
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# === Input fields ===
st.markdown("### 📋 Enter Match Details")

match_id = st.text_input("Match ID", "1001")
date = st.date_input("Date")
home_team = st.selectbox("Home Team", [
    "Mumbai Indians", "Chennai Super Kings", "Royal Challengers Bangalore",
    "Kolkata Knight Riders", "Rajasthan Royals", "Delhi Capitals",
    "Sunrisers Hyderabad", "Punjab Kings", "Gujarat Titans", "Lucknow Super Giants"
])
away_team = st.selectbox("Away Team", [
    "Mumbai Indians", "Chennai Super Kings", "Royal Challengers Bangalore",
    "Kolkata Knight Riders", "Rajasthan Royals", "Delhi Capitals",
    "Sunrisers Hyderabad", "Punjab Kings", "Gujarat Titans", "Lucknow Super Giants"
])
tournament_phase = st.selectbox("Tournament Phase", ["League", "Qualifier", "Eliminator", "Final"])
day_of_week = st.selectbox("Day of Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
time_slot = st.selectbox("Time Slot", ["Afternoon", "Evening", "Night"])
city = st.selectbox("City", ["Mumbai", "Delhi", "Chennai", "Kolkata", "Bangalore", "Hyderabad", "Jaipur", "Ahmedabad", "Lucknow"])
broadcaster = st.selectbox("Broadcaster", ["Star Sports", "Jio Cinema", "Sony Sports", "Hotstar"])
past_viewership_avg_millions = st.number_input("Past Viewership Avg (Millions)", min_value=0.0, value=4.5)
viewership_tv_millions = st.number_input("Viewership TV (Millions)", min_value=0.0, value=3.5)
viewership_digital_millions = st.number_input("Viewership Digital (Millions)", min_value=0.0, value=2.0)
total_viewership_millions = st.number_input("Total Viewership (Millions)", min_value=0.0, value=5.5)
marquee_match = st.selectbox("Marquee Match (1=Yes, 0=No)", [0, 1])
rivalry_match = st.selectbox("Rivalry Match (1=Yes, 0=No)", [0, 1])
weather_rain = st.selectbox("Weather Rain (1=Yes, 0=No)", [0, 1])

# === Prepare input data ===
input_data = {
    "match_id": match_id,
    "date": str(date),
    "home_team": home_team,
    "away_team": away_team,
    "tournament_phase": tournament_phase,
    "day_of_week": day_of_week,
    "time_slot": time_slot,
    "city": city,
    "broadcaster": broadcaster,
    "past_viewership_avg_millions": past_viewership_avg_millions,
    "viewership_tv_millions": viewership_tv_millions,
    "viewership_digital_millions": viewership_digital_millions,
    "total_viewership_millions": total_viewership_millions,
    "marquee_match": marquee_match,
    "rivalry_match": rivalry_match,
    "weather_rain": weather_rain
}

# Convert to DataFrame
input_df = pd.DataFrame([input_data])

# === Encode categorical columns ===
for col, le in label_encoders.items():
    if col in input_df.columns:
        try:
            input_df[col] = le.transform(input_df[col].astype(str))
        except Exception:
            input_df[col] = le.transform([le.classes_[0]])[0]

# === Align input DataFrame with model features ===
try:
    available_features = [f for f in features if f in input_df.columns]
    input_df = input_df[available_features]

    # Add missing columns with zeros
    for f in features:
        if f not in input_df.columns:
            input_df[f] = 0

    input_df = input_df[features]
except Exception as e:
    st.error(f"Feature alignment error: {e}")
    st.write("Input columns:", list(input_df.columns))
    st.write("Expected features:", features)
    st.stop()

# === Predict Button ===
if st.button("🎯 Predict Advertisement Price"):
    try:
        prediction = model.predict(input_df)[0]
        st.success(f"💰 **Predicted Ad Inventory Price:** ₹ {prediction:.2f} Lakhs")

        # === Plot Graph: Past vs Predicted ===
        previous_price = past_viewership_avg_millions * 10  # Example: scaling factor
        categories = ["Previous Avg Ad Price", "Predicted Ad Price"]
        values = [previous_price, prediction]

        fig, ax = plt.subplots()
        ax.bar(categories, values)
        ax.set_ylabel("Ad Price (Lakhs)")
        ax.set_title("📊 Comparison: Previous vs Predicted Ad Inventory Price")
        st.pyplot(fig)

    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")
