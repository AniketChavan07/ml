import streamlit as st
import pandas as pd
import joblib
import numpy as np
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

# === Input Fields ===
st.markdown("### 📋 Enter Match Details")

home_team = st.selectbox("Home Team", [
    "Mumbai Indians", "Chennai Super Kings", "Royal Challengers Bangalore",
    "Kolkata Knight Riders", "Delhi Capitals", "Rajasthan Royals",
    "Punjab Kings", "Sunrisers Hyderabad", "Gujarat Titans", "Lucknow Super Giants"
])

away_team = st.selectbox("Away Team", [
    "Chennai Super Kings", "Mumbai Indians", "Royal Challengers Bangalore",
    "Kolkata Knight Riders", "Delhi Capitals", "Rajasthan Royals",
    "Punjab Kings", "Sunrisers Hyderabad", "Gujarat Titans", "Lucknow Super Giants"
])

tournament_phase = st.selectbox("Tournament Phase", ["League", "Qualifier", "Eliminator", "Final"])
day_of_week = st.selectbox("Day of Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
time_slot = st.selectbox("Time Slot", ["Afternoon", "Evening", "Night"])
city = st.selectbox("City", ["Mumbai", "Chennai", "Delhi", "Bangalore", "Kolkata", "Hyderabad", "Jaipur", "Ahmedabad", "Lucknow", "Pune"])
broadcaster = st.selectbox("Broadcaster", ["Star Sports", "Jio Cinema", "Sony Sports", "Hotstar"])

past_viewership_avg_millions = st.slider("Past Viewership Avg (Millions)", 0.0, 10.0, 4.5, 0.1)
viewership_tv_millions = st.slider("Viewership TV (Millions)", 0.0, 10.0, 3.5, 0.1)
viewership_digital_millions = st.slider("Viewership Digital (Millions)", 0.0, 10.0, 2.0, 0.1)
total_viewership_millions = st.slider("Total Viewership (Millions)", 0.0, 15.0, 5.5, 0.1)
marquee_match = st.selectbox("Marquee Match", ["No", "Yes"])
rivalry_match = st.selectbox("Rivalry Match", ["No", "Yes"])
weather_rain = st.selectbox("Weather Rain", ["No", "Yes"])
past_ad_inventory_price = st.number_input("Past Ad Inventory Price (Lakhs)", min_value=0.0, value=50.0)

# Convert Yes/No to 0/1
marquee_match = 1 if marquee_match == "Yes" else 0
rivalry_match = 1 if rivalry_match == "Yes" else 0
weather_rain = 1 if weather_rain == "Yes" else 0

# === Prepare Input Data ===
input_data = {
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

input_df = pd.DataFrame([input_data])

# === Encode categorical columns ===
for col, le in label_encoders.items():
    if col in input_df.columns:
        try:
            input_df[col] = le.transform(input_df[col].astype(str))
        except Exception:
            input_df[col] = le.transform([le.classes_[0]])[0]

# Ensure correct feature order
input_df = input_df[features]

# === Prediction Button ===
if st.button("🎯 Predict Advertisement Price"):
    try:
        predicted_price = model.predict(input_df)[0]

        st.markdown(f"## 💰 Predicted Ad Price: ₹ **{predicted_price:.2f} Lakhs**")

        # === Comparison Graph: Past vs Predicted ===
        st.markdown("### 📊 Comparison: Past vs Predicted Ad Inventory Price")

        fig, ax = plt.subplots(figsize=(5, 4))
        bars = ax.bar(["Past Price", "Predicted Price"], [past_ad_inventory_price, predicted_price],
                      color=["skyblue", "mediumseagreen"])
        ax.set_ylabel("Price (Lakhs)")
        ax.set_title("Ad Inventory Price Comparison")

        # Add price labels on bars
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, yval + 1, f"₹ {yval:.2f}", ha='center', fontsize=10)

        st.pyplot(fig)

        # === Feature Importance Graph ===
        st.markdown("### 🔍 Top 10 Influencing Features")
        importance = model.feature_importances_
        imp_df = pd.DataFrame({"Feature": features, "Importance": importance})
        imp_df = imp_df.sort_values(by="Importance", ascending=False).head(10)

        fig2, ax2 = plt.subplots(figsize=(8, 4))
        ax2.barh(imp_df["Feature"], imp_df["Importance"], color="royalblue")
        ax2.set_xlabel("Importance")
        ax2.set_ylabel("Feature")
        ax2.invert_yaxis()
        ax2.set_title("Top Feature Importance")
        st.pyplot(fig2)

    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")
