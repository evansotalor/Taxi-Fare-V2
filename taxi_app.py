import streamlit as st
import joblib
import pandas as pd

# -------------------------------
# Load trained model (cached)
# -------------------------------
@st.cache_resource
def load_model():
    return joblib.load("taxi_model_v2.pkl")

model = load_model()

# -------------------------------
# CaseLearn Branding
# -------------------------------
st.markdown(
    """
    <center>
        <p float="center">
          <img src="https://i.postimg.cc/gkG9vKXG/Case-Learn.png" width="400"/>
        </p>
        <br>
        <font size=6>CaseLearn</font>
        <br><br>
        <font size=11><b>Introduction to Machine Learning</b></font>
        <br>
        <font size=6>Supervised Learning - Regression</font>
    </center>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# -------------------------------
# Sidebar for inputs
# -------------------------------
st.sidebar.header("Trip Details")

# Categorical dropdowns
time_of_day = st.sidebar.selectbox("Time of Day", ["Morning", "Afternoon", "Evening", "Night"])
day_of_week = st.sidebar.selectbox("Day of Week", ["Weekday", "Weekend"])
traffic = st.sidebar.selectbox("Traffic Conditions", ["Low", "Medium", "High"])
weather = st.sidebar.selectbox("Weather", ["Clear", "Rain", "Snow"])

# Numerical inputs with sensible ranges
trip_distance = st.sidebar.number_input("Trip Distance (km)", min_value=0.5, max_value=100.0, value=5.0, step=0.5)
passenger_count = st.sidebar.number_input("Passenger Count", min_value=1, max_value=6, value=1, step=1)
base_fare = st.sidebar.number_input("Base Fare", min_value=1.0, max_value=20.0, value=3.0, step=0.5)
per_km_rate = st.sidebar.number_input("Per Km Rate", min_value=0.5, max_value=10.0, value=2.0, step=0.1)
per_minute_rate = st.sidebar.number_input("Per Minute Rate", min_value=0.1, max_value=5.0, value=0.5, step=0.1)
trip_duration = st.sidebar.number_input("Trip Duration (minutes)", min_value=1, max_value=180, value=15, step=1)

# -------------------------------
# Prediction
# -------------------------------
if st.sidebar.button("Predict Fare"):
    # Create dataframe with input
    input_data = pd.DataFrame([{
        "Time_of_Day": time_of_day,
        "Day_of_Week": day_of_week,
        "Traffic_Conditions": traffic,
        "Weather": weather,
        "Trip_Distance_km": trip_distance,
        "Passenger_Count": passenger_count,
        "Base_Fare": base_fare,
        "Per_Km_Rate": per_km_rate,
        "Per_Minute_Rate": per_minute_rate,
        "Trip_Duration_Minutes": trip_duration
    }])

    # Predict
    prediction = model.predict(input_data)[0]

    # Display result
    st.markdown(
        f"""
        <div style='text-align: center; background-color:#fff3e6; padding:20px; border-radius:15px;'>
            <h2 style='color:#ff6600;'>Predicted Taxi Fare: ${prediction:.2f}</h2>
        </div>
        """,
        unsafe_allow_html=True
    )
