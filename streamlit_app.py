# streamlit run streamlit_app.py

# import streamlit as st
# import pandas as pd
# import numpy as np
# import joblib
# import tensorflow as tf
# from sklearn.preprocessing import MinMaxScaler, LabelEncoder
# from PIL import Image

# # Set page config
# st.set_page_config(
#     page_title="Traffic Congestion Predictor",
#     page_icon="🚦",
#     layout="wide"
# )

# # Load models and encoders
# @st.cache_resource
# def load_model():
#     model = tf.keras.models.load_model('traffic_cnn_model_with_location.h5')
#     scaler = joblib.load('traffic_scaler_with_location.pkl')
#     label_encoder = joblib.load('traffic_label_encoder_with_location.pkl')
#     location_encoder = joblib.load('location_encoder.pkl')
#     return model, scaler, label_encoder, location_encoder

# model, scaler, label_encoder, location_encoder = load_model()

# # Helper function to convert time to minutes
# def time_to_minutes(time_str):
#     time_part, period = time_str.split()
#     h, m, s = map(int, time_part.split(':'))
    
#     # Convert to 24-hour format
#     if period == 'PM' and h != 12:
#         h += 12
#     elif period == 'AM' and h == 12:
#         h = 0
#     return h * 60 + m

# # Prediction function
# def predict_traffic(date, time, location):
#     # Convert time to minutes
#     minutes = time_to_minutes(time)
    
#     # Calculate day of week (simplified)
#     day_num = date % 7
    
#     # Prepare temporal features
#     time_sin = np.sin(2 * np.pi * minutes/1440)
#     time_cos = np.cos(2 * np.pi * minutes/1440)
#     day_sin = np.sin(2 * np.pi * day_num/7)
#     day_cos = np.cos(2 * np.pi * day_num/7)
    
#     temporal_features = np.array([[time_sin, time_cos, day_sin, day_cos, date]])
#     temporal_features = scaler.transform(temporal_features)
#     temporal_features = temporal_features.reshape(temporal_features.shape[0], temporal_features.shape[1], 1)
    
#     # Prepare location features
#     location_features = location_encoder.transform([[location]])
    
#     # Make prediction
#     prediction = model.predict({'temporal_input': temporal_features, 'location_input': location_features})
#     predicted_class = np.argmax(prediction, axis=1)
#     traffic_situation = label_encoder.inverse_transform(predicted_class)[0]
    
#     return traffic_situation

# # UI Components
# st.title("🚦 Traffic Congestion Predictor")
# st.markdown("Predict traffic congestion levels based on date, time and location")

# # Add sidebar
# with st.sidebar:
#     st.header("About")
#     st.markdown("""
#     This app predicts traffic congestion levels using a deep learning model trained on historical traffic data.
#     - **Inputs**: Date, Time, Location
#     - **Output**: Predicted traffic situation (Low, Normal, High, Heavy)
#     """)

# # Main form
# col1, col2 = st.columns([1, 2])

# with col1:
#     # Input form
#     with st.form("prediction_form"):
#         st.subheader("Enter Prediction Details")
        
#         date = st.number_input("Day of Month (1-31)", min_value=1, max_value=31, value=15)

#         def generate_time_options():
#             times = []
#             for hour in range(24):  # 0 to 23 hours
#                 for minute in [0, 15, 30, 45]:  # 15-minute intervals
#                     # Format as 12-hour time with AM/PM
#                     if hour == 0:
#                         time_str = f"12:{minute:02d}:00 AM"
#                     elif hour < 12:
#                         time_str = f"{hour}:{minute:02d}:00 AM"
#                     elif hour == 12:
#                         time_str = f"12:{minute:02d}:00 PM"
#                     else:
#                         time_str = f"{hour-12}:{minute:02d}:00 PM"
#                     times.append(time_str)
#             return times

#         time_options = generate_time_options()
        
#         time = st.selectbox("Time", time_options)
        
#         location = st.radio("Location", ("Delhi", "Mumbai"))
        
#         submitted = st.form_submit_button("Predict Traffic")

# with col2:
#     if submitted:
#         st.subheader("Prediction Result")
        
#         with st.spinner("Predicting traffic situation..."):
#             prediction = predict_traffic(date, time, location)
            
#             # Display result with appropriate color
#             if prediction == "low":
#                 st.success(f"Predicted Traffic: **{prediction}** 🟢")
#             elif prediction == "normal":
#                 st.info(f"Predicted Traffic: **{prediction}** 🔵")
#             elif prediction == "high":
#                 st.warning(f"Predicted Traffic: **{prediction}** 🟡")
#             else:  # heavy
#                 st.error(f"Predicted Traffic: **{prediction}** 🔴")
            
#             # Add some visual indicators
#             st.progress({
#                 "low": 0.25,
#                 "normal": 0.5,
#                 "high": 0.75,
#                 "heavy": 1.0
#             }[prediction])
            
#             # Add explanation
#             st.markdown("### What this means:")
#             if prediction == "low":
#                 st.markdown("Expect smooth traffic flow with minimal delays.")
#             elif prediction == "normal":
#                 st.markdown("Typical traffic conditions with moderate flow.")
#             elif prediction == "high":
#                 st.markdown("Heavier than usual traffic, expect some delays.")
#             else:
#                 st.markdown("Severe congestion expected, consider alternate routes or times.")
    
#     # Add sample predictions when form not submitted
#     else:
#         st.subheader("Sample Predictions")
#         st.markdown("Try these examples or enter your own values:")
        
#         examples = [
#             {"date": 15, "time": "8:15:00 AM", "location": "Delhi", "result": "heavy"},
#             {"date": 20, "time": "10:30:00 AM", "location": "Mumbai", "result": "high"},
#             {"date": 5, "time": "2:00:00 PM", "location": "Delhi", "result": "normal"},
#             {"date": 30, "time": "11:30:00 PM", "location": "Mumbai", "result": "low"}
#         ]
        
#         for example in examples:
#             with st.expander(f"{example['time']} on day {example['date']} in {example['location']}"):
#                 st.markdown(f"**Expected traffic:** {example['result']}")

import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder
import joblib
from datetime import datetime

# Load the trained model and preprocessing objects
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('traffic_model.h5')
    label_encoder = joblib.load('label_encoder.pkl')
    scaler = joblib.load('scaler.pkl')
    location_encoder = joblib.load('location_encoder.pkl')
    return model, label_encoder, scaler, location_encoder

model, label_encoder, scaler, location_encoder = load_model()

# Load historical data for vehicle counts
@st.cache_data
def load_data():
    return pd.read_csv('TrafficTwoMonth.csv')

df = load_data()

# Convert time string to minutes
def time_to_minutes(time_str):
    time_part, period = time_str.split()
    h, m, s = map(int, time_part.split(':'))
    if period == 'PM' and h != 12:
        h += 12
    elif period == 'AM' and h == 12:
        h = 0
    return h * 60 + m

# Preprocess the dataframe for vehicle count predictions
def preprocess_data(df):
    df['Time_min'] = df['Time'].apply(time_to_minutes)
    
    # Map day names to numbers (Monday=0, Sunday=6)
    day_map = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 
               'Friday': 4, 'Saturday': 5, 'Sunday': 6}
    df['Day_num'] = df['Day of the week'].map(day_map)
    
    # Create cyclical time features
    df['Time_sin'] = np.sin(2 * np.pi * df['Time_min']/1440)
    df['Time_cos'] = np.cos(2 * np.pi * df['Time_min']/1440)
    df['Day_sin'] = np.sin(2 * np.pi * df['Day_num']/7)
    df['Day_cos'] = np.cos(2 * np.pi * df['Day_num']/7)
    
    return df

df = preprocess_data(df)

# Function to get vehicle counts based on similar conditions
def get_vehicle_counts(date, time_str, location):
    # Convert input time to minutes
    input_time_min = time_to_minutes(time_str)
    
    # Get day of week from date
    input_date = pd.to_datetime(date)
    input_day_num = input_date.dayofweek  # Monday=0, Sunday=6
    
    # Find similar time windows (±30 minutes)
    time_window = 30
    similar_times = df[
        (df['Location'] == location) &
        (df['Time_min'] >= input_time_min - time_window) &
        (df['Time_min'] <= input_time_min + time_window) &
        (df['Day_num'] == input_day_num)
    ]
    
    if len(similar_times) == 0:
        # If no exact matches, broaden search to same hour
        similar_times = df[
            (df['Location'] == location) &
            (df['Time_min'] // 60 == input_time_min // 60) &  # Same hour
            (df['Day_num'] == input_day_num)
        ]
    
    if len(similar_times) == 0:
        # If still no matches, use all data for that location
        similar_times = df[df['Location'] == location]
    
    # Calculate average counts
    if len(similar_times) > 0:
        return {
            'CarCount': int(similar_times['CarCount'].mean()),
            'BikeCount': int(similar_times['BikeCount'].mean()),
            'BusCount': int(similar_times['BusCount'].mean()),
            'TruckCount': int(similar_times['TruckCount'].mean()),
            'Total': int(similar_times['Total'].mean())
        }
    else:
        # Default values if no data found
        return {
            'CarCount': 0,
            'BikeCount': 0,
            'BusCount': 0,
            'TruckCount': 0,
            'Total': 0
        }

# Function to preprocess input and make predictions
def predict_traffic(date, time_str, location):
    # Convert time to minutes
    input_time_min = time_to_minutes(time_str)
    
    # Get day of week from date
    input_date = pd.to_datetime(date)
    input_day_num = input_date.dayofweek  # Monday=0, Sunday=6
    
    # Create temporal features
    time_sin = np.sin(2 * np.pi * input_time_min/1440)
    time_cos = np.cos(2 * np.pi * input_time_min/1440)
    day_sin = np.sin(2 * np.pi * input_day_num/7)
    day_cos = np.cos(2 * np.pi * input_day_num/7)
    
    # Create temporal input
    temporal_features = np.array([[time_sin, time_cos, day_sin, day_cos, input_date.day]])
    temporal_features_scaled = scaler.transform(temporal_features)
    temporal_input = temporal_features_scaled.reshape(temporal_features_scaled.shape[0], 
                                                     temporal_features_scaled.shape[1], 1)
    
    # Create location input
    location_encoded = location_encoder.transform([[location]])
    
    # Make prediction
    prediction = model.predict([temporal_input, location_encoded])
    predicted_class = label_encoder.inverse_transform([np.argmax(prediction)])[0]
    
    return predicted_class

# Streamlit UI
st.title('🚦 Traffic Situation Prediction')
st.markdown("""
Predict traffic conditions and vehicle counts for specific times and locations in Delhi and Mumbai.
""")

# Input fields
col1, col2 = st.columns(2)
with col1:
    date = st.date_input('Select Date', datetime.now())
with col2:
    time = st.time_input('Select Time', datetime.now().time())

location = st.selectbox('Select Location', ['Delhi', 'Mumbai'])

if st.button('Predict Traffic'):
    time_str = time.strftime("%I:%M:%S %p")  # Convert to 12-hour format with AM/PM
    
    with st.spinner('Making prediction...'):
        # Get vehicle counts
        counts = get_vehicle_counts(date, time_str, location)
        
        # Make traffic prediction
        prediction = predict_traffic(date, time_str, location)
        
        # Display results
        st.success('Prediction complete!')
        
        st.subheader('🚦 Traffic Situation Prediction')
        st.markdown(f"**Predicted Traffic Condition:** `{prediction}`")
        
        st.subheader('🚗 Vehicle Count Estimates')
        
        cols = st.columns(5)
        with cols[0]:
            st.metric("Cars", counts['CarCount'])
        with cols[1]:
            st.metric("Bikes", counts['BikeCount'])
        with cols[2]:
            st.metric("Buses", counts['BusCount'])
        with cols[3]:
            st.metric("Trucks", counts['TruckCount'])
        with cols[4]:
            st.metric("Total Vehicles", counts['Total'])
        
        # Show explanation
        st.markdown("""
        **Note:** Vehicle counts are estimated based on historical averages for similar:
        - Time windows (±30 minutes)
        - Days of the week
        - Locations
        """)

# Add some info about the model
st.sidebar.title("About")
st.sidebar.info("""
This app predicts traffic situations and estimates vehicle counts based on:
- Time of day
- Day of week
- Location (Delhi or Mumbai)

The model was trained on 2 months of traffic data using a hybrid CNN architecture.
""")