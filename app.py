import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import joblib
import pandas as pd


import os
import gdown


MODEL_DIR = "models"

@st.cache_resource
def download_models():

    required_files = [
        "best_crop_model.keras",
        "disease_risk_model.pkl",
        "disease_risk_columns.pkl",
        "yield_prediction_model.pkl",
        "yield_prediction_columns.pkl",
        "tea_yield_model_v3.pkl",
        "tea_yield_columns_v3.pkl",
        "tomato_yield_model_v2.pkl",
        "tomato_yield_columns_v2.pkl",
        "market_price_model_v2.pkl",
        "market_price_columns_v2.pkl",
        "market_price_scaler.pkl"
    ]

    os.makedirs(MODEL_DIR, exist_ok=True)

    missing = False

    for file in required_files:
        if not os.path.exists(os.path.join(MODEL_DIR, file)):
            missing = True
            break

    if missing:
        print("Downloading models...")

        gdown.download_folder(
            url="https://drive.google.com/drive/folders/1SZMYhNvGG8wkbmV0iLjOIEQQKWZYFYaQ",
            output=MODEL_DIR,
            quiet=False,
            use_cookies=False
        )

        print("Models downloaded.")

download_models()

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="AgriAI", page_icon="🌱", layout="wide")

st.title("AgriAI")
st.subheader("AI-Powered Agricultural Decision Support System")
st.divider()

# -----------------------------
# SESSION STATE
# -----------------------------
if "predicted_yield" not in st.session_state:
    st.session_state.predicted_yield = None

# -----------------------------
# RECOMMENDATION SYSTEM
# -----------------------------
HEALTHY_CLASSES = {
    "Maize_Healthy",
    "Rice_Healthy",
    "Tea_Healthy",
    "Potato_Healthy",
    "Tomato_Healthy"
}

DISEASE_RECOMMENDATIONS = {
    "Rice_LeafBlast": [
        "Improve field drainage",
        "Reduce excess nitrogen fertilizer",
        "Apply recommended fungicide"
    ],

    "Rice_BacterialLeafBlight": [
        "Improve field sanitation",
        "Avoid excess nitrogen fertilizer",
        "Use resistant varieties"
    ],

    "Potato_LateBlight": [
        "Remove infected plants immediately",
        "Avoid overhead irrigation",
        "Apply preventive fungicide"
    ],

    "Tomato_YellowLeafCurlVirus": [
        "Control whitefly population",
        "Remove infected plants",
        "Use resistant varieties"
    ],

    "Tomato_MosaicVirus": [
        "Remove infected plants",
        "Disinfect tools regularly",
        "Avoid handling wet plants"
    ]
}


def generate_recommendations(disease, health_score, disease_risk):

    recommendations = []

    if health_score >= 90:
        recommendations.append("Crop health is excellent")

    elif health_score >= 70:
        recommendations.append("Crop health is good. Continue monitoring")

    elif health_score >= 50:
        recommendations.append("Crop health is declining. Preventive action recommended")

    else:
        recommendations.append("Crop health is poor. Immediate action required")

    if disease_risk >= 70:
        recommendations.append("High disease risk detected")

    elif disease_risk >= 40:
        recommendations.append("Moderate disease risk detected")

    else:
        recommendations.append("Low disease risk detected")

    if disease in HEALTHY_CLASSES:
        recommendations.append("No disease detected")
        recommendations.append("Maintain current crop practices")

    elif disease in DISEASE_RECOMMENDATIONS:
        recommendations.extend(DISEASE_RECOMMENDATIONS[disease])

    else:
        recommendations.append("Monitor crop closely")
        recommendations.append("Follow general agricultural guidelines")

    return recommendations


# -----------------------------
# MODEL LOADING
# -----------------------------
@st.cache_resource
def load_disease_model():
    return tf.keras.models.load_model("models/best_crop_model.keras")

disease_model = load_disease_model()


@st.cache_resource
def load_risk_model():
    risk_model = joblib.load("models/disease_risk_model.pkl")
    risk_columns = joblib.load("models/disease_risk_columns.pkl")
    return risk_model, risk_columns

risk_model, risk_columns = load_risk_model()


@st.cache_resource
def load_yield_models():
    rice_model = joblib.load("models/yield_prediction_model.pkl")
    rice_columns = joblib.load("models/yield_prediction_columns.pkl")

    tea_model = joblib.load("models/tea_yield_model_v3.pkl")
    tea_columns = joblib.load("models/tea_yield_columns_v3.pkl")

    tomato_model = joblib.load("models/tomato_yield_model_v2.pkl")
    tomato_columns = joblib.load("models/tomato_yield_columns_v2.pkl")

    return {
        "rice": (rice_model, rice_columns),
        "maize": (rice_model, rice_columns),
        "potato": (rice_model, rice_columns),
        "tea": (tea_model, tea_columns),
        "tomato": (tomato_model, tomato_columns)
    }

yield_models = load_yield_models()


@st.cache_resource
def load_price_model():
    price_model = joblib.load("models/market_price_model_v2.pkl")
    price_scaler = joblib.load("models/market_price_scaler.pkl")
    price_columns = joblib.load("models/market_price_columns_v2.pkl")
    return price_model, price_scaler, price_columns

price_model, price_scaler, price_columns = load_price_model()


# -----------------------------
# CLASS LABELS
# -----------------------------
class_names = [
    'Maize_Blight','Maize_CommonRust','Maize_GrayLeafSpot','Maize_Healthy',
    'Potato_EarlyBlight','Potato_Healthy','Potato_LateBlight',
    'Rice_BacterialLeafBlight','Rice_BrownSpot','Rice_Healthy','Rice_LeafBlast',
    'Rice_LeafScald','Rice_SheathBlight',
    'Tea_AlgalLeafSpot','Tea_Anthracnose','Tea_BirdEyeSpot','Tea_BrownBlight',
    'Tea_GrayBlight','Tea_Healthy','Tea_RedLeafSpot','Tea_WhiteSpot',
    'Tomato_BacterialSpot','Tomato_EarlyBlight','Tomato_Healthy','Tomato_LateBlight',
    'Tomato_LeafMold','Tomato_MosaicVirus','Tomato_SeptoriaLeafSpot',
    'Tomato_SpiderMites','Tomato_TargetSpot','Tomato_YellowLeafCurlVirus'
]

# -----------------------------
# INPUTS
# -----------------------------
st.header("Weather Information")

avg_temp = st.number_input("Average Temperature", value=25.0)
humidity = st.number_input("Humidity", value=70.0)
rainfall = st.number_input("Rainfall", value=100.0)

st.header("Upload Image")

uploaded_file = st.file_uploader("Upload leaf image", type=["jpg", "jpeg", "png"])

predicted_class = None
confidence = 0.0
disease_risk = 0

# -----------------------------
# DISEASE PREDICTION
# -----------------------------
if uploaded_file is not None:

    image = Image.open(uploaded_file)
    st.image(image, use_container_width=True)

    img = image.resize((224, 224))
    img = np.array(img)
    img = np.expand_dims(img, axis=0)

    prediction = disease_model.predict(img, verbose=0)

    class_index = np.argmax(prediction)
    confidence = np.max(prediction)

    predicted_class = class_names[class_index]

    st.write("Disease:", predicted_class)
    st.write("Confidence:", confidence)

# -----------------------------
# HEALTH SCORE
# -----------------------------
def get_health_score(label):
    return 100 if label and "Healthy" in label else 50

health_score = get_health_score(predicted_class)

# -----------------------------
# RISK
# -----------------------------
if predicted_class is not None:

    risk_input = pd.DataFrame([[health_score, avg_temp, humidity, rainfall]],
                              columns=risk_columns)

    disease_risk = risk_model.predict(risk_input)[0]

    st.subheader("Disease Risk")
    st.write(disease_risk)

    # -----------------------------
    # RECOMMENDATIONS (NEW SECTION)
    # -----------------------------
    recs = generate_recommendations(predicted_class, health_score, disease_risk)

    st.subheader("Recommendations")

    for r in recs:
        st.write("-", r)

# -----------------------------
# YIELD
# -----------------------------
if predicted_class is not None:

    crop_map = {
        "Rice": "rice",
        "Maize": "maize",
        "Potato": "potato",
        "Tea": "tea",
        "Tomato": "tomato"
    }

    crop_type = crop_map.get(predicted_class.split("_")[0])

    st.write("Crop:", crop_type)

    crop_year = st.number_input("Year", value=2025)
    area = st.number_input("Area", value=1.0)
    rainfall_y = st.number_input("Annual Rainfall", value=1000.0)
    fertilizer = st.number_input("Fertilizer", value=50.0)
    pesticide = st.number_input("Pesticide", value=5.0)
    avg_temp_y = st.number_input("Avg Temp", value=28.0)
    max_temp = st.number_input("Max Temp", value=35.0)
    min_temp = st.number_input("Min Temp", value=20.0)

    if st.button("Predict Yield"):

        model, columns = yield_models[crop_type]

        row = {c: 0 for c in columns}

        base = {
            "Crop_Year": crop_year,
            "Area": area,
            "Annual_Rainfall": rainfall_y,
            "Fertilizer": fertilizer,
            "Pesticide": pesticide,
            "Avg_Temperature": avg_temp_y,
            "Max_Temperature": max_temp,
            "Min_Temperature": min_temp
        }

        for k, v in base.items():
            if k in row:
                row[k] = v

        crop_col = f"Crop_{crop_type.capitalize()}"
        if crop_col in row:
            row[crop_col] = 1

        X = pd.DataFrame([row])[columns]

        st.session_state.predicted_yield = model.predict(X)[0]

        st.write("Yield:", st.session_state.predicted_yield)

# -----------------------------
# MARKET PRICE
# -----------------------------
if predicted_class is not None:

    st.header("Market Price")

    demand = st.number_input("Demand", value=1.0)
    supply = st.number_input("Supply", value=1.0)
    inflation = st.number_input("Inflation", value=5.0)
    transport = st.number_input("Transport Cost", value=10.0)

    if st.button("Predict Price"):

        if st.session_state.predicted_yield is None:
            st.error("Predict yield first")
            st.stop()

        price_input = pd.DataFrame([{
            "Demand_Index": demand,
            "Supply_Index": supply,
            "Inflation_Rate": inflation,
            "Transport_Cost": transport,
            "predicted_yield": st.session_state.predicted_yield
        }])

        price_input = price_input.reindex(columns=price_columns, fill_value=0)

        price_input = price_scaler.transform(price_input)

        price = price_model.predict(price_input)[0]

        st.write("Price:", price)
