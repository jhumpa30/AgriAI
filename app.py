import streamlit as st
import numpy as np
from PIL import Image
import joblib
import pandas as pd
import os
import requests
import tensorflow as tf
from threading import Lock

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="AgriAI", page_icon="🌱", layout="wide")

st.title("AgriAI")
st.subheader("AI-Powered Agricultural Decision Support System")
st.divider()

# -----------------------------
# HUGGING FACE BASE URL
# -----------------------------
BASE_URL = "https://huggingface.co/Jhumpa30/agriai-models/resolve/main/"

MODEL_LOCK = Lock()   # prevents race conditions

# -----------------------------
# SAFE DOWNLOAD (IMPORTANT FIX)
# -----------------------------
def get_file(filename):
    os.makedirs("models", exist_ok=True)
    path = os.path.join("models", filename)

    if os.path.exists(path):
        return path

    url = BASE_URL + filename

    with MODEL_LOCK:   # prevents double-download crashes
        if os.path.exists(path):
            return path

        try:
            r = requests.get(url, stream=True, timeout=60)
            r.raise_for_status()

            with open(path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

        except Exception as e:
            st.error(f"Model download failed: {filename}")
            raise e

    return path


# -----------------------------
# SESSION STATE
# -----------------------------
if "predicted_yield" not in st.session_state:
    st.session_state.predicted_yield = None


# -----------------------------
# TFLITE MODEL (STABLE CACHE)
# -----------------------------
@st.cache_resource
def load_disease_model():
    path = get_file("best_crop_model.tflite")

    interpreter = tf.lite.Interpreter(model_path=path)
    interpreter.allocate_tensors()

    return interpreter


# -----------------------------
# RISK MODEL
# -----------------------------
@st.cache_resource
def load_risk_model():
    model = joblib.load(get_file("disease_risk_model.pkl"))
    cols = joblib.load(get_file("disease_risk_columns.pkl"))
    return model, cols


# -----------------------------
# YIELD MODELS
# -----------------------------
@st.cache_resource
def load_yield_models():
    return {
        "rice": (
            joblib.load(get_file("yield_prediction_model.pkl")),
            joblib.load(get_file("yield_prediction_columns.pkl"))
        ),
        "maize": (
            joblib.load(get_file("yield_prediction_model.pkl")),
            joblib.load(get_file("yield_prediction_columns.pkl"))
        ),
        "potato": (
            joblib.load(get_file("yield_prediction_model.pkl")),
            joblib.load(get_file("yield_prediction_columns.pkl"))
        ),
        "tea": (
            joblib.load(get_file("tea_yield_model_v3.pkl")),
            joblib.load(get_file("tea_yield_columns_v3.pkl"))
        ),
        "tomato": (
            joblib.load(get_file("tomato_yield_model_v2.pkl")),
            joblib.load(get_file("tomato_yield_columns_v2.pkl"))
        )
    }


# -----------------------------
# PRICE MODEL
# -----------------------------
@st.cache_resource
def load_price_model():
    return (
        joblib.load(get_file("market_price_model_v2.pkl")),
        joblib.load(get_file("market_price_scaler.pkl")),
        joblib.load(get_file("market_price_columns_v2.pkl"))
    )


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


# -----------------------------
# DISEASE PREDICTION
# -----------------------------
if uploaded_file is not None:

    image = Image.open(uploaded_file)
    st.image(image, use_container_width=True)

    interpreter = load_disease_model()

    img = image.resize((224, 224))
    img = np.array(img, dtype=np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()

    prediction = interpreter.get_tensor(output_details[0]['index'])

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

    predicted_class = class_names[np.argmax(prediction)]
    confidence = float(np.max(prediction))

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

    risk_model, risk_columns = load_risk_model()

    risk_input = pd.DataFrame([[health_score, avg_temp, humidity, rainfall]],
                              columns=risk_columns)

    disease_risk = risk_model.predict(risk_input)[0]

    st.subheader("Disease Risk")
    st.write(disease_risk)


# -----------------------------
# YIELD
# -----------------------------
if predicted_class is not None:

    yield_models = load_yield_models()

    crop_map = {
        "Rice": "rice",
        "Maize": "maize",
        "Potato": "potato",
        "Tea": "tea",
        "Tomato": "tomato"
    }

    crop_type = crop_map.get(predicted_class.split("_")[0])

    st.write("Crop:", crop_type)

    crop_year = st.number_input("Year", value=2025, key="year")
    area = st.number_input("Area", value=1.0, key="area")
    rainfall_y = st.number_input("Annual Rainfall", value=1000.0, key="rain")
    fertilizer = st.number_input("Fertilizer", value=50.0, key="fert")
    pesticide = st.number_input("Pesticide", value=5.0, key="pest")
    avg_temp_y = st.number_input("Avg Temp", value=28.0, key="temp")
    max_temp = st.number_input("Max Temp", value=35.0, key="max")
    min_temp = st.number_input("Min Temp", value=20.0, key="min")

    if st.button("Predict Yield", key="yield_btn"):

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
# PRICE
# -----------------------------
if predicted_class is not None:

    st.header("Market Price")

    price_model, price_scaler, price_columns = load_price_model()

    demand = st.number_input("Demand", value=1.0, key="demand")
    supply = st.number_input("Supply", value=1.0, key="supply")
    inflation = st.number_input("Inflation", value=5.0, key="inflation")
    transport = st.number_input("Transport Cost", value=10.0, key="transport")

    if st.button("Predict Price", key="price_btn"):

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
