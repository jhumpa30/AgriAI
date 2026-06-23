import streamlit as st
import numpy as np
from PIL import Image
import joblib
import pandas as pd
import os
import requests
import tensorflow as tf
import gc

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

# -----------------------------
# SAFE DOWNLOAD
# -----------------------------
def get_file(filename):
    os.makedirs("models", exist_ok=True)
    path = os.path.join("models", filename)

    if os.path.exists(path):
        return path

    url = BASE_URL + filename
    r = requests.get(url, stream=True, timeout=60)
    r.raise_for_status()

    with open(path, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)

    return path


# -----------------------------
# SESSION STATE
# -----------------------------
if "predicted_yield" not in st.session_state:
    st.session_state.predicted_yield = None


# -----------------------------
# DISEASE MODEL
# -----------------------------
def load_disease_model():
    if "tflite_model" not in st.session_state:
        path = get_file("best_crop_model.tflite")
        interpreter = tf.lite.Interpreter(model_path=path)
        interpreter.allocate_tensors()
        st.session_state.tflite_model = interpreter

    return st.session_state.tflite_model


# -----------------------------
# RISK MODEL
# -----------------------------
@st.cache_resource
def load_risk_model():
    model = joblib.load(get_file("disease_risk_model.pkl"))
    cols = joblib.load(get_file("disease_risk_columns.pkl"))
    return model, cols


# -----------------------------
# PRICE MODEL
# -----------------------------
@st.cache_resource
def load_price_model():
    model = joblib.load(get_file("market_price_model_v2.pkl"))
    scaler = joblib.load(get_file("market_price_scaler.pkl"))
    cols = joblib.load(get_file("market_price_columns_v2.pkl"))
    return model, scaler, cols


# -----------------------------
# RECOMMENDATIONS
# -----------------------------
HEALTHY_CLASSES = {
    "Maize_Healthy",
    "Rice_Healthy",
    "Tea_Healthy",
    "Potato_Healthy",
    "Tomato_Healthy"
}

DISEASE_RECOMMENDATIONS = {
    "Rice_LeafBlast": ["Improve drainage", "Reduce nitrogen", "Use fungicide"],
    "Rice_BacterialLeafBlight": ["Improve sanitation", "Avoid excess nitrogen"],
    "Potato_LateBlight": ["Remove infected plants", "Avoid overhead irrigation"],
    "Tomato_YellowLeafCurlVirus": ["Control whiteflies", "Remove infected plants"],
    "Tomato_MosaicVirus": ["Disinfect tools", "Remove infected plants"]
}

def generate_recommendations(disease, health_score, disease_risk):

    rec = []

    if health_score >= 90:
        rec.append("Crop health is excellent.")
    elif health_score >= 70:
        rec.append("Crop health is good.")
    elif health_score >= 50:
        rec.append("Crop health is declining.")
    else:
        rec.append("Crop health is poor.")

    try:
        risk_val = float(disease_risk)
    except:
        risk_val = 0

    if risk_val >= 70:
        rec.append("High disease risk.")
    elif risk_val >= 40:
        rec.append("Moderate disease risk.")
    else:
        rec.append("Low disease risk.")

    if disease in HEALTHY_CLASSES:
        rec.append("No disease detected.")
        rec.append("Maintain current practices.")

    elif disease in DISEASE_RECOMMENDATIONS:
        rec.extend(DISEASE_RECOMMENDATIONS[disease])

    else:
        rec.append("Monitor crop closely.")
        rec.append("Consult expert if needed.")

    return rec


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
disease_risk = 0


# -----------------------------
# DISEASE PREDICTION (FIXED ONCE & FOR ALL)
# -----------------------------
if uploaded_file is not None:

    image = Image.open(uploaded_file)
    st.image(image, use_container_width=True)

    interpreter = load_disease_model()

    img = image.resize((224, 224))
    img = np.array(img.convert("RGB"), dtype=np.float32)

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # FIX 1: correct dtype handling
    if input_details[0]['dtype'] == np.float32:
        img = img / 255.0
    img = np.expand_dims(img, axis=0).astype(input_details[0]['dtype'])

    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()

    prediction = interpreter.get_tensor(output_details[0]['index'])
    prediction = np.array(prediction).flatten()

    # 🔥 FIX 2: stable softmax correction (prevents collapse to Tomato)
    prediction = np.exp(prediction - np.max(prediction))
    prediction = prediction / np.sum(prediction)

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

    idx = int(np.argmax(prediction))
    confidence = float(np.max(prediction))

    predicted_class = class_names[idx]

    st.write("Disease:", predicted_class)
    st.write("Confidence:", confidence)

    gc.collect()


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

    gc.collect()


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

        model = joblib.load(get_file("yield_prediction_model.pkl"))
        columns = joblib.load(get_file("yield_prediction_columns.pkl"))

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

        del model, columns
        gc.collect()


# -----------------------------
# PRICE
# -----------------------------
if predicted_class is not None:

    st.header("Market Price")

    price_model, price_scaler, price_columns = load_price_model()

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

        gc.collect()


# -----------------------------
# RECOMMENDATIONS
# -----------------------------
st.header("Recommendations")

if predicted_class is not None:

    risk_val = disease_risk if "disease_risk" in locals() else 0

    recommendations = generate_recommendations(
        predicted_class,
        health_score,
        risk_val
    )

    for r in recommendations:
        st.write("•", r)
