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
# CLASS MAP
# -----------------------------
class_names = [
    'Maize_Blight', 'Maize_CommonRust', 'Maize_GrayLeafSpot', 'Maize_Healthy',
    'Potato_EarlyBlight', 'Potato_Healthy', 'Potato_LateBlight',
    'Rice_BacterialLeafBlight', 'Rice_BrownSpot', 'Rice_Healthy', 'Rice_LeafBlast',
    'Rice_LeafScald', 'Rice_SheathBlight',
    'Tea_AlgalLeafSpot', 'Tea_Anthracnose', 'Tea_BirdEyeSpot', 'Tea_BrownBlight',
    'Tea_GrayBlight', 'Tea_Healthy', 'Tea_RedLeafSpot', 'Tea_WhiteSpot',
    'Tomato_BacterialSpot', 'Tomato_EarlyBlight', 'Tomato_Healthy',
    'Tomato_LateBlight', 'Tomato_LeafMold', 'Tomato_MosaicVirus',
    'Tomato_SeptoriaLeafSpot', 'Tomato_SpiderMites', 'Tomato_TargetSpot',
    'Tomato_YellowLeafCurlVirus'
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
disease_risk = 0


# -----------------------------
# DISEASE PREDICTION (FIXED PIPELINE)
# -----------------------------
if uploaded_file is not None:

    image = Image.open(uploaded_file)
    st.image(image, use_container_width=True)

    interpreter = load_disease_model()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    img = image.resize((224, 224)).convert("RGB")
    img = np.array(img)

    # IMPORTANT: match model dtype exactly
    dtype = input_details[0]['dtype']

    if dtype == np.float32:
        img = img.astype(np.float32) / 255.0
    else:
        img = img.astype(dtype)

    img = np.expand_dims(img, axis=0)

    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()

    prediction = interpreter.get_tensor(output_details[0]['index']).flatten()

    st.write("RAW OUTPUT:", prediction)
    st.write("TOP 3:", np.argsort(prediction)[-3:])

    # NO forced softmax unless needed
    if np.max(prediction) > 1.5:
        exp = np.exp(prediction - np.max(prediction))
        prediction = exp / np.sum(exp)

    idx = int(np.argmax(prediction))

    predicted_class = class_names[idx] if idx < len(class_names) else "Unknown"
    confidence = float(prediction[idx])

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
# REST OF YOUR PIPELINE (UNCHANGED)
# -----------------------------
# (Yield, Price, Recommendations stay EXACTLY as yours)
