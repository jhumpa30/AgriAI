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
# SAFE DOWNLOAD
# -----------------------------
BASE_URL = "https://huggingface.co/Jhumpa30/agriai-models/resolve/main/"

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
# MODEL LOADER
# -----------------------------
def load_disease_model():
    if "tflite_model" not in st.session_state:
        path = get_file("best_crop_model.tflite")
        interpreter = tf.lite.Interpreter(model_path=path)
        interpreter.allocate_tensors()
        st.session_state.tflite_model = interpreter
    return st.session_state.tflite_model


@st.cache_resource
def load_risk_model():
    model = joblib.load(get_file("disease_risk_model.pkl"))
    cols = joblib.load(get_file("disease_risk_columns.pkl"))
    return model, cols


@st.cache_resource
def load_price_model():
    model = joblib.load(get_file("market_price_model_v2.pkl"))
    scaler = joblib.load(get_file("market_price_scaler.pkl"))
    cols = joblib.load(get_file("market_price_columns_v2.pkl"))
    return model, scaler, cols


# -----------------------------
# LABELS
# -----------------------------
class_names = [
    'Maize_Blight','Maize_CommonRust','Maize_GrayLeafSpot','Maize_Healthy',
    'Potato_EarlyBlight','Potato_Healthy','Potato_LateBlight',
    'Rice_BacterialLeafBlight','Rice_BrownSpot','Rice_Healthy','Rice_LeafBlast',
    'Rice_LeafScald','Rice_SheathBlight',
    'Tea_AlgalLeafSpot','Tea_Anthracnose','Tea_BirdEyeSpot','Tea_BrownBlight',
    'Tea_GrayBlight','Tea_Healthy','Tea_RedLeafSpot','Tea_WhiteSpot',
    'Tomato_BacterialSpot','Tomato_EarlyBlight','Tomato_Healthy',
    'Tomato_LateBlight','Tomato_LeafMold','Tomato_MosaicVirus',
    'Tomato_SeptoriaLeafSpot','Tomato_SpiderMites','Tomato_TargetSpot',
    'Tomato_YellowLeafCurlVirus'
]


# -----------------------------
# INPUTS
# -----------------------------
st.header("Weather Information")

avg_temp = st.number_input("Average Temperature", value=25.0)
humidity = st.number_input("Humidity", value=70.0)
rainfall = st.number_input("Rainfall", value=100.0)

uploaded_file = st.file_uploader("Upload leaf image", type=["jpg","jpeg","png"])

predicted_class = None
disease_risk = 0


# -----------------------------
# DISEASE PREDICTION (FIXED ROOT ISSUE)
# -----------------------------
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, use_container_width=True)

    interpreter = load_disease_model()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # 🔥 ALWAYS match training preprocessing (safe universal version)
    img = image.resize((224, 224))
    img = np.array(img, dtype=np.float32)

    img = img / 255.0  # force normalization (no conditional guessing)
    img = np.expand_dims(img, axis=0)

    interpreter.set_tensor(input_details[0]["index"], img)
    interpreter.invoke()

    prediction = interpreter.get_tensor(output_details[0]["index"])
    prediction = np.array(prediction).flatten()

    # 🔥 DEBUG (keep for diagnosis)
    st.write("RAW:", prediction)
    st.write("TOP 3:", np.argsort(prediction)[-3:])
    st.write("TOP 3 VALUES:", np.sort(prediction)[-3:])
    st.write("SUM:", np.sum(prediction))

    # 🔥 FIX: ONLY apply softmax if clearly logits
    if np.sum(prediction) > 1.5:
        exp = np.exp(prediction - np.max(prediction))
        prediction = exp / np.sum(exp)

    idx = int(np.argmax(prediction))
    confidence = float(prediction[idx])

    predicted_class = class_names[idx] if idx < len(class_names) else "Unknown"

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


# -----------------------------
# RECOMMENDATIONS (UNCHANGED LOGIC)
# -----------------------------
st.header("Recommendations")

HEALTHY_CLASSES = {
    "Maize_Healthy","Rice_Healthy","Tea_Healthy","Potato_Healthy","Tomato_Healthy"
}

DISEASE_RECOMMENDATIONS = {
    "Rice_LeafBlast": ["Improve drainage", "Reduce nitrogen", "Use fungicide"],
    "Rice_BacterialLeafBlight": ["Improve sanitation", "Avoid excess nitrogen"],
    "Potato_LateBlight": ["Remove infected plants", "Avoid irrigation"],
    "Tomato_YellowLeafCurlVirus": ["Control whiteflies", "Remove plants"],
    "Tomato_MosaicVirus": ["Disinfect tools", "Remove plants"]
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

    rec.append(f"Disease risk level: {risk_val}")

    if disease in HEALTHY_CLASSES:
        rec.append("No disease detected.")
    elif disease in DISEASE_RECOMMENDATIONS:
        rec.extend(DISEASE_RECOMMENDATIONS[disease])
    else:
        rec.append("Monitor crop closely.")

    return rec


if predicted_class is not None:

    recommendations = generate_recommendations(
        predicted_class,
        health_score,
        disease_risk
    )

    for r in recommendations:
        st.write("•", r)
