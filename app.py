import streamlit as st
import numpy as np
from PIL import Image
import joblib
import pandas as pd

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
# MODEL LOADERS (SAFE + LIGHT)
# -----------------------------

@st.cache_resource
def load_disease_model():
    import tensorflow as tf
    tf.keras.backend.clear_session()
    return tf.keras.models.load_model("models/best_crop_model.keras")


@st.cache_resource
def load_risk_model():
    model = joblib.load("models/disease_risk_model.pkl")
    cols = joblib.load("models/disease_risk_columns.pkl")
    return model, cols

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
# DISEASE PREDICTION
# -----------------------------
if uploaded_file is not None:

    image = Image.open(uploaded_file)
    st.image(image, use_container_width=True)

    import tensorflow as tf

    model = load_disease_model()

    img = image.resize((224, 224))
    img = np.array(img)
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img, verbose=0)

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
