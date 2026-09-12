
import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
import folium
from streamlit_folium import st_folium

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------

st.set_page_config(
    page_title="Streetlight Failure Mapping",
    page_icon="💡",
    layout="wide"
)

MODEL_PATH = "models/streetlight_cnn.keras"
THRESHOLD_PATH = "models/best_threshold.txt"
MAP_DATA = "failure_map.csv"

# ------------------------------------------------------------
# LOAD MODEL
# ------------------------------------------------------------

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

with open(THRESHOLD_PATH, "r") as f:
    THRESHOLD = float(f.read())

# ------------------------------------------------------------
# LOAD MAP DATA
# ------------------------------------------------------------

@st.cache_data
def load_map_data():
    return pd.read_csv(MAP_DATA)

map_df = load_map_data()

# ------------------------------------------------------------
# PREDICTION
# ------------------------------------------------------------

def predict_image(image):

    image = image.convert("RGB")
    image = image.resize((128, 128))

    image_array = np.array(
        image,
        dtype=np.float32
    ) / 255.0

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    probability = float(
        model.predict(
            image_array,
            verbose=0
        )[0][0]
    )

    prediction = (
        "FAULTY"
        if probability >= THRESHOLD
        else "NORMAL"
    )

    return prediction, probability

# ------------------------------------------------------------
# HEADER
# ------------------------------------------------------------

st.title("💡 Streetlight Failure Detection & Mapping")
st.write(
    "AI-powered streetlight inspection using a custom CNN."
)

st.divider()

# ------------------------------------------------------------
# IMAGE PREDICTION
# ------------------------------------------------------------

st.subheader("🔍 Inspect a Streetlight")

uploaded_file = st.file_uploader(
    "Upload a streetlight image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:

    image = Image.open(uploaded_file)

    prediction, probability = predict_image(image)

    col1, col2 = st.columns(2)

    with col1:
        st.image(
            image,
            caption="Uploaded Streetlight",
            use_container_width=True
        )

    with col2:

        st.metric(
            "Fault Probability",
            f"{probability:.2%}"
        )

        st.write(
            f"Decision Threshold: `{THRESHOLD:.4f}`"
        )

        if prediction == "FAULTY":
            st.error(
                "⚠️ STREETLIGHT LIKELY FAULTY"
            )
        else:
            st.success(
                "✅ STREETLIGHT NORMAL"
            )

# ------------------------------------------------------------
# MAP
# ------------------------------------------------------------

st.divider()

st.subheader("📍 Predicted Streetlight Failures")

if len(map_df) > 0:

    center_lat = map_df["lat"].mean()
    center_lon = map_df["lon"].mean()

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=13,
        tiles="OpenStreetMap"
    )

    for _, row in map_df.iterrows():

        folium.CircleMarker(
            location=[
                row["lat"],
                row["lon"]
            ],
            radius=7,
            color="red",
            fill=True,
            fill_opacity=0.8,
            popup=(
                f"Fault Probability: "
                f"{row['fault_probability']:.2%}"
            )
        ).add_to(m)

    st_folium(
        m,
        width=1100,
        height=550
    )

else:
    st.info("No predicted failures available.")

# ------------------------------------------------------------
# SUMMARY
# ------------------------------------------------------------

st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Predicted Failures",
        len(map_df)
    )

with col2:
    st.metric(
        "Detection Threshold",
        f"{THRESHOLD:.3f}"
    )

with col3:
    st.metric(
        "Model Type",
        "Custom CNN"
    )

st.caption(
    "Streetlight Failure Mapping | Deep Learning Project"
)
