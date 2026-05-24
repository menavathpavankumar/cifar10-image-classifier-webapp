import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="CIFAR10 Classifier",
    layout="centered"
)

# -----------------------------
# Load Model
# -----------------------------
model = load_model("cifar10_model.h5")

# -----------------------------
# Class Labels
# -----------------------------
classes = [
    "Airplane ✈️",
    "Automobile 🚗",
    "Bird 🐦",
    "Cat 🐱",
    "Deer 🦌",
    "Dog 🐶",
    "Frog 🐸",
    "Horse 🐴",
    "Ship 🚢",
    "Truck 🚚"
]

# -----------------------------
# Title
# -----------------------------
st.title("🧠 CIFAR10 Image Classifier")

st.write("""
Upload an image and the AI model will try to classify it.

⚠️ Important:
This model ONLY knows CIFAR10 classes.
It cannot recognize elephants, lions, humans, etc.
""")

# -----------------------------
# Upload File
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

# -----------------------------
# Prediction
# -----------------------------
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_column_width=True
    )

    # Resize Image
    resized_image = image.resize((32, 32))

    # Convert to Array
    img_array = np.array(resized_image)

    # Normalize
    img_array = img_array.astype('float32') / 255.0

    # Reshape
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)

    predicted_class = np.argmax(prediction)

    confidence = np.max(prediction) * 100

    # Confidence Warning
    if confidence < 60:
        st.warning(
            "⚠️ Model is unsure about this image. "
            "Try a clearer image."
        )

    # Show Prediction
    st.success(
        f"Prediction: {classes[predicted_class]}"
    )

    # Show Confidence
    st.info(
        f"Confidence: {confidence:.2f}%"
    )

    # Show Raw Scores
    st.subheader("Prediction Scores")

    for i, score in enumerate(prediction[0]):
        st.write(
            f"{classes[i]} : {score * 100:.2f}%"
        )