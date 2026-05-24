import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="CIFAR10 Image Classifier",
    layout="centered"
)

# -------------------------------
# Load Model
# -------------------------------
model = load_model("cifar10_model.h5")

# -------------------------------
# CIFAR10 Class Names
# -------------------------------
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

# -------------------------------
# Title
# -------------------------------
st.title("🧠 CIFAR10 Image Classifier")
st.write("Upload an image and the AI model will predict its class.")

# -------------------------------
# File Upload
# -------------------------------
uploaded_file = st.file_uploader(
    "Choose an image",
    type=["png", "jpg", "jpeg"]
)

# -------------------------------
# Prediction Logic
# -------------------------------
if uploaded_file is not None:

    # Open Image
    image = Image.open(uploaded_file).convert("RGB")

    # Show Uploaded Image
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Resize Image
    resized_image = image.resize((32, 32))

    # Convert to Array
    img_array = np.array(resized_image)

    # Normalize
    img_array = img_array / 255.0

    # Reshape for Model
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)

    # Predicted Class
    predicted_class = np.argmax(prediction)

    # Confidence
    confidence = np.max(prediction) * 100

    # Display Result
    st.success(f"Prediction: {classes[predicted_class]}")

    st.info(f"Confidence: {confidence:.2f}%")