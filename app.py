import os
import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

training_path = "dataset/Training"
fruits_names = os.listdir(training_path)

# Model loading
model = load_model("fruit_classifier_model.keras")

# Interface
st.title("Images classification")
uploaded_image = st.file_uploader("Upload the image")

if uploaded_image:
    image = Image.open(uploaded_image).convert('RGB')
    st.image(image, caption="Uploaded image", use_container_width=True)

    img_resized = image.resize((100, 100))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predictions
    predictions = model.predict(img_array)[0]
    top2_indices = predictions.argsort()[-2:][::-1]
    top2 = [(fruits_names[i], predictions[i] * 100) for i in top2_indices]

    # Display top 2 predictions
    st.subheader("Top 2 predictions:")
    for name, score in top2:
        st.write(f"**{name}** — {score:.2f}%")

