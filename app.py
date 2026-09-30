import json

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model

st.set_page_config(
    page_title="CIFAR-100 Image Classification",
    page_icon="🖼️",
    layout="wide",
)


@st.cache_resource
def load_model_and_resources():
    model = load_model("cnn_image_classifier.keras", compile=False)
    with open("class_names.json", "r", encoding="utf-8") as f:
        class_names = json.load(f)
    return model, class_names


model, class_names = load_model_and_resources()


def preprocess_image(img, target_size=32):
    width, height = img.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    img = img.crop((left, top, left + side, top + side))
    img = img.resize((target_size, target_size), Image.LANCZOS)
    return img


def to_model_input(img):
    arr = np.array(img, dtype="float32") / 255.0
    return np.expand_dims(arr, axis=0)


def pretty(name):
    return str(name).replace("_", " ").capitalize()


with st.sidebar:
    st.header("Model information")
    st.write("**Model type:** Convolutional Neural Network (CNN)")
    st.write("**Framework:** TensorFlow / Keras")
    st.write("**Dataset:** CIFAR-100")
    st.write(f"**Number of classes:** {len(class_names)}")
    st.write("**Input size:** 32x32 px, RGB")

    st.divider()
    st.subheader("Settings")
    top_k = st.slider("Number of top predictions to show", 3, 10, 5)
    threshold = st.slider("Confidence threshold for warning (%)", 0, 100, 50)

    st.divider()
    with st.expander("List of available classes"):
        st.write(", ".join(pretty(c) for c in class_names))


st.title("🖼️ CIFAR-100 Image Classification")
st.markdown(
    "Upload an image and the convolutional neural network will determine "
    "which of the **100 classes** it belongs to. "
    "The result is shown on the right together with the probability distribution."
)

with st.expander("How to use the app"):
    st.markdown(
        """
1. Click **Upload** and choose an image (`jpg`, `png`).
2. The image is automatically center-cropped to a square and resized to 32x32 px.
3. Both the original and the preprocessed image are shown, so you can see what the model receives.
4. On the right you will see the top prediction, its confidence, and a chart of the top-N predictions.

> Images with a single, clear object (an animal, a vehicle, a tree, etc.) work best.
"""
    )

st.divider()

col_left, col_right = st.columns([1, 1.3], gap="large")

with col_left:
    with st.container(border=True):
        st.subheader("1. Upload an image")
        uploaded_file = st.file_uploader(
            "Choose an image", type=["jpg", "png"]
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            preprocessed_image = preprocess_image(image)

            img_col1, img_col2 = st.columns(2)
            with img_col1:
                st.image(image, caption="Original image", use_container_width=True)
                st.caption(f"Size: {image.size[0]}x{image.size[1]} px")
            with img_col2:
                preview = preprocessed_image.resize((256, 256), Image.NEAREST)
                st.image(
                    preview,
                    caption="Preprocessed image (model input)",
                    use_container_width=True,
                )
                st.caption(
                    f"Size: {preprocessed_image.size[0]}x{preprocessed_image.size[1]} px "
                    "(enlarged for display)"
                )

with col_right:
    with st.container(border=True):
        st.subheader("2. Classification result")

        if uploaded_file is None:
            st.info("Upload an image to see the result.")
        else:
            input_data = to_model_input(preprocessed_image)

            with st.spinner("The model is analyzing the image..."):
                prediction = model.predict(input_data, verbose=0)[0]

            best = int(np.argmax(prediction))
            confidence = float(prediction[best])

            st.success("Classification complete")

            m1, m2 = st.columns(2)
            m1.metric("Predicted class", pretty(class_names[best]))
            m2.metric("Confidence", f"{confidence:.2%}")
            st.progress(confidence)

            if confidence * 100 < threshold:
                st.warning(
                    "Confidence is below the selected threshold, so the result may be inaccurate."
                )

            st.divider()

            top_idx = np.argsort(prediction)[::-1][:top_k]
            top_names = [pretty(class_names[i]) for i in top_idx]
            top_probs = [float(prediction[i]) * 100 for i in top_idx]

            tab_chart, tab_table = st.tabs(["Chart", "Table"])

            with tab_chart:
                fig, ax = plt.subplots(figsize=(10, 0.5 * top_k + 1))
                colors = ["#1c83e1"] * (top_k - 1)
                bars = ax.barh(top_names, top_probs, color=colors)
                ax.invert_yaxis()
                ax.set_xlabel("Probability")
                ax.set_xlim(0, 100)
                ax.bar_label(bars, fmt="%.1f%%", padding=3)
                ax.spines[["top", "right"]].set_visible(False)
                st.pyplot(fig)

            with tab_table:
                st.dataframe(
                    {
                        "Class": top_names,
                        "Probability (%)": [round(p, 2) for p in top_probs],
                    },
                    use_container_width=True,
                    hide_index=True,
                )