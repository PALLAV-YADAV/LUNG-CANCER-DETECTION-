import os
import numpy as np
import streamlit as st
import tensorflow as tf
import cv2
from PIL import Image
import gdown

# ----------------------------
# App config
# ----------------------------
st.set_page_config(
    page_title="Lung Cancer Detection",
    page_icon="🫁",
    layout="wide"
)

st.title("🫁 Lung Cancer Detection System (IQ-OTHNCCD)")
st.write("Upload a CT image to get prediction, confidence, and Grad-CAM visualization.")

# ----------------------------
# Constants
# ----------------------------
WEIGHTS_PATH = "resnet.weights.h5"

MODEL_URL = "https://drive.google.com/file/d/176Xk4FEV-cdC2V-kuaMXnrQDr3UQfcRV/view?usp=drive_link"

CLASS_NAMES = ["BENIGN", "MALIGNANT", "NORMAL"]
IMG_SIZE = 224

# ----------------------------
# Download model safely
# ----------------------------
def download_model():
    if not os.path.exists(WEIGHTS_PATH):
        with st.spinner("Downloading model (first time only)... Please wait ⏳"):
            try:
                gdown.download(
                    MODEL_URL,
                    WEIGHTS_PATH,
                    quiet=False,
                    fuzzy=True
                )
            except Exception as e:
                st.error("❌ Model download failed. Check Google Drive link permissions.")
                st.stop()

download_model()

# ----------------------------
# Load model
# ----------------------------
@st.cache_resource
def load_trained_model():

    from tensorflow.keras.applications import ResNet50
    from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
    from tensorflow.keras.models import Model

    base_model = ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3)
    )

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    outputs = Dense(3, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=outputs)

    model.load_weights(WEIGHTS_PATH)

    return model


model = load_trained_model()

# ----------------------------
# Helper functions
# ----------------------------
def preprocess_image(pil_img):
    img = np.array(pil_img.convert("RGB"))
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_array = img_resized.astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img, img_array


def find_last_conv_layer_name(model):
    for layer in reversed(model.layers):
        try:
            if len(layer.output.shape) == 4:
                return layer.name
        except:
            pass
    raise ValueError("No convolutional layer found")


def make_gradcam_heatmap(img_array, model):
    last_conv_layer_name = find_last_conv_layer_name(model)
    last_conv_layer = model.get_layer(last_conv_layer_name)

    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[last_conv_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        loss = predictions[:, pred_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

    heatmap = heatmap.numpy() if hasattr(heatmap, "numpy") else heatmap
    heatmap = np.maximum(heatmap, 0)

    if np.max(heatmap) != 0:
        heatmap /= np.max(heatmap)

    return heatmap


def overlay_heatmap(original_img, heatmap, alpha=0.4):
    heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    original_bgr = cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(original_bgr, 1 - alpha, heatmap_color, alpha, 0)
    overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    return heatmap_color, overlay


# ----------------------------
# Sidebar
# ----------------------------
st.sidebar.header("Model Info")
st.sidebar.write("Model: ResNet50 (Fine-tuned)")
st.sidebar.write("Dataset: IQ-OTHNCCD")
st.sidebar.write("Classes:")
st.sidebar.write("- BENIGN")
st.sidebar.write("- MALIGNANT")
st.sidebar.write("- NORMAL")

# ----------------------------
# Upload section
# ----------------------------
uploaded_file = st.file_uploader(
    "Upload a CT Scan Image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:

    pil_img = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Uploaded Image")
        st.image(pil_img, use_container_width=True)

    original_img, img_array = preprocess_image(pil_img)

    # Prediction
    preds = model.predict(img_array, verbose=0)[0]
    pred_idx = int(np.argmax(preds))
    pred_label = CLASS_NAMES[pred_idx]
    confidence = float(preds[pred_idx]) * 100

    # Grad-CAM
    heatmap = make_gradcam_heatmap(img_array, model)
    heatmap_color, overlay_img = overlay_heatmap(original_img, heatmap)

    with col2:
        st.subheader("Prediction Result")
        st.success(f"{pred_label}")
        st.write(f"Confidence: **{confidence:.2f}%**")

        st.write("### Class Probabilities")
        for i, class_name in enumerate(CLASS_NAMES):
            st.write(f"{class_name}: {preds[i]*100:.2f}%")

    st.divider()

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Grad-CAM Heatmap")
        st.image(heatmap_color, use_container_width=True)

    with c2:
        st.subheader("Overlay Result")
        st.image(overlay_img, use_container_width=True)

    # Download
    save_path = "gradcam_output.png"
    cv2.imwrite(save_path, cv2.cvtColor(overlay_img, cv2.COLOR_RGB2BGR))

    with open(save_path, "rb") as f:
        st.download_button(
            label="Download Grad-CAM Result",
            data=f,
            file_name="gradcam.png",
            mime="image/png"
        )

else:
    st.info("Upload an image to start prediction.")
