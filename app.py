import os
import re
import requests
import numpy as np
import streamlit as st
import tensorflow as tf
import cv2
from PIL import Image

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
FILE_ID = "176Xk4FEV-cdC2V-kuaMXnrQDr3UQfcRV"
DOWNLOAD_URL = f"https://drive.google.com/uc?export=download&id={FILE_ID}"

CLASS_NAMES = ["BENIGN", "MALIGNANT", "NORMAL"]
IMG_SIZE = 224


# ----------------------------
# Google Drive download helpers
# ----------------------------
def _get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value
    return None


def _get_filename_from_response(response, default_name):
    content_disposition = response.headers.get("Content-Disposition", "")
    match = re.search(r'filename="(.+)"', content_disposition)
    if match:
        return match.group(1)
    return default_name


def download_file_from_google_drive(file_id, destination):
    session = requests.Session()

    url = "https://drive.google.com/uc?export=download"
    response = session.get(url, params={"id": file_id}, stream=True, timeout=60)

    token = _get_confirm_token(response)
    if token:
        response = session.get(
            url,
            params={"id": file_id, "confirm": token},
            stream=True,
            timeout=60
        )

    if response.status_code != 200:
        raise RuntimeError(f"Download failed with status code {response.status_code}")

    # Detect HTML error pages from Drive
    content_type = response.headers.get("Content-Type", "").lower()
    if "text/html" in content_type:
        text_snippet = response.text[:500].lower()
        if "permission" in text_snippet or "access" in text_snippet:
            raise RuntimeError("Google Drive file is not publicly accessible.")
        if "quota" in text_snippet:
            raise RuntimeError("Google Drive download quota exceeded.")
        if "virus scan" in text_snippet or "cannot preview" in text_snippet:
            raise RuntimeError("Google Drive returned a confirmation page instead of the file.")

    tmp_path = destination + ".part"
    with open(tmp_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=32768):
            if chunk:
                f.write(chunk)

    os.replace(tmp_path, destination)


def ensure_weights():
    if os.path.exists(WEIGHTS_PATH):
        return

    with st.spinner("Downloading model weights from Google Drive..."):
        try:
            download_file_from_google_drive(FILE_ID, WEIGHTS_PATH)
        except Exception as e:
            st.error(f"Model download failed: {e}")
            st.stop()


ensure_weights()

# ----------------------------
# Model builder
# ----------------------------
def build_model():
    base_model = tf.keras.applications.ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3),
        name="resnet50_base"
    )

    base_model.trainable = False

    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(3, activation="softmax")(x)

    model = tf.keras.Model(inputs=base_model.input, outputs=outputs)
    return model


@st.cache_resource
def load_trained_model():
    model = build_model()
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


def get_base_model_from_full_model(full_model):
    for layer in full_model.layers:
        if isinstance(layer, tf.keras.Model) and layer.name == "resnet50_base":
            return layer
    raise ValueError("Base model not found inside the loaded model.")


def make_gradcam_heatmap(img_array, full_model, last_conv_layer_name="conv5_block3_out"):
    base_model = get_base_model_from_full_model(full_model)
    last_conv_layer = base_model.get_layer(last_conv_layer_name)

    grad_model = tf.keras.models.Model(
        inputs=full_model.inputs,
        outputs=[last_conv_layer.output, full_model.output]
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

    max_val = np.max(heatmap)
    if max_val != 0:
        heatmap = heatmap / max_val

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

st.sidebar.write("Weights file:")
st.sidebar.code(WEIGHTS_PATH)

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
            st.write(f"{class_name}: {preds[i] * 100:.2f}%")

    st.divider()

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Grad-CAM Heatmap")
        st.image(heatmap_color, use_container_width=True)

    with c2:
        st.subheader("Overlay Result")
        st.image(overlay_img, use_container_width=True)

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
