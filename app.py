import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import tempfile
import os
import gdown
from datetime import datetime, timedelta

# ==========================================================
# PAGE CONFIG
# ==========================================================
st.set_page_config(page_title="Deepfake Video Detection", layout="wide")
st.title("🎥 Deepfake Video Detection with Grad-CAM")

# ==========================================================
# MODEL CONFIG
# ==========================================================
FILE_ID = "1AINXnr-X5IgR3M8UV9TLjZPJcWBqflBQ"
MODEL_PATH = "video_model.h5"
MODEL_URL = f"https://drive.google.com/uc?id={FILE_ID}"

CHUNK_SIZE = 8
LAST_CONV_LAYER = "time_distributed_2"

# ==========================================================
# CLEAN OLD FILES
# ==========================================================
def cleanup_old_files():
    if os.path.exists("processed_output.mp4"):
        file_time = datetime.fromtimestamp(os.path.getctime("processed_output.mp4"))
        if datetime.now() - file_time > timedelta(minutes=5):
            os.remove("processed_output.mp4")

cleanup_old_files()

# ==========================================================
# DOWNLOAD MODEL
# ==========================================================
def download_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model from Google Drive..."):
            gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

download_model()

# ==========================================================
# LOAD MODEL (Keras 3 compatible)
# ==========================================================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        MODEL_PATH,
        compile=False,
        safe_mode=False
    )

model = load_model()

# ==========================================================
# SIDEBAR SETTINGS
# ==========================================================
st.sidebar.header("⚙️ Settings")
threshold = st.sidebar.slider("Decision Threshold", 0.0, 1.0, 0.2, 0.01)
alpha = st.sidebar.slider("Heatmap Intensity", 0.0, 1.0, 0.5, 0.05)

# ==========================================================
# VIDEO LOADING
# ==========================================================
def load_video(video_path):
    cap = cv2.VideoCapture(video_path)
    original_frames = []
    processed_frames = []
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        original_frames.append(frame)
        resized = cv2.resize(frame, (224, 224)) / 255.0
        processed_frames.append(resized.astype(np.float32))

    cap.release()
    return original_frames, np.array(processed_frames), fps

# ==========================================================
# GRAD-CAM
# ==========================================================
def generate_gradcam(model, chunk):
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(LAST_CONV_LAYER).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(chunk)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 2, 3))

    conv_outputs = conv_outputs.numpy()
    pooled_grads = pooled_grads.numpy()

    heatmaps = []

    for i in range(conv_outputs.shape[1]):
        conv_map = conv_outputs[0, i]
        heatmap = np.sum(conv_map * pooled_grads[i], axis=-1)
        heatmap = np.maximum(heatmap, 0)
        heatmap /= (np.max(heatmap) + 1e-8)
        heatmaps.append(heatmap)

    return heatmaps

def overlay_heatmap(frame, heatmap):
    heatmap = cv2.resize(heatmap, (frame.shape[1], frame.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return cv2.addWeighted(frame, alpha, heatmap_color, 1 - alpha, 0)

# ==========================================================
# UI SECTION
# ==========================================================
uploaded_video = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

if uploaded_video:

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Video")
        st.video(uploaded_video)

    if st.button("Start Processing"):

        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(uploaded_video.read())
        input_path = temp_file.name

        original_frames, model_frames, fps = load_video(input_path)
        num_frames = len(model_frames)

        if num_frames == 0:
            st.error("Invalid video.")
            st.stop()

        height, width, _ = original_frames[0].shape
        output_path = "processed_output.mp4"

        out = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            (width, height)
        )

        predictions = []
        progress_bar = st.progress(0)

        for start in range(0, num_frames, CHUNK_SIZE):

            end = min(start + CHUNK_SIZE, num_frames)
            chunk = model_frames[start:end]

            if len(chunk) < CHUNK_SIZE:
                pad = np.tile(chunk[-1:], (CHUNK_SIZE - len(chunk), 1, 1, 1))
                chunk = np.concatenate((chunk, pad), axis=0)

            chunk = np.expand_dims(chunk, axis=0)

            pred = model(chunk)
            predictions.append(float(pred.numpy()[0]))

            heatmaps = generate_gradcam(model, chunk)

            for j in range(end - start):
                overlay = overlay_heatmap(original_frames[start + j], heatmaps[j])
                out.write(overlay)

            progress_bar.progress(min(end / num_frames, 1.0))

        out.release()

        avg_prob = np.mean(predictions)
        decision = "Manipulated" if avg_prob > threshold else "Real"

        with col2:
            st.subheader("Processed Video")
            with open(output_path, "rb") as f:
                video_bytes = f.read()

            st.video(video_bytes)

            st.download_button(
                "Download Processed Video",
                video_bytes,
                "processed_video.mp4",
                "video/mp4"
            )

        st.subheader("Final Results")
        st.write(f"Average Probability: {avg_prob:.4f}")
        st.write(f"Threshold: {threshold}")
        st.write(f"Final Decision: {decision}")

        os.remove(input_path)
