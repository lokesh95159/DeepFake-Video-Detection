import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import tempfile
import os
import time
import gdown
from datetime import datetime, timedelta

# ==========================================
# Page Configuration
# ==========================================
st.set_page_config(
    page_title="Video Manipulation Detection",
    layout="wide"
)

st.title("🎥 Video Manipulation Detection with Grad-CAM")

# ==========================================
# Model Configuration
# ==========================================
MODEL_ID = "1X7xOD0rz_abVHpuSM3Gh_xVoe0ceerie"
MODEL_PATH = "video_model.keras"
MODEL_URL = f"https://drive.google.com/uc?id={MODEL_ID}"

CHUNK_SIZE = 8
LAST_CONV_LAYER_NAME = "time_distributed_2"

# ==========================================
# Cleanup Old Files (5 Minutes)
# ==========================================
def cleanup_old_files():
    output_path = "processed_output.mp4"
    if os.path.exists(output_path):
        file_time = datetime.fromtimestamp(os.path.getctime(output_path))
        if datetime.now() - file_time > timedelta(minutes=5):
            os.remove(output_path)

cleanup_old_files()

# ==========================================
# Download Model if Not Exists
# ==========================================
if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model..."):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# ==========================================
# Sidebar Controls
# ==========================================
st.sidebar.header("⚙️ Settings")

threshold = st.sidebar.slider(
    "Decision Threshold", 0.0, 1.0, 0.2, 0.01
)

alpha = st.sidebar.slider(
    "Heatmap Intensity", 0.0, 1.0, 0.5, 0.05
)

# ==========================================
# Load Model (Cached)
# ==========================================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

model = load_model()

# ==========================================
# Video Preprocessing
# ==========================================
def load_and_preprocess_video(video_path):
    cap = cv2.VideoCapture(video_path)

    original_frames = []
    model_frames = []

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        original_frames.append(frame)

        resized = cv2.resize(frame, (224, 224))
        resized = resized.astype(np.float32) / 255.0
        model_frames.append(resized)

    cap.release()

    return original_frames, np.array(model_frames), fps

# ==========================================
# Prediction
# ==========================================
def predict_video(model, video_batch):
    prediction = model(video_batch, training=False)
    return prediction.numpy()

# ==========================================
# GradCAM
# ==========================================
def make_gradcam_heatmaps_for_chunk(model, chunk):
    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[
            model.get_layer(LAST_CONV_LAYER_NAME).output,
            model.output
        ]
    )

    chunk_tensor = tf.convert_to_tensor(chunk, dtype=tf.float32)

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(chunk_tensor, training=False)

        if predictions.shape[-1] == 1:
            loss = predictions[:, 0]
        else:
            loss = tf.reduce_max(predictions, axis=-1)

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 2, 3))

    conv_outputs = conv_outputs.numpy()
    pooled_grads = pooled_grads.numpy()

    heatmaps = []

    for f in range(conv_outputs.shape[1]):
        conv_map = conv_outputs[0, f]
        pooled = pooled_grads[f]

        heatmap = np.sum(conv_map * pooled[None, None, :], axis=-1)
        heatmap = np.maximum(heatmap, 0)
        heatmap /= (np.max(heatmap) + 1e-8)

        heatmaps.append(heatmap.astype(np.float32))

    return heatmaps

# ==========================================
# Overlay Heatmap
# ==========================================
def overlay_heatmap(frame, heatmap):
    heatmap = cv2.resize(
        heatmap, (frame.shape[1], frame.shape[0])
    )

    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(
        heatmap, cv2.COLORMAP_JET
    )

    overlay = cv2.addWeighted(
        frame, alpha, heatmap_color, 1 - alpha, 0
    )

    return overlay

# ==========================================
# Upload Section
# ==========================================
uploaded_video = st.file_uploader(
    "Upload Video", type=["mp4", "avi", "mov"]
)

if uploaded_video is not None:

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📥 Original Video")
        st.video(uploaded_video)

    if st.button("🚀 Start Processing"):

        with tempfile.NamedTemporaryFile(delete=False) as temp_input:
            temp_input.write(uploaded_video.read())
            input_path = temp_input.name

        st.info("Processing started...")

        original_frames, model_frames, fps = load_and_preprocess_video(input_path)

        if len(original_frames) == 0:
            st.error("Could not read video.")
            st.stop()

        num_frames = len(model_frames)

        height, width, _ = original_frames[0].shape
        output_path = "processed_output.mp4"

        # Codec fallback
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        out = cv2.VideoWriter(
            output_path,
            fourcc,
            fps,
            (width, height)
        )

        predictions = []
        progress_bar = st.progress(0)

        start_time = time.time()
        processed_frames = 0

        for start in range(0, num_frames, CHUNK_SIZE):

            end = min(start + CHUNK_SIZE, num_frames)
            chunk = model_frames[start:end]

            if len(chunk) < CHUNK_SIZE:
                pad_len = CHUNK_SIZE - len(chunk)
                pad = np.tile(chunk[-1:], (pad_len, 1, 1, 1))
                chunk = np.concatenate((chunk, pad), axis=0)

            chunk = np.expand_dims(chunk, axis=0).astype(np.float32)

            pred = predict_video(model, chunk)
            predictions.append(float(pred[0]))

            heatmaps = make_gradcam_heatmaps_for_chunk(model, chunk)

            for j in range(end - start):
                overlay = overlay_heatmap(
                    original_frames[start + j],
                    heatmaps[j]
                )
                out.write(overlay)
                processed_frames += 1

            progress_bar.progress(min(end / num_frames, 1.0))

        out.release()

        overall_probability = float(np.mean(predictions))
        result = (
            "Manipulated"
            if overall_probability > threshold
            else "Real"
        )

        with col2:
            st.subheader("📤 Processed Video")

            if os.path.exists(output_path):
                with open(output_path, "rb") as f:
                    video_bytes = f.read()

                st.video(video_bytes)

                st.download_button(
                    label="⬇️ Download Processed Video",
                    data=video_bytes,
                    file_name="processed_video.mp4",
                    mime="video/mp4"
                )

                st.info("⚠️ File auto-deletes after 5 minutes.")
            else:
                st.error("Output video failed.")

        st.subheader("📊 Final Results")
        st.write(f"**Average Probability:** {overall_probability:.4f}")
        st.write(f"**Final Decision:** {result}")

        os.remove(input_path)
