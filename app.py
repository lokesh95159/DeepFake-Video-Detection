import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import tempfile
import os
import time
import requests
from datetime import datetime, timedelta

# ==============================
# Page Config
# ==============================
st.set_page_config(page_title="Video Manipulation Detection", layout="wide")
st.title("🎥 Video Manipulation Detection with Grad-CAM")

# ==============================
# Auto Cleanup (Delete After 5 Minutes)
# ==============================
def cleanup_old_files():
    if os.path.exists("processed_output.mp4"):
        file_time = datetime.fromtimestamp(os.path.getctime("processed_output.mp4"))
        if datetime.now() - file_time > timedelta(minutes=5):
            os.remove("processed_output.mp4")

cleanup_old_files()

# ==============================
# Model Source Configuration
# ==============================
# Your GitHub release direct download URL
GITHUB_MODEL_URL = "https://github.com/lokesh95159/deepfake-model/releases/download/v1.0/video_classification_model_20250523_001615.keras"
MODEL_PATH = "video_model.keras"

# ==============================
# Sidebar Settings
# ==============================
st.sidebar.header("⚙️ Settings")

# Allow user to override with a custom URL
custom_url = st.sidebar.text_input(
    "Custom Model URL (optional)",
    help="Paste a direct download link to your model file (e.g., from GitHub Releases)."
)

threshold = st.sidebar.slider("Decision Threshold", 0.0, 1.0, 0.2, 0.01)
alpha = st.sidebar.slider("Heatmap Intensity", 0.0, 1.0, 0.5, 0.05)

CHUNK_SIZE = 8
last_conv_layer_name = "time_distributed_2"  # Adjust if your model uses a different name

# ==============================
# Download Model Function
# ==============================
def download_file(url, destination, expected_size_mb=87):
    """Download a file from any URL with a progress bar."""
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            progress_bar = st.progress(0, text="Downloading model...")
            with open(destination, 'wb') as f:
                downloaded = 0
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size:
                        progress_bar.progress(downloaded / total_size)
            progress_bar.empty()
        # Verify size
        if os.path.getsize(destination) < expected_size_mb * 1e6:
            st.warning("Downloaded file seems too small. It may be incomplete.")
            return False
        return True
    except Exception as e:
        st.error(f"Download failed: {e}")
        return False

# ==============================
# Get Model Path (Download or Use Existing)
# ==============================
def get_model_path():
    """Return path to a valid model file."""
    # If already downloaded and valid, use it
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 1e6:
        return MODEL_PATH

    # 1. Try custom URL if provided
    if custom_url:
        st.info("Downloading model from custom URL...")
        if download_file(custom_url, MODEL_PATH):
            return MODEL_PATH
        else:
            st.error("Custom URL download failed. Falling back to default GitHub URL.")

    # 2. Try default GitHub URL
    st.info("Downloading model from GitHub...")
    if download_file(GITHUB_MODEL_URL, MODEL_PATH):
        return MODEL_PATH

    # 3. Manual upload fallback
    st.error(
        "❌ **Could not download model automatically.**\n\n"
        "Please upload the model file manually below."
    )
    uploaded_file = st.file_uploader(
        "Upload `video_model.keras`", type=["keras"], key="model_upload"
    )
    if uploaded_file is not None:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".keras")
        tmp.write(uploaded_file.read())
        tmp.close()
        st.session_state["uploaded_model_path"] = tmp.name
        st.success("Model uploaded successfully.")
        return tmp.name

    return None  # No valid model yet

model_path = get_model_path()
if model_path is None:
    st.stop()

# ==============================
# Load Model (Cached)
# ==============================
@st.cache_resource
def load_model(path):
    """Load the Keras model with compatibility fallback."""
    try:
        model = tf.keras.models.load_model(path, compile=False)
        return model
    except Exception as e:
        st.warning("Standard model loading failed. Attempting with custom objects...")
        custom_objs = {
            'TimeDistributed': tf.keras.layers.TimeDistributed,
            'Conv2D': tf.keras.layers.Conv2D,
            'MaxPooling2D': tf.keras.layers.MaxPooling2D,
            'Flatten': tf.keras.layers.Flatten,
            'LSTM': tf.keras.layers.LSTM,
            'Dense': tf.keras.layers.Dense,
            'Dropout': tf.keras.layers.Dropout,
        }
        try:
            model = tf.keras.models.load_model(
                path, compile=False, custom_objects=custom_objs
            )
            return model
        except Exception as e2:
            st.error("❌ Failed to load model. Ensure the model file is compatible with TensorFlow 2.15.")
            st.exception(e2)
            st.stop()

model = load_model(model_path)

# ==============================
# Helper Functions
# ==============================
def load_and_preprocess_video(video_path):
    cap = cv2.VideoCapture(video_path)
    original_frames = []
    model_frames = []
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        original_frames.append(frame)
        resized = cv2.resize(frame, (224, 224)) / 255.0
        model_frames.append(resized.astype(np.float32))
    cap.release()
    return original_frames, np.array(model_frames), fps

def predict_video(model, video_batch):
    return model(video_batch, training=False)

def make_gradcam_heatmaps_for_chunk(model, chunk):
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[
            model.get_layer(last_conv_layer_name).output,
            model.output
        ]
    )
    chunk_tensor = tf.convert_to_tensor(chunk, dtype=tf.float32)
    with tf.GradientTape() as tape:
        conv_outputs, preds = grad_model(chunk_tensor, training=False)
        loss = preds[:, 0] if preds.shape[-1] == 1 else tf.reduce_max(preds, axis=-1)
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 2, 3))
    conv_outputs_np = conv_outputs.numpy()
    pooled_np = pooled_grads.numpy()
    heatmaps = []
    for f in range(conv_outputs_np.shape[1]):
        conv_map = conv_outputs_np[0, f]
        pooled = pooled_np[f]
        heatmap = np.sum(conv_map * pooled[None, None, :], axis=-1)
        heatmap = np.maximum(heatmap, 0)
        heatmap /= (np.max(heatmap) + 1e-10)
        heatmaps.append(heatmap.astype(np.float32))
    return heatmaps

def overlay_heatmap(frame, heatmap):
    heatmap = cv2.resize(heatmap, (frame.shape[1], frame.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return cv2.addWeighted(frame, alpha, heatmap_color, 1 - alpha, 0)

# ==============================
# Video Upload and Processing
# ==============================
uploaded_video = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

if uploaded_video is not None:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📥 Original Video")
        st.video(uploaded_video)

    if st.button("🚀 Start Processing"):
        temp_input = tempfile.NamedTemporaryFile(delete=False)
        temp_input.write(uploaded_video.read())
        input_path = temp_input.name

        st.info("Processing started...")
        original_frames, model_frames, fps = load_and_preprocess_video(input_path)
        num_frames = len(model_frames)
        height, width, _ = original_frames[0].shape
        output_path = "processed_output.mp4"
        out = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*'avc1'),
            fps,
            (width, height)
        )

        predictions = []
        progress_bar = st.progress(0)
        frame_text = st.empty()
        fps_text = st.empty()
        start_time = time.time()
        processed_frames = 0

        for start in range(0, num_frames, CHUNK_SIZE):
            end = min(start + CHUNK_SIZE, num_frames)
            chunk = model_frames[start:end]
            if len(chunk) < CHUNK_SIZE:
                pad_len = CHUNK_SIZE - len(chunk)
                pad_frames = np.tile(chunk[-1:], (pad_len, 1, 1, 1))
                chunk = np.concatenate((chunk, pad_frames), axis=0)
            chunk = np.expand_dims(chunk, axis=0).astype(np.float32)

            pred = predict_video(model, chunk)
            predictions.append(float(np.array(pred)[0]))

            heatmaps = make_gradcam_heatmaps_for_chunk(model, chunk)

            for j in range(end - start):
                overlay = overlay_heatmap(original_frames[start + j], heatmaps[j])
                out.write(overlay)
                processed_frames += 1

            elapsed = time.time() - start_time
            current_fps = processed_frames / elapsed if elapsed > 0 else 0
            progress_bar.progress(min(end / num_frames, 1.0))
            frame_text.markdown(f"**Frames Processed:** {processed_frames}/{num_frames}")
            fps_text.markdown(f"**Processing FPS:** {current_fps:.2f}")

        out.release()
        time.sleep(1)

        overall_probability = np.mean(predictions)
        result = "Manipulated" if overall_probability > threshold else "Real"

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
                st.info("⚠️ File will auto-delete after 5 minutes.")
            else:
                st.error("Output video failed.")

        st.subheader("📊 Final Results")
        st.write(f"**Average Probability:** {overall_probability:.4f}")
        st.write(f"**Final Decision:** {result}")

        os.remove(input_path)
