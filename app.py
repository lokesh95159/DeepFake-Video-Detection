import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import tempfile
import os
import time
import requests
from datetime import datetime, timedelta
import time as time_module  # for sleep in retries

# ==============================
# Page Config
# ==============================
st.set_page_config(page_title="Video Manipulation Detection", layout="wide")
st.title("🎥 Video Manipulation Detection with Grad-CAM")

# ==============================
# Auto Cleanup
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
GITHUB_MODEL_URL = "https://github.com/lokesh95159/deepfake-model/releases/download/v1.0/video_classification_model_20250523_001615.keras"
MODEL_PATH = "video_model.keras"

# ==============================
# Sidebar Settings
# ==============================
st.sidebar.header("⚙️ Settings")
custom_url = st.sidebar.text_input(
    "Custom Model URL (optional)",
    help="Paste a direct download link to your model file."
)
threshold = st.sidebar.slider("Decision Threshold", 0.0, 1.0, 0.2, 0.01)
alpha = st.sidebar.slider("Heatmap Intensity", 0.0, 1.0, 0.5, 0.05)

CHUNK_SIZE = 8
DEFAULT_CONV_LAYER = "time_distributed_2"

# ==============================
# Robust Download Function with Retries and Verification
# ==============================
def download_file_with_retry(url, destination, expected_size_mb=87, max_retries=3):
    """
    Download a file from a URL with retries and integrity verification.
    Returns True if successful and file is a valid Keras model.
    """
    for attempt in range(1, max_retries + 1):
        try:
            st.info(f"Download attempt {attempt}/{max_retries}...")
            # Stream download with progress
            with requests.get(url, stream=True, timeout=30) as r:
                r.raise_for_status()
                total_size = int(r.headers.get('content-length', 0))
                progress_bar = st.progress(0, text=f"Downloading model (attempt {attempt})...")
                with open(destination, 'wb') as f:
                    downloaded = 0
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size:
                            progress_bar.progress(downloaded / total_size)
                progress_bar.empty()

            # Basic size check (allow some variation)
            file_size = os.path.getsize(destination)
            min_acceptable = expected_size_mb * 0.9 * 1e6  # 90% of expected
            if file_size < min_acceptable:
                st.warning(f"Downloaded file size ({file_size/1e6:.1f} MB) is below expected ({expected_size_mb} MB). Retrying...")
                os.remove(destination)
                continue

            # Verify that it's a valid Keras model
            try:
                st.info("Verifying downloaded model...")
                _ = tf.keras.models.load_model(destination, compile=False)
                st.success("Model verified successfully.")
                return True
            except Exception as e:
                st.warning(f"Downloaded file is not a valid Keras model: {e}")
                os.remove(destination)
                # If not last attempt, wait before retrying
                if attempt < max_retries:
                    wait_time = 2 ** attempt  # exponential backoff
                    st.info(f"Retrying in {wait_time} seconds...")
                    time_module.sleep(wait_time)
                continue

        except Exception as e:
            st.warning(f"Download attempt {attempt} failed: {e}")
            if os.path.exists(destination):
                os.remove(destination)
            if attempt < max_retries:
                wait_time = 2 ** attempt
                st.info(f"Retrying in {wait_time} seconds...")
                time_module.sleep(wait_time)
            continue

    return False

# ==============================
# Get Model Path (with robust download)
# ==============================
def get_model_path():
    # If already downloaded and valid, use it
    if os.path.exists(MODEL_PATH):
        try:
            # Quick check: try loading (but we don't want to keep it loaded yet)
            _ = tf.keras.models.load_model(MODEL_PATH, compile=False)
            return MODEL_PATH
        except:
            st.warning("Existing model file is corrupted. Re-downloading...")
            os.remove(MODEL_PATH)

    # Determine which URL to use
    download_url = custom_url if custom_url else GITHUB_MODEL_URL
    source_name = "custom URL" if custom_url else "GitHub"

    st.info(f"Downloading model from {source_name}...")
    if download_file_with_retry(download_url, MODEL_PATH):
        return MODEL_PATH

    # If custom URL failed and it's not the default, try the default as fallback
    if custom_url and custom_url != GITHUB_MODEL_URL:
        st.warning("Custom URL download failed. Falling back to default GitHub URL.")
        if download_file_with_retry(GITHUB_MODEL_URL, MODEL_PATH):
            return MODEL_PATH

    # All download attempts failed – offer manual upload
    st.error(
        "❌ **Could not download model automatically after multiple attempts.**\n\n"
        "Please upload the model file manually below."
    )
    uploaded_file = st.file_uploader("Upload `video_model.keras`", type=["keras"], key="model_upload")
    if uploaded_file is not None:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".keras")
        tmp.write(uploaded_file.read())
        tmp.close()
        st.session_state["uploaded_model_path"] = tmp.name
        return tmp.name

    return None

model_path = get_model_path()
if model_path is None:
    st.stop()

# ==============================
# Load Model (Cached)
# ==============================
@st.cache_resource
def load_model(path):
    """Try multiple loading strategies to handle Keras 2/3 compatibility."""
    st.write(f"TensorFlow version: {tf.__version__}")
    try:
        import keras
        st.write(f"Keras version: {keras.__version__}")
    except ImportError:
        st.write("Standalone keras not available, using tf.keras")

    # Strategy 1: Standard tf.keras load
    try:
        st.write("Attempting to load with tf.keras.models.load_model...")
        model = tf.keras.models.load_model(path, compile=False)
        st.success("Model loaded successfully with tf.keras.")
        return model
    except Exception as e:
        st.warning(f"tf.keras load failed: {e}")
        with st.expander("Show full error"):
            st.exception(e)

    # Strategy 2: Standalone keras
    try:
        import keras
        st.write("Attempting to load with keras.models.load_model...")
        model = keras.models.load_model(path)
        st.success("Model loaded successfully with keras.")
        return model
    except ImportError:
        st.warning("Standalone keras not available, skipping.")
    except Exception as e:
        st.warning(f"keras load failed: {e}")
        with st.expander("Show full error"):
            st.exception(e)

    # Strategy 3: Custom objects (common layers)
    custom_objs = {
        'TimeDistributed': tf.keras.layers.TimeDistributed,
        'Conv2D': tf.keras.layers.Conv2D,
        'DepthwiseConv2D': tf.keras.layers.DepthwiseConv2D,
        'BatchNormalization': tf.keras.layers.BatchNormalization,
        'ReLU': tf.keras.layers.ReLU,
        'GlobalAveragePooling2D': tf.keras.layers.GlobalAveragePooling2D,
        'Reshape': tf.keras.layers.Reshape,
        'Multiply': tf.keras.layers.Multiply,
        'Add': tf.keras.layers.Add,
        'Concatenate': tf.keras.layers.Concatenate,
        'Dropout': tf.keras.layers.Dropout,
        'Flatten': tf.keras.layers.Flatten,
        'Dense': tf.keras.layers.Dense,
        'InputLayer': tf.keras.layers.InputLayer,
        'Functional': tf.keras.models.Model,
    }
    try:
        st.write("Attempting with custom objects...")
        model = tf.keras.models.load_model(path, compile=False, custom_objects=custom_objs)
        st.success("Model loaded successfully with custom objects.")
        return model
    except Exception as e:
        st.warning(f"Custom objects load failed: {e}")
        with st.expander("Show full error"):
            st.exception(e)

    st.error("All loading attempts failed. Please ensure the model file is compatible with TensorFlow 2.16+.")
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

def find_last_conv_layer(model):
    """Dynamically find the last Conv2D layer wrapped in TimeDistributed or not."""
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.TimeDistributed):
            inner_layer = layer.layer
            for inner in reversed(inner_layer.layers):
                if isinstance(inner, tf.keras.layers.Conv2D):
                    return layer.name
        elif isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    return DEFAULT_CONV_LAYER

def make_gradcam_heatmaps_for_chunk(model, chunk, conv_layer_name):
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[
            model.get_layer(conv_layer_name).output,
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

# Dynamically determine last conv layer
last_conv_layer_name = find_last_conv_layer(model)
st.sidebar.info(f"Using conv layer: {last_conv_layer_name}")

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

            heatmaps = make_gradcam_heatmaps_for_chunk(model, chunk, last_conv_layer_name)

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
