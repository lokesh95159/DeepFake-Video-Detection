import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import tempfile
import os
import time
import requests
from datetime import datetime, timedelta
import time as time_module

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
# Robust Download Function
# ==============================
def download_file_with_retry(url, destination, expected_size_mb=87, max_retries=3):
    for attempt in range(1, max_retries + 1):
        try:
            st.info(f"Download attempt {attempt}/{max_retries}...")
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

            # Size sanity check
            file_size = os.path.getsize(destination)
            min_acceptable = expected_size_mb * 0.9 * 1e6
            if file_size < min_acceptable:
                st.warning(f"File too small ({file_size/1e6:.1f} MB). Retrying...")
                os.remove(destination)
                continue

            # Quick load test (but we don't keep the model)
            try:
                _ = tf.keras.models.load_model(destination, compile=False)
                st.success("Model verified.")
                return True
            except Exception as e:
                st.warning(f"Downloaded file is not a valid model: {e}")
                os.remove(destination)
                if attempt < max_retries:
                    time_module.sleep(2 ** attempt)
                continue

        except Exception as e:
            st.warning(f"Attempt {attempt} failed: {e}")
            if os.path.exists(destination):
                os.remove(destination)
            if attempt < max_retries:
                time_module.sleep(2 ** attempt)
            continue
    return False

# ==============================
# Get Model Path
# ==============================
def get_model_path():
    # Use existing valid file
    if os.path.exists(MODEL_PATH):
        try:
            _ = tf.keras.models.load_model(MODEL_PATH, compile=False)
            return MODEL_PATH
        except:
            st.warning("Existing model file corrupted. Re‑downloading...")
            os.remove(MODEL_PATH)

    # Try custom URL first
    if custom_url:
        st.info(f"Downloading from custom URL...")
        if download_file_with_retry(custom_url, MODEL_PATH):
            return MODEL_PATH
        else:
            st.warning("Custom URL failed, falling back to default GitHub URL.")

    # Then default GitHub URL
    st.info("Downloading from GitHub...")
    if download_file_with_retry(GITHUB_MODEL_URL, MODEL_PATH):
        return MODEL_PATH

    # If all downloads fail, ask for manual upload
    st.error("❌ Automatic download failed after multiple attempts. Please upload the model file.")
    uploaded_file = st.file_uploader("Upload `video_model.keras`", type=["keras"], key="model_upload")
    if uploaded_file is not None:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".keras")
        tmp.write(uploaded_file.read())
        tmp.close()
        return tmp.name
    return None

model_path = get_model_path()
if model_path is None:
    st.stop()

# ==============================
# Ultra‑Robust Model Loading
# ==============================
@st.cache_resource
def load_model(path):
    st.write(f"TensorFlow version: {tf.__version__}")
    try:
        import keras
        st.write(f"Keras version: {keras.__version__}")
    except ImportError:
        st.write("Standalone Keras not available, using tf.keras only.")

    # ---------- Strategy 1: Standalone Keras with safe_mode=False ----------
    try:
        import keras
        st.write("Attempt 1: keras.models.load_model with safe_mode=False...")
        model = keras.models.load_model(path, safe_mode=False)
        st.success("✅ Loaded with standalone Keras (safe_mode=False)")
        return model
    except Exception as e:
        st.warning(f"Attempt 1 failed: {e}")
        with st.expander("Show full error"):
            st.exception(e)

    # ---------- Strategy 2: Standalone Keras without safe_mode ----------
    try:
        import keras
        st.write("Attempt 2: keras.models.load_model (default)...")
        model = keras.models.load_model(path)
        st.success("✅ Loaded with standalone Keras")
        return model
    except Exception as e:
        st.warning(f"Attempt 2 failed: {e}")
        with st.expander("Show full error"):
            st.exception(e)

    # ---------- Strategy 3: tf.keras with safe_mode=False (if available) ----------
    try:
        st.write("Attempt 3: tf.keras.models.load_model with safe_mode=False...")
        # safe_mode argument exists in TF 2.16+?
        model = tf.keras.models.load_model(path, compile=False, safe_mode=False)
        st.success("✅ Loaded with tf.keras (safe_mode=False)")
        return model
    except Exception as e:
        st.warning(f"Attempt 3 failed: {e}")
        with st.expander("Show full error"):
            st.exception(e)

    # ---------- Strategy 4: tf.keras with extensive custom objects ----------
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
        'Functional': tf.keras.models.Model,          # critical for the inner model
    }
    try:
        st.write("Attempt 4: tf.keras.models.load_model with custom objects...")
        model = tf.keras.models.load_model(path, compile=False, custom_objects=custom_objs)
        st.success("✅ Loaded with tf.keras + custom objects")
        return model
    except Exception as e:
        st.warning(f"Attempt 4 failed: {e}")
        with st.expander("Show full error"):
            st.exception(e)

    # ---------- Strategy 5: Monkey‑patch missing compute_output_shape ----------
    try:
        st.write("Attempt 5: Monkey‑patching Functional.compute_output_shape...")
        # Ensure the base class has the method
        if not hasattr(tf.keras.models.Model, 'compute_output_shape'):
            def compute_output_shape(self, input_shape):
                # fallback: try to call the layer on a dummy tensor to infer shape
                import numpy as np
                dummy = tf.zeros((1,) + input_shape[1:])
                out = self(dummy, training=False)
                return out.shape
            tf.keras.models.Model.compute_output_shape = compute_output_shape
            st.info("Monkey-patched compute_output_shape onto Model.")

        # Also patch the specific Functional class if needed
        try:
            from keras.src.models.functional import Functional as KerasFunctional
            if not hasattr(KerasFunctional, 'compute_output_shape'):
                KerasFunctional.compute_output_shape = compute_output_shape
        except ImportError:
            pass

        # Now try loading again with custom objects
        model = tf.keras.models.load_model(path, compile=False, custom_objects=custom_objs)
        st.success("✅ Loaded after monkey‑patching")
        return model
    except Exception as e:
        st.warning(f"Attempt 5 failed: {e}")
        with st.expander("Show full error"):
            st.exception(e)

    # ---------- Give up ----------
    st.error("❌ All loading attempts failed. Please check that the model file is compatible with TensorFlow 2.16+.")
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
