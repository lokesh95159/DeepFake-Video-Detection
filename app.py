import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import tempfile
import os
import time
import gdown
from datetime import datetime, timedelta

# ==============================
# Page Config
# ==============================
st.set_page_config(page_title="Video Manipulation Detection", layout="wide")
st.title("🎥 Video Manipulation Detection with Grad-CAM")

# ==============================
# Google Drive Model Setup
# ==============================
MODEL_ID = "1X7xOD0rz_abVHpuSM3Gh_xVoe0ceerie"
MODEL_PATH = "video_model.keras"
MODEL_URL = f"https://drive.google.com/uc?id={MODEL_ID}"
# Direct link for manual download (user-friendly)
MANUAL_URL = f"https://drive.google.com/file/d/{MODEL_ID}/view?usp=sharing"

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
# Download Model If Needed
# ==============================
if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model from Google Drive... (87 MB)"):
        try:
            # Download the file
            gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
            # Verify that the file was actually created and is not empty/corrupted
            if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 1e6:  # less than 1 MB
                st.error(
                    "❌ **Download failed.** The file may not be publicly accessible "
                    "or Google Drive has blocked the download due to many requests.\n\n"
                    f"Please open the following link in your browser and ensure the file "
                    f"is shared with **'Anyone with the link'**:\n\n"
                    f"🔗 [Open Model in Google Drive]({MANUAL_URL})\n\n"
                    "After confirming sharing settings, restart the app. "
                    "If the problem persists, try again later (quota may reset)."
                )
                st.stop()
        except Exception as e:
            st.error(f"❌ Download error: {e}")
            st.stop()
else:
    # Optional: check if existing file is too small (corrupted)
    if os.path.getsize(MODEL_PATH) < 1e6:
        st.warning("Existing model file seems corrupted (too small). Re-downloading...")
        os.remove(MODEL_PATH)
        st.rerun()

# ==============================
# Sidebar Controls
# ==============================
st.sidebar.header("⚙️ Settings")

threshold = st.sidebar.slider("Decision Threshold", 0.0, 1.0, 0.2, 0.01)
alpha = st.sidebar.slider("Heatmap Intensity", 0.0, 1.0, 0.5, 0.05)

CHUNK_SIZE = 8
# The name of the last convolutional layer in the model (adjust if needed)
last_conv_layer_name = "time_distributed_2"

# ==============================
# Load Model (Cached) with Error Handling
# ==============================
@st.cache_resource
def load_model():
    """Load the Keras model with compatibility fallback."""
    try:
        # First attempt: standard load with compile=False
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        return model
    except Exception as e:
        st.warning("Standard model loading failed. Attempting with custom objects...")
        # Some models need specific layers to be passed (e.g., TimeDistributed)
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
                MODEL_PATH, compile=False, custom_objects=custom_objs
            )
            return model
        except Exception as e2:
            st.error("❌ Failed to load model. Please ensure the model file is compatible with TensorFlow 2.15.")
            st.exception(e2)
            st.stop()

# Try to load the model; if it fails, the app will stop with a message
model = load_model()

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
    # Build a sub-model that outputs the conv layer and predictions
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
# Upload Section
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
