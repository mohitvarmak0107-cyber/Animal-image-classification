"""
Animal Image Classification - Streamlit Web Application
BTech Deep Learning Mini Project - Stage 4: Live Web Application

Interactive user interface providing:
1. Fast image upload (JPG, JPEG, PNG).
2. Live image inspection and preprocessing (128x128, RGB, normalized [0, 1]).
3. Cached model inference using the trained CNN from Stage 3.
4. Real-time class prediction, confidence percentage, and breakdown across all 5 classes.
5. Clean error handling and setup instructions when model or datasets are missing.
"""

import sys
from pathlib import Path
from PIL import Image

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from src.config import (
    APP_TITLE,
    ANIMAL_CLASSES,
    IMAGE_SIZE,
    MODEL_SAVE_PATH,
    CLASS_INDICES_PATH,
    RAW_DATA_DIR,
    TRAINING_HISTORY_PATH,
    CONFUSION_MATRIX_PATH,
    CLASS_DISTRIBUTION_PATH,
    TRAINING_METRICS_PATH,
)
from src.utils import check_system_environment, inspect_raw_dataset
from src.inference import load_model_and_labels, preprocess_image, predict_animal

# 1. Page Configuration
st.set_page_config(
    page_title="Animal Image Classification",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Animal Emojis & Display Mapping
CLASS_EMOJIS = {
    "cats": "🐱",
    "dogs": "🐶",
    "elephants": "🐘",
    "horses": "🐴",
    "lions": "🦁",
}


# 2. Cached Model Loading (Loads model once to prevent retraining on reload)
@st.cache_resource(show_spinner="Loading trained CNN model...")
def get_cached_model():
    """
    Loads and caches the trained Keras CNN model.
    Cached via Streamlit so it is not re-loaded on every user interaction.
    """
    return load_model_and_labels(MODEL_SAVE_PATH, CLASS_INDICES_PATH)


model, class_labels, model_error = get_cached_model()
if class_labels is None:
    class_labels = ANIMAL_CLASSES


# 3. Sidebar Setup
with st.sidebar:
    st.title("🐾 Project Dashboard")
    st.info("**Mini Project:** Animal Image Classification\n\n**Stage:** 4 (Live Web Application)")

    st.markdown("---")
    st.subheader("🤖 Model Status")
    if model is not None:
        st.success("🟢 **CNN Model Loaded**\nReady for live inference.")
        st.caption(f"Input: `{IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}x3` | Classes: `5`")
    else:
        st.warning("🟠 **Model Not Trained Yet**")
        st.caption("Awaiting training in Stage 3.")

    st.markdown("---")
    st.subheader("🎯 Recognized Classes (5 Categories)")
    for cls in class_labels:
        emoji = CLASS_EMOJIS.get(cls, "🐾")
        st.write(f"{emoji} **{cls.capitalize()}**")

    st.markdown("---")
    st.subheader("📂 Dataset Inspection")
    if st.button("Scan Raw Dataset Folders"):
        counts = inspect_raw_dataset(RAW_DATA_DIR)
        total_images = sum(counts.values())
        st.metric(label="Total Images Detected", value=total_images)
        for cls, count in counts.items():
            emoji = CLASS_EMOJIS.get(cls, "🐾")
            st.write(f"{emoji} `{cls}`: {count} images")

    st.markdown("---")
    st.subheader("🖥️ Environment")
    env_info = check_system_environment()
    st.caption(f"Python: `{env_info['python_version']}` | TensorFlow: `{env_info['tensorflow_version']}`")


# 4. Main Page Header
st.title(APP_TITLE)
st.markdown(
    """
    Welcome to the **Animal Image Classification** application! Upload an image of an animal 
    belonging to one of the **5 target classes** to receive real-time classification results 
    from our Convolutional Neural Network (CNN).
    """
)

# 5. Tabs for Organization
tab_classify, tab_analytics, tab_viva = st.tabs([
    "📸 Live Image Classifier",
    "📊 Training & Evaluation Analytics",
    "🎓 Viva & DL Concepts Guide",
])

# ---------------------------------------------------------------------
# TAB 1: Live Image Classifier
# ---------------------------------------------------------------------
with tab_classify:
    st.subheader("Upload an Animal Photo for Classification")
    st.write("Supported formats: **JPG**, **JPEG**, **PNG**. For optimal accuracy, use clear, well-lit photos.")

    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=["jpg", "jpeg", "png"],
        help="Upload a photo of a Cat, Dog, Elephant, Horse, or Lion."
    )

    if uploaded_file is not None:
        try:
            # Load and display uploaded image
            raw_image = Image.open(uploaded_file)
            col_img, col_pred = st.columns([1, 1], gap="medium")

            with col_img:
                st.markdown("#### Uploaded Image")
                st.image(raw_image, caption=f"File: {uploaded_file.name}", use_container_width=True)
                st.caption(
                    f"Resolution: `{raw_image.size[0]} x {raw_image.size[1]}` | "
                    f"Format: `{raw_image.format}` | Mode: `{raw_image.mode}`"
                )

            with col_pred:
                st.markdown("#### Model Prediction")

                if model is None:
                    # Model not trained yet - Helpful educational notice
                    st.warning("⚠️ **Trained Model Weights Not Found**")
                    st.markdown(
                        f"""
                        The CNN model file (`{MODEL_SAVE_PATH.name}`) has not been generated yet.

                        **How to train and connect your model:**
                        1. Place 100-300 images per animal into:  
                           `data/raw/cats/`, `data/raw/dogs/`, `data/raw/elephants/`, `data/raw/horses/`, `data/raw/lions/`
                        2. In your terminal, run:
                           ```powershell
                           py -3.12 train.py
                           ```
                        3. Refresh this page to enable real-time predictions!
                        """
                    )
                    st.info(
                        f"**Preprocessing Check:** Image validated successfully. Target size for inference is "
                        f"`{IMAGE_SIZE[0]} x {IMAGE_SIZE[1]}` with RGB normalization to `[0.0, 1.0]`."
                    )
                else:
                    # Preprocess exactly as trained
                    with st.spinner("Preprocessing image and calculating probabilities..."):
                        preprocessed_tensor = preprocess_image(raw_image, target_size=IMAGE_SIZE)
                        results = predict_animal(model, preprocessed_tensor, class_labels)

                    top_class = results["predicted_class"]
                    top_conf = results["confidence"]
                    top_emoji = CLASS_EMOJIS.get(top_class, "🐾")
                    conf_level = results.get("confidence_level", "Standard Confidence")
                    is_low_conf = results.get("is_low_confidence", False)

                    # Highlight Top Prediction with Low-Confidence Guardrail
                    if is_low_conf:
                        st.warning(
                            f"⚠️ **Low-Confidence Warning ({top_conf:.2f}%)**\n\n"
                            f"The model is uncertain about this image. The image may have unusual angles, "
                            f"heavy background clutter, or contain a subject outside the 5 trained categories."
                        )
                        st.info(f"### {top_emoji} Best Match: **{top_class.capitalize()}**")
                    else:
                        st.success(f"### {top_emoji} Predicted Class: **{top_class.capitalize()}**")

                    st.metric(
                        label="Prediction Confidence",
                        value=f"{top_conf:.2f}%",
                        delta=conf_level,
                    )

                    st.markdown("---")
                    st.markdown("##### 📈 Ranked Confidence Scores (Descending Order):")
                    for cls_name, score in results["all_scores"].items():
                        emoji = CLASS_EMOJIS.get(cls_name, "🐾")
                        col_lbl, col_bar, col_pct = st.columns([2, 5, 2])
                        with col_lbl:
                            st.write(f"{emoji} **{cls_name.capitalize()}**")
                        with col_bar:
                            st.progress(score / 100.0)
                        with col_pct:
                            st.write(f"`{score:.2f}%`")

        except Exception as e:
            st.error(f"❌ Error processing image file: {str(e)}")

    else:
        st.info("👆 Please upload an image using the file uploader above to begin classification.")

    # Model Scope and Disclaimer
    st.markdown("---")
    st.warning(
        """
        ℹ️ **Important Model Limitation & Demonstration Notice:**  
        This model is trained specifically to distinguish among **five animal categories**:  
        **Cats**, **Dogs**, **Elephants**, **Horses**, and **Lions**.  
        * If an image containing an animal outside these 5 classes (or a non-animal object) is uploaded, 
          the model will still assign 100% of its probability across these 5 categories, which may lead to incorrect predictions.  
        * For best accuracy, provide clear, focused photos of the target animals.
        """
    )


# ---------------------------------------------------------------------
# TAB 2: Training & Evaluation Analytics
# ---------------------------------------------------------------------
with tab_analytics:
    st.subheader("Model Performance & Training Metrics")
    st.write(
        "Visual outputs and evaluation metrics measured on the independent holdout test dataset."
    )

    if TRAINING_METRICS_PATH.exists():
        import json
        try:
            with open(TRAINING_METRICS_PATH, "r", encoding="utf-8") as f:
                metrics_data = json.load(f)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Holdout Test Accuracy", f"{metrics_data.get('test_accuracy', 0)*100:.2f}%")
            m2.metric("Holdout Test Loss", f"{metrics_data.get('test_loss', 0):.4f}")
            m3.metric("Epochs Trained", str(metrics_data.get("epochs_trained", "-")))
            m4.metric("Best Val Accuracy", f"{metrics_data.get('best_validation_accuracy', 0)*100:.2f}%")
        except Exception as e:
            st.caption(f"Could not parse metrics file: {e}")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### 1. Training & Validation Curves")
        if TRAINING_HISTORY_PATH.exists():
            st.image(str(TRAINING_HISTORY_PATH), caption="Loss & Accuracy across Epochs", use_container_width=True)
        else:
            st.info("Training history plot will appear here after running `train.py`.")

    with col_b:
        st.markdown("#### 2. Confusion Matrix")
        if CONFUSION_MATRIX_PATH.exists():
            st.image(str(CONFUSION_MATRIX_PATH), caption="Validation Confusion Matrix", use_container_width=True)
        else:
            st.info("Confusion matrix plot will appear here after running `train.py`.")

    st.markdown("---")
    st.markdown("#### 3. Dataset Distribution")
    if CLASS_DISTRIBUTION_PATH.exists():
        st.image(str(CLASS_DISTRIBUTION_PATH), caption="Image Counts per Animal Category", use_container_width=True)
    else:
        st.info("Dataset distribution plot will appear here after running `prepare_dataset.py`.")


# ---------------------------------------------------------------------
# TAB 3: Viva & DL Concepts Guide
# ---------------------------------------------------------------------
with tab_viva:
    st.subheader("College Project Viva Reference Guide")
    st.markdown(
        """
        ### Key Technical Concepts & Answers:

        1. **Why use a Convolutional Neural Network (CNN)?**
           Traditional Dense networks flatten images, destroying 2D spatial relationships. CNNs use 
           sliding kernels (Conv2D) to extract spatial hierarchies—from basic edges to textures and object shapes.

        2. **Why convert to RGB and resize to 128x128?**
           Neural network layers require fixed tensor shapes. Resizing standardizes varying camera resolutions, 
           and converting to 3-channel RGB ensures consistent input dimensions without transparency bugs from PNGs.

        3. **Why normalize pixel values to [0.0, 1.0]?**
           Raw pixels range from 0 to 255. High input values cause large gradients, leading to exploding gradients 
           or slow convergence. Normalizing ($x / 255.0$) speeds up optimization and stabilizes training.

        4. **What does Softmax activation do in the output layer?**
           Softmax transforms unconstrained real-valued outputs (logits) into a valid probability distribution 
           satisfying $\sum_{i=1}^5 p_i = 1.0$.

        5. **Why use Dropout (0.5)?**
           Dropout randomly deactivates 50% of the neurons during training. This forces the network to learn 
           redundant, robust features rather than memorizing noise, dramatically reducing overfitting.

        6. **What is Early Stopping?**
           A callback that monitors validation loss and halts training when loss fails to decrease for 5 consecutive epochs, 
           automatically restoring the best model weights.
        """
    )

# 6. Footer
st.markdown("---")
st.caption("Animal Image Classification Mini Project | BTech Project - Stage 4 Complete")
