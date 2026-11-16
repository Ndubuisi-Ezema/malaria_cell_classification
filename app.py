"""
Malaria Cell Image Research Studio
A Streamlit research interface based on the uploaded notebook:
- binary cell-image classification: Uninfected / Parasitized
- CNN / VGG16 / ResNet50 / EfficientNetB0 transfer-learning workflow
- image augmentation, evaluation, ROC/PR, confusion matrix, Grad-CAM
Research/education only. Not a diagnostic device.
"""
from __future__ import annotations

import io
import os
import zipfile
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

# Optional plotting / metrics / deep-learning dependencies are imported lazily.
st.set_page_config(
    page_title="Malaria Cell Image Research Studio",
    page_icon="🧫",
    layout="wide",
    initial_sidebar_state="expanded",
)

CLASS_NAMES = ["Parasitized", "Uninfected"]
IMG_SIZE = 224

# ------------------------------- Styling ------------------------------------
st.markdown("""
<style>
:root { --ink:#17243a; --muted:#65748b; --line:#e6ebf2; }
.block-container { padding-top: 1.35rem; padding-bottom: 3rem; max-width: 1500px; }
[data-testid="stSidebar"] { border-right: 1px solid #e8edf4; }
.hero {
  padding: 1.35rem 1.5rem; border-radius: 18px;
  background: linear-gradient(115deg,#102b46 0%,#1e5572 58%,#267c83 100%);
  color: white; margin-bottom: 1rem;
}
.hero h1 { color:white; font-size:2rem; margin:0 0 .35rem 0; }
.hero p { color:#dcebf2; margin:0; font-size:1rem; }
.eyebrow { text-transform:uppercase; letter-spacing:.13em; font-size:.72rem;
  font-weight:700; color:#9fd6d5; margin-bottom:.45rem; }
.metric-card { border:1px solid var(--line); border-radius:14px;
  padding: .9rem 1rem; background: white; min-height:100px; }
.metric-label { color:var(--muted); font-size:.82rem; }
.metric-value { color:var(--ink); font-size:1.55rem; font-weight:750; margin-top:.15rem; }
.note { border-left:4px solid #e3a83d; background:#fff8e9; padding:.8rem 1rem;
  border-radius:8px; color:#5c4a25; font-size:.9rem; }
.small-muted { color:#758399; font-size:.82rem; }
div[data-testid="stMetric"] { background:#fff; border:1px solid #e6ebf2;
  padding: .8rem 1rem; border-radius:13px; }
</style>
""", unsafe_allow_html=True)

# ------------------------------- Utilities ----------------------------------
def require_tensorflow():
    try:
        import tensorflow as tf
        return tf
    except Exception as exc:
        st.error("TensorFlow is not available in this environment. Install the packages in requirements.txt, "
                 "or use a Python 3.11 environment with a compatible TensorFlow build.")
        st.code(str(exc))
        return None

def load_model_bytes(uploaded_file):
    tf = require_tensorflow()
    if tf is None:
        return None
    suffix = Path(uploaded_file.name).suffix.lower()
    try:
        if suffix == ".h5":
            return tf.keras.models.load_model(uploaded_file, compile=False)
        if suffix == ".keras":
            # Keras can load from a temporary file more reliably than a file-like object.
            with tempfile.NamedTemporaryFile(suffix=".keras", delete=False) as tmp:
                tmp.write(uploaded_file.getvalue())
                temp_path = tmp.name
            try:
                return tf.keras.models.load_model(temp_path, compile=False)
            finally:
                os.unlink(temp_path)
        st.error("Upload a .keras or .h5 model file.")
    except Exception as exc:
        st.error(f"Could not load model: {exc}")
    return None

def preprocess_image(image: Image.Image, size: int = IMG_SIZE):
    image = ImageOps.exif_transpose(image).convert("RGB").resize((size, size))
    arr = np.asarray(image, dtype=np.float32)
    # The notebook's EfficientNet workflow uses ImageDataGenerator(rescale=1./255).
    return arr / 255.0

def predict_one(model, image: Image.Image, threshold: float):
    arr = preprocess_image(image)
    raw = np.asarray(model.predict(np.expand_dims(arr, 0), verbose=0)).reshape(-1)
    score = float(raw[0])
    # Notebook convention: sigmoid > 0.5 => class index 1 (Parasitized).
    pred_idx = int(score >= threshold)
    return pred_idx, score, arr

def image_bytes(image: Image.Image, fmt="PNG"):
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    return buf.getvalue()

def safe_extract_zip(uploaded_zip, target_dir: str):
    root = Path(target_dir).resolve()
    with zipfile.ZipFile(uploaded_zip) as zf:
        for member in zf.infolist():
            dest = (root / member.filename).resolve()
            if root not in dest.parents and dest != root:
                raise ValueError("ZIP contains an unsafe path.")
        zf.extractall(root)
    return root

def find_dataset_root(path: Path) -> Path:
    # Accept either ZIP containing cell_images/ or direct class folders.
    if (path / "Parasitized").is_dir() and (path / "Uninfected").is_dir():
        return path
    candidates = [p for p in path.rglob("*") if p.is_dir()
                  and (p / "Parasitized").is_dir() and (p / "Uninfected").is_dir()]
    if candidates:
        return candidates[0]
    raise ValueError("Could not find sibling folders named 'Parasitized' and 'Uninfected'.")

def list_images(folder: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in exts]

def get_last_conv_layer(model):
    # Prefer EfficientNetB0's named final convolution layer used in the notebook.
    try:
        layer = model.get_layer("top_conv")
        return layer.name
    except Exception:
        pass
    for layer in reversed(model.layers):
        if isinstance(layer, __import__("tensorflow").keras.layers.Conv2D):
            return layer.name
    return None

def make_gradcam(model, image_array):
    tf = require_tensorflow()
    if tf is None:
        return None
    # Find the final Conv2D layer, including inside a nested transfer-learning backbone.
    nested_backbone = None
    conv_name = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.Model):
            for sub in reversed(layer.layers):
                if isinstance(sub, tf.keras.layers.Conv2D):
                    nested_backbone = layer
                    conv_name = sub.name
                    break
            if conv_name:
                break
    try:
        if nested_backbone is not None:
            conv_tensor = nested_backbone.get_layer(conv_name).output
            grad_model = tf.keras.Model(
                inputs=model.input, outputs=[conv_tensor, model.output]
            )
        else:
            for layer in reversed(model.layers):
                if isinstance(layer, tf.keras.layers.Conv2D):
                    conv_name = layer.name
                    break
            if not conv_name:
                raise ValueError("No accessible Conv2D layer was found for Grad-CAM.")
            grad_model = tf.keras.Model(
                inputs=model.input,
                outputs=[model.get_layer(conv_name).output, model.output]
            )
        x = tf.convert_to_tensor(np.expand_dims(image_array, 0))
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(x, training=False)
            score = predictions[:, 0]
        grads = tape.gradient(score, conv_outputs)
        if grads is None:
            raise ValueError("Gradients were not available for the selected convolutional layer.")
        weights = tf.reduce_mean(grads, axis=(1, 2))
        heatmap = tf.reduce_sum(conv_outputs * weights[:, None, None, :], axis=-1)[0]
        heatmap = tf.maximum(heatmap, 0)
        heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)
        return heatmap.numpy()
    except Exception as exc:
        raise ValueError(f"Grad-CAM could not access the model's convolutional graph: {exc}")

def overlay_heatmap(original: Image.Image, heatmap: np.ndarray, alpha: float = 0.38):
    import matplotlib.cm as cm
    base = np.asarray(original.convert("RGB").resize((IMG_SIZE, IMG_SIZE))).astype(np.float32)
    h = Image.fromarray(np.uint8(np.clip(heatmap, 0, 1) * 255)).resize((IMG_SIZE, IMG_SIZE))
    color = (cm.get_cmap("jet")(np.asarray(h) / 255.0)[:, :, :3] * 255).astype(np.float32)
    mixed = np.clip((1-alpha)*base + alpha*color, 0, 255).astype(np.uint8)
    return Image.fromarray(mixed), Image.fromarray(np.uint8(np.asarray(h)))

def build_transfer_model(architecture: str, dropout: float):
    tf = require_tensorflow()
    if tf is None:
        return None
    from tensorflow.keras import layers, Model
    apps = tf.keras.applications
    constructors = {
        "EfficientNetB0": apps.EfficientNetB0,
        "VGG16": apps.VGG16,
        "ResNet50": apps.ResNet50,
    }
    base = constructors[architecture](weights="imagenet", include_top=False,
                                      input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base.trainable = False
    x = base.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    model = Model(inputs=base.input, outputs=out, name=f"{architecture}_malaria_classifier")
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model, base

def make_figure():
    import matplotlib.pyplot as plt
    return plt

def save_model_download(model):
    tf = require_tensorflow()
    if tf is None:
        return None
    with tempfile.NamedTemporaryFile(suffix=".keras", delete=False) as tmp:
        path = tmp.name
    try:
        model.save(path)
        return Path(path).read_bytes()
    finally:
        if os.path.exists(path):
            os.unlink(path)

# ------------------------------- Sidebar ------------------------------------
with st.sidebar:
    st.markdown("## 🧫 Research Studio")
    st.caption("Malaria cell image classification")
    page = st.radio(
        "Workspace",
        ["Overview", "Single image", "Batch screening", "Dataset explorer",
         "Model evaluation", "Train a model", "Help & methodology"],
        label_visibility="collapsed"
    )
    st.divider()
    st.markdown("### Model workspace")
    model_file = st.file_uploader("Load trained model (.keras / .h5)", type=["keras", "h5"],
                                  help="Upload a model trained with the notebook workflow.")
    threshold = st.slider("Parasitized threshold", 0.05, 0.95, 0.50, 0.01,
                          help="Adjusts the probability cut-off used for the binary class label.")
    if model_file is not None:
        if st.button("Load / replace model", use_container_width=True):
            st.session_state["loaded_model"] = load_model_bytes(model_file)
            st.session_state["model_filename"] = model_file.name
    if st.session_state.get("loaded_model") is not None:
        st.success(f"Model ready: {st.session_state.get('model_filename','uploaded model')}")
    else:
        st.info("Upload a trained model to enable predictions.")
    st.divider()
    st.caption("Notebook-aligned classes")
    st.markdown("🟥 **Parasitized** · class 0  \n🟦 **Uninfected** · class 1")

model = st.session_state.get("loaded_model")
# ------------------------------- Header -------------------------------------
st.markdown("""
<div class="hero">
  <div class="eyebrow">Deep learning · Transfer learning · Explainable AI</div>
  <h1>Malaria Cell Image Research Studio</h1>
  <p>Explore microscopy images, run model predictions, evaluate performance and inspect Grad-CAM explanations.</p>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="note"><b>Research-use notice:</b> This application is an educational/research prototype. '
            'It is not a medical device and must not be used to diagnose, screen, or make treatment decisions. '
            'Model outputs require expert review and independent validation.</div>', unsafe_allow_html=True)
st.write("")

# -------------------------------- Overview -----------------------------------
if page == "Overview":
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Image size", "224 × 224", "RGB")
    with c2: st.metric("Classes", "2", "Binary sigmoid")
    with c3: st.metric("Architectures", "4", "CNN + 3 transfer models")
    with c4: st.metric("Explainability", "Grad-CAM", "Heatmap overlay")
    st.subheader("Your analysis workspace")
    cards = st.columns(3)
    descriptions = [
        ("🖼️", "Single image", "Upload one cell image, inspect the class score and generate a Grad-CAM heatmap."),
        ("⚡", "Batch screening", "Run predictions over multiple images and export a sortable CSV report."),
        ("📊", "Model evaluation", "Evaluate a model against labelled folders and inspect confusion matrix, ROC and PR curves."),
        ("🗂️", "Dataset explorer", "Inspect class balance, sample images, image dimensions and augmentation previews."),
        ("🧠", "Train a model", "Train a transfer-learning classifier from a ZIP dataset with adjustable settings."),
        ("🔬", "Explainability", "Compare the original cell image, activation heatmap and blended overlay.")
    ]
    for i, (ico, title, desc) in enumerate(descriptions):
        with cards[i % 3]:
            st.markdown(f"### {ico} {title}")
            st.write(desc)
            st.write("")
    st.subheader("Notebook workflow")
    st.code("Dataset → preprocessing / augmentation → CNN & transfer learning → evaluation → Grad-CAM", language="text")
    st.info("Keras' default alphabetical folder order in the notebook is Parasitized = 0 and Uninfected = 1. The notebook's classification-report target-name order conflicts with that mapping; verify labels before interpreting results.")

# ----------------------------- Single image ---------------------------------
elif page == "Single image":
    st.subheader("Single-image prediction")
    st.write("Upload a Giemsa-stained cell image. The model input is resized to 224 × 224 RGB and scaled to [0, 1].")
    uploaded = st.file_uploader("Choose a cell image", type=["jpg", "jpeg", "png", "bmp", "webp"], key="single_img")
    if uploaded:
        img = Image.open(uploaded)
        left, right = st.columns([1, 1.2], gap="large")
        with left:
            st.image(img, caption=uploaded.name, use_container_width=True)
            st.caption(f"Original dimensions: {img.width} × {img.height}")
        with right:
            if model is None:
                st.warning("Load a trained .keras or .h5 model from the sidebar to run prediction.")
            else:
                if st.button("Analyse image", type="primary", use_container_width=True):
                    with st.spinner("Running model inference…"):
                        try:
                            pred_idx, score, arr = predict_one(model, img, threshold)
                            st.session_state["single_result"] = {
                                "name": uploaded.name, "pred": pred_idx, "score": score,
                                "arr": arr, "image": img.copy()
                            }
                        except Exception as exc:
                            st.error(f"Prediction failed: {exc}")
                result = st.session_state.get("single_result")
                if result and result["name"] == uploaded.name:
                    pred_idx, score = result["pred"], result["score"]
                    label = CLASS_NAMES[pred_idx]
                    st.markdown(f"### {'🟥' if pred_idx else '🟦'} {label}")
                    st.metric("Class 1 score · Uninfected", f"{score:.1%}")
                    st.progress(min(max(score, 0), 1))
                    st.caption(f"Decision threshold: {threshold:.2f}")
                    st.write("Class scores (notebook folder-order mapping)")
                    st.progress(1-score, text=f"Parasitized · {1-score:.1%}")
                    st.progress(score, text=f"Uninfected · {score:.1%}")
                    if st.button("Generate Grad-CAM explanation", use_container_width=True):
                        try:
                            with st.spinner("Computing Grad-CAM…"):
                                hm = make_gradcam(model, result["arr"])
                                overlay, heat = overlay_heatmap(result["image"], hm)
                                st.session_state["single_gradcam"] = (overlay, heat)
                        except Exception as exc:
                            st.warning(str(exc))
                    if st.session_state.get("single_gradcam") is not None:
                        a, b = st.columns(2)
                        with a: st.image(st.session_state["single_gradcam"][1], caption="Grad-CAM heatmap", use_container_width=True)
                        with b: st.image(st.session_state["single_gradcam"][0], caption="Activation overlay", use_container_width=True)
                        st.download_button("Download Grad-CAM overlay", image_bytes(st.session_state["single_gradcam"][0]),
                                           file_name="gradcam_overlay.png", mime="image/png")
                    st.caption("The score is a model output, not a clinical probability or diagnosis.")

# ------------------------------ Batch screening ----------------------------
elif page == "Batch screening":
    st.subheader("Batch image analysis")
    st.write("Upload several cell images at once. Predictions are processed one image at a time and can be exported.")
    files = st.file_uploader("Select images", type=["jpg", "jpeg", "png", "bmp", "webp"],
                             accept_multiple_files=True, key="batch_files")
    if files:
        st.write(f"**{len(files)}** image(s) selected")
        if model is None:
            st.warning("Load a trained model in the sidebar before running batch analysis.")
        elif st.button("Run batch analysis", type="primary"):
            rows = []
            progress = st.progress(0)
            for i, f in enumerate(files):
                try:
                    im = Image.open(f)
                    idx, score, _ = predict_one(model, im, threshold)
                    rows.append({"filename": f.name, "predicted_class": CLASS_NAMES[idx],
                                 "uninfected_score": score, "parasitized_score": 1-score,
                                 "threshold": threshold, "status": "OK"})
                except Exception as exc:
                    rows.append({"filename": f.name, "predicted_class": "Error",
                                 "parasitized_score": np.nan, "uninfected_score": np.nan,
                                 "threshold": threshold, "status": str(exc)})
                progress.progress((i+1)/len(files))
            st.session_state["batch_results"] = pd.DataFrame(rows)
        df = st.session_state.get("batch_results")
        if df is not None:
            valid = df[df["status"] == "OK"]
            a,b,c = st.columns(3)
            a.metric("Processed", len(valid))
            b.metric("Parasitized predictions", int((valid["predicted_class"]=="Parasitized").sum()))
            c.metric("Uninfected predictions", int((valid["predicted_class"]=="Uninfected").sum()))
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.download_button("Download batch CSV", df.to_csv(index=False).encode("utf-8"),
                               "malaria_batch_predictions.csv", "text/csv")

# ----------------------------- Dataset explorer ----------------------------
elif page == "Dataset explorer":
    st.subheader("Dataset explorer")
    st.write("Upload a ZIP containing the two class folders, or a folder structure archived with a parent directory.")
    ds_zip = st.file_uploader("Dataset ZIP", type=["zip"], key="explore_zip")
    if ds_zip:
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = safe_extract_zip(ds_zip, tmp)
                ds = find_dataset_root(root)
                paths = {cls: list_images(ds / cls) for cls in CLASS_NAMES}
                counts = pd.DataFrame({"Class": CLASS_NAMES, "Images": [len(paths[c]) for c in CLASS_NAMES]})
                a,b = st.columns([1,1])
                with a:
                    st.markdown("#### Class distribution")
                    st.bar_chart(counts.set_index("Class"))
                    st.dataframe(counts, hide_index=True, use_container_width=True)
                with b:
                    st.markdown("#### Dataset summary")
                    total = sum(len(v) for v in paths.values())
                    st.metric("Total images", total)
                    st.metric("Class folders found", 2)
                    st.caption("Supported formats: JPG, JPEG, PNG, BMP, WEBP")
                st.markdown("#### Random image gallery")
                sample_count = st.slider("Images per class", 1, 8, 4, key="gallery_n")
                cols = st.columns(sample_count)
                for cls in CLASS_NAMES:
                    st.markdown(f"**{cls}**")
                    sample = paths[cls][:sample_count]
                    if not sample:
                        st.info(f"No images found in {cls}.")
                    for j, path in enumerate(sample):
                        with cols[j]:
                            im = Image.open(path).convert("RGB")
                            st.image(im, caption=f"{im.width}×{im.height}", use_container_width=True)
                st.markdown("#### Augmentation preview")
                if paths["Parasitized"] or paths["Uninfected"]:
                    all_paths = paths["Parasitized"] + paths["Uninfected"]
                    pick = st.selectbox("Choose source image", [str(p.relative_to(ds)) for p in all_paths])
                    source = Image.open(ds / pick).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
                    from PIL import ImageEnhance
                    aug_cols = st.columns(4)
                    variants = [
                        ("Original", source),
                        ("Horizontal flip", ImageOps.mirror(source)),
                        ("Brightness", ImageEnhance.Brightness(source).enhance(1.2)),
                        ("Contrast", ImageEnhance.Contrast(source).enhance(1.25))
                    ]
                    for col, (label, im) in zip(aug_cols, variants):
                        with col: st.image(im, caption=label, use_container_width=True)
        except Exception as exc:
            st.error(f"Dataset could not be read: {exc}")

# ------------------------------ Evaluation ----------------------------------
elif page == "Model evaluation":
    st.subheader("Evaluate on a labelled dataset")
    st.write("Upload a ZIP with `Uninfected/` and `Parasitized/` folders. Images are evaluated in folder order.")
    eval_zip = st.file_uploader("Labelled evaluation dataset ZIP", type=["zip"], key="eval_zip")
    if eval_zip and model is None:
        st.warning("Load a trained model from the sidebar first.")
    if eval_zip and model is not None and st.button("Evaluate model", type="primary"):
        try:
            from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                                         confusion_matrix, classification_report, roc_curve, auc,
                                         precision_recall_curve, average_precision_score)
            with tempfile.TemporaryDirectory() as tmp:
                root = safe_extract_zip(eval_zip, tmp)
                ds = find_dataset_root(root)
                rows, y_true, y_score = [], [], []
                all_files = []
                for idx, cls in enumerate(CLASS_NAMES):
                    all_files.extend([(p, idx) for p in list_images(ds / cls)])
                if not all_files:
                    raise ValueError("No supported images were found.")
                prog = st.progress(0)
                for i, (path, truth) in enumerate(all_files):
                    im = Image.open(path).convert("RGB")
                    _, score, _ = predict_one(model, im, threshold)
                    pred = int(score >= threshold)
                    y_true.append(truth); y_score.append(score)
                    rows.append({"filename": path.name, "actual": CLASS_NAMES[truth],
                                 "predicted": CLASS_NAMES[pred], "uninfected_score": score,
                                 "correct": truth == pred})
                    prog.progress((i+1)/len(all_files))
                y_true = np.asarray(y_true); y_score = np.asarray(y_score)
                y_pred = (y_score >= threshold).astype(int)
                st.session_state["evaluation"] = {
                    "rows": pd.DataFrame(rows), "y_true": y_true, "y_score": y_score,
                    "y_pred": y_pred
                }
        except Exception as exc:
            st.error(f"Evaluation failed: {exc}")
    ev = st.session_state.get("evaluation")
    if ev is not None:
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
        yt, yp, ys = ev["y_true"], ev["y_pred"], ev["y_score"]
        m1,m2,m3,m4 = st.columns(4)
        m1.metric("Accuracy", f"{accuracy_score(yt,yp):.3f}")
        m2.metric("Precision", f"{precision_score(yt,yp,zero_division=0):.3f}")
        m3.metric("Recall / sensitivity", f"{recall_score(yt,yp,zero_division=0):.3f}")
        m4.metric("F1 score", f"{f1_score(yt,yp,zero_division=0):.3f}")
        cm = confusion_matrix(yt, yp, labels=[0,1])
        c1,c2 = st.columns(2)
        with c1:
            st.markdown("#### Confusion matrix")
            st.dataframe(pd.DataFrame(cm, index=["Actual Uninfected","Actual Parasitized"],
                                      columns=["Pred Uninfected","Pred Parasitized"]),
                         use_container_width=True)
            st.caption("Rows = actual class; columns = predicted class.")
        with c2:
            st.markdown("#### Classification report")
            st.code(classification_report(yt, yp, target_names=CLASS_NAMES, zero_division=0))
        try:
            import matplotlib.pyplot as plt
            from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
            fpr,tpr,_ = roc_curve(yt,ys); roc_auc = auc(fpr,tpr)
            prec,rec,_ = precision_recall_curve(yt,ys)
            fig, ax = plt.subplots(1,2,figsize=(10,4))
            ax[0].plot(fpr,tpr,label=f"AUC = {roc_auc:.3f}"); ax[0].plot([0,1],[0,1],"--")
            ax[0].set(xlabel="False positive rate",ylabel="True positive rate",title="ROC curve"); ax[0].legend()
            ap = average_precision_score(yt,ys)
            ax[1].plot(rec,prec,label=f"AP = {ap:.3f}")
            ax[1].set(xlabel="Recall",ylabel="Precision",title="Precision–recall curve"); ax[1].legend()
            fig.tight_layout(); st.pyplot(fig); plt.close(fig)
        except Exception as exc:
            st.info(f"Curve plots unavailable: {exc}")
        st.markdown("#### Per-image results")
        st.dataframe(ev["rows"], use_container_width=True, hide_index=True)
        st.download_button("Download evaluation report", ev["rows"].to_csv(index=False).encode(),
                           "malaria_model_evaluation.csv", "text/csv")

# -------------------------------- Training -----------------------------------
elif page == "Train a model":
    st.subheader("Train a transfer-learning model")
    st.write("Train using a ZIP containing `Uninfected/` and `Parasitized/` folders. Training runs in this Streamlit session.")
    st.warning("Training can be resource-intensive. On free cloud hosting, use a small dataset and few epochs; a local GPU is recommended.")
    train_zip = st.file_uploader("Training dataset ZIP", type=["zip"], key="train_zip")
    left,right = st.columns(2)
    with left:
        architecture = st.selectbox("Backbone", ["EfficientNetB0","VGG16","ResNet50"])
        epochs = st.slider("Epochs", 1, 20, 5)
        batch = st.select_slider("Batch size", options=[8,16,32], value=16)
    with right:
        val_split = st.slider("Validation split", 0.10, 0.40, 0.20, 0.05)
        dropout = st.slider("Dropout", 0.10, 0.70, 0.30, 0.05)
        fine_tune = st.checkbox("Fine-tune final backbone layers", value=False)
        fine_layers = st.slider("Trainable final layers", 5, 40, 20, 5, disabled=not fine_tune)
    if train_zip and st.button("Start training", type="primary"):
        tf = require_tensorflow()
        if tf is not None:
            try:
                from tensorflow.keras.preprocessing.image import ImageDataGenerator
                with tempfile.TemporaryDirectory() as tmp:
                    root = safe_extract_zip(train_zip, tmp)
                    ds = find_dataset_root(root)
                    train_gen = ImageDataGenerator(
                        rescale=1./255, validation_split=val_split,
                        rotation_range=20, zoom_range=.15, width_shift_range=.10,
                        height_shift_range=.10, horizontal_flip=True
                    ).flow_from_directory(str(ds), target_size=(IMG_SIZE,IMG_SIZE),
                        batch_size=batch, class_mode="binary", subset="training",
                        shuffle=True, seed=42, classes=CLASS_NAMES)
                    valid_gen = ImageDataGenerator(
                        rescale=1./255, validation_split=val_split
                    ).flow_from_directory(str(ds), target_size=(IMG_SIZE,IMG_SIZE),
                        batch_size=batch, class_mode="binary", subset="validation",
                        shuffle=False, seed=42, classes=CLASS_NAMES)
                    built = build_transfer_model(architecture, dropout)
                    if built is None: st.stop()
                    trained_model, base = built
                    if fine_tune:
                        base.trainable = True
                        for layer in base.layers[:-fine_layers]:
                            layer.trainable = False
                        trained_model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
                                              loss="binary_crossentropy", metrics=["accuracy"])
                    prog = st.progress(0)
                    status = st.empty()
                    history_rows = []
                    for epoch in range(epochs):
                        status.write(f"Training epoch {epoch+1} of {epochs}…")
                        hist = trained_model.fit(train_gen, validation_data=valid_gen,
                                                 epochs=1, verbose=0)
                        history_rows.append({k: float(v[-1]) for k,v in hist.history.items()})
                        prog.progress((epoch+1)/epochs)
                    st.session_state["trained_model"] = trained_model
                    st.session_state["training_history"] = pd.DataFrame(history_rows)
                    st.session_state["model_filename"] = f"{architecture}_trained_in_app.keras"
                    status.success("Training completed.")
            except Exception as exc:
                st.error(f"Training failed: {exc}")
    if st.session_state.get("training_history") is not None:
        st.markdown("#### Training history")
        hist_df = st.session_state["training_history"]
        st.dataframe(hist_df, use_container_width=True)
        try:
            st.line_chart(hist_df[["accuracy","val_accuracy"]])
            st.line_chart(hist_df[["loss","val_loss"]])
        except Exception:
            pass
        trained = st.session_state.get("trained_model")
        if trained is not None:
            st.download_button("Download trained model (.keras)", save_model_download(trained),
                               f"{architecture}_malaria_model.keras",
                               "application/octet-stream")
            if st.button("Use this model for predictions"):
                st.session_state["loaded_model"] = trained
                st.session_state["model_filename"] = f"{architecture} (trained this session)"
                st.success("Trained model selected. Open Single image or Batch screening.")

# --------------------------- Help & methodology ----------------------------
elif page == "Help & methodology":
    st.subheader("Help & methodology")
    with st.expander("Expected dataset layout", expanded=True):
        st.code("""dataset.zip
└── cell_images/
    ├── Parasitized/
    │   ├── image_001.png
    │   └── ...
    └── Uninfected/
        ├── image_001.png
        └── ...""")
        st.write("The ZIP may also contain the two class folders directly at its root.")
    with st.expander("Model compatibility"):
        st.markdown("""
- Upload a TensorFlow/Keras `.keras` or `.h5` model.
- Expected input: RGB image, 224 × 224, scaled to [0, 1].
- Expected output: one sigmoid score; values at or above the threshold map to class index 1.
- Keras `flow_from_directory` sorts folder names alphabetically by default. With the notebook's folders, `Parasitized` = 0 and `Uninfected` = 1.
- The notebook's classification report supplies `target_names=['Uninfected','Parasitized']`, which reverses that folder order. Verify the mapping of any saved model before interpreting labels.
- Models trained with a different preprocessing pipeline or reversed class mapping may produce invalid labels.
        """)
    with st.expander("Notebook methods"):
        st.markdown("""
1. **Preprocessing:** 224 × 224 RGB images, batch size 16 in the notebook, rescaling by 1/255 and an 80/20 training-validation split.
2. **Augmentation:** rotations, zoom, horizontal/vertical shifts and horizontal flips.
3. **Models:** baseline CNN, VGG16, ResNet50 and EfficientNetB0 with ImageNet weights and a binary sigmoid head.
4. **Evaluation:** accuracy/loss curves, classification report, confusion matrix and ROC curve.
5. **Explainability:** Grad-CAM heatmap and overlay using convolutional activations.
        """)
    with st.expander("Troubleshooting"):
        st.markdown("""
- **Model load fails:** confirm the model is a TensorFlow/Keras `.keras` or `.h5` artifact.
- **Predictions seem reversed:** verify training class order and that `Parasitized` is class index 1.
- **Grad-CAM unavailable:** some saved model formats or nested architectures do not expose a compatible convolutional graph.
- **Out of memory:** reduce batch size, image count or epoch count; consider local execution.
- **TensorFlow install issues:** use Python 3.11 and install the pinned packages in a clean virtual environment.
        """)
    st.markdown("### Responsible use")
    st.write("The app is a research prototype. It does not provide a diagnosis, does not replace microscopy review, "
             "and has not been clinically validated. Avoid uploading identifiable or sensitive patient information.")
