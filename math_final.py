import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import cv2
from torchvision import transforms
from streamlit_drawable_canvas import st_canvas
from PIL import Image
from collections import Counter
import re
import pickle
import os

# ===============================
#  Load Models & Mappings (All classes)
# ===============================
@st.cache_resource
def load_models_and_data():
    state_math = torch.load("improved_dataset_state.pth", weights_only=False)
    label_map_math = state_math["label_map"]
    class_name_map_math = state_math["class_name_map"]
    num_classes_math = len(label_map_math)

    class OptimizedSymbolCNN(nn.Module):
        def __init__(self, num_classes, dropout_rate=0.3):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2), nn.Dropout2d(0.1),
                nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2), nn.Dropout2d(0.1),
                nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2), nn.Dropout2d(0.2),
                nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2), nn.Dropout2d(0.2),
            )
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(256, 512), nn.BatchNorm1d(512), nn.ReLU(inplace=True), nn.Dropout(dropout_rate),
                nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True), nn.Dropout(dropout_rate),
                nn.Linear(256, num_classes)
            )

        def forward(self, x):
            x = self.features(x)
            x = self.classifier(x)
            return x

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model_math = OptimizedSymbolCNN(num_classes_math).to(device)
    checkpoint_math = torch.load("best_model_optimized.pth", map_location=device, weights_only=False)
    model_math.load_state_dict(checkpoint_math["model_state_dict"])
    model_math.eval()

    return model_math, device, class_name_map_math, checkpoint_math

model_math, device, class_name_map_math, checkpoint_math = load_models_and_data()
st.info(f"Math Model on {device} (Val Acc: {checkpoint_math.get('val_acc', 'N/A'):.2f}%)")

# Custom dataset file
CUSTOM_FILE = "custom_symbols.pkl"

@st.cache_data
def load_custom():
    if os.path.exists(CUSTOM_FILE):
        with open(CUSTOM_FILE, 'rb') as f:
            return pickle.load(f)
    return []

def save_to_custom(binary_img, label):
    custom = load_custom()
    custom.append((label, binary_img))
    with open(CUSTOM_FILE, 'wb') as f:
        pickle.dump(custom, f)
    st.cache_data.clear()
    st.success("Added to custom dataset!")

# Shared transform
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# ===============================
#  🔧 Updated Preprocessing (Dataset-matched)
# ===============================
def preprocess_image(img, photo_mode=False, thin_mode=False):
    # Convert RGBA → grayscale
    if len(img.shape) == 3:
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Invert only if the background is dark
    if np.mean(img) < 127:
        img = cv2.bitwise_not(img)

    # Clean threshold (keep black ink on white)
    _, binary = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)

    # Crop to bounding box (centering)
    coords = cv2.findNonZero(255 - binary)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        binary = binary[y:y+h, x:x+w]
    else:
        binary = np.ones((45, 45), np.uint8) * 255  # blank fallback

    # Resize with aspect ratio + padding
    desired_size = 45
    old_size = binary.shape[:2]
    ratio = float(desired_size - 8) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])
    binary = cv2.resize(binary, (new_size[1], new_size[0]))
    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    binary = cv2.copyMakeBorder(binary, top, bottom, left, right, cv2.BORDER_CONSTANT, value=255)

    # Convert for model input (black on white)
    binary = cv2.bitwise_not(binary)

    # Apply minor smoothing
    binary = cv2.GaussianBlur(binary, (3, 3), 0)

    pil_img = Image.fromarray(binary.astype(np.uint8))
    processed_tensor = transform(pil_img).unsqueeze(0)
    return processed_tensor, binary

# ===============================
#  Math Model Prediction with Symbol Filtering and Custom Check
# ===============================
def math_predict(img_tensor, binary_img=None, repeats=5):
    def is_symbol(label):
        return not re.match(r'^[0-9a-zA-Z]$', label)

    symbol_label_indices = [i for i, lbl in class_name_map_math.items() if is_symbol(lbl)]

    with torch.no_grad():
        img_tensor_dev = img_tensor.to(device)
        math_preds, math_confs = [], []
        for _ in range(repeats):
            outputs_math = model_math(img_tensor_dev)
            probs_math = torch.softmax(outputs_math, dim=1)
            symbol_probs = probs_math[0][symbol_label_indices]
            pred_relative = symbol_probs.argmax().item()
            pred_abs = symbol_label_indices[pred_relative]
            conf_math = symbol_probs[pred_relative].item()
            math_preds.append(pred_abs)
            math_confs.append(conf_math)
    final_math = Counter(math_preds).most_common(1)[0][0]
    final_conf_math = np.mean([c for p, c in zip(math_preds, math_confs) if p == final_math])
    pred_label_math = class_name_map_math[final_math]
    source = "math_model"

    # Check custom dataset
    if binary_img is not None:
        custom = load_custom()
        if custom:
            flat_query = binary_img.flatten().astype(np.float32)
            norm_query = np.linalg.norm(flat_query)
            if norm_query > 0:
                flat_query /= norm_query
                max_sim = 0
                custom_label = None
                for label, cust_img in custom:
                    flat_cust = cust_img.flatten().astype(np.float32)
                    norm_cust = np.linalg.norm(flat_cust)
                    if norm_cust > 0:
                        flat_cust /= norm_cust
                        sim = np.dot(flat_query, flat_cust)
                        if sim > max_sim:
                            max_sim = sim
                            custom_label = label
                if max_sim > 0.7 and final_conf_math < 0.5:
                    pred_label_math = custom_label
                    final_conf_math = max_sim
                    source = "custom"

    return final_math, pred_label_math, final_conf_math, source

# ===============================
#  Streamlit UI
# ===============================
st.set_page_config(page_title="Math Symbol Live Prediction", layout="wide")
st.title("✏️ Handwritten Math Symbols Live Recognition")

st.markdown("""
🎨 Draw or upload. Math model for symbols (sin, √, λ, +, -, =, etc.). Digits and letters excluded from prediction!
Custom dataset integrated: Correct predictions and add to customs for better future matches.
""")

st.sidebar.header("Options")
thin_mode = st.sidebar.checkbox("Thin Lines? (symbols/ops)", value=False)
photo_tweaks = st.sidebar.checkbox("Photo Tweaks? (noisy imgs)", value=False)

mode = st.radio("Input Mode:", ("🖊️ Draw on Canvas", "📁 Upload Image"))

if 'current_binary' not in st.session_state:
    st.session_state.current_binary = None
if 'current_pred' not in st.session_state:
    st.session_state.current_pred = None
if 'current_conf' not in st.session_state:
    st.session_state.current_conf = None
if 'current_source' not in st.session_state:
    st.session_state.current_source = None

prediction_made = False

if mode == "🖊️ Draw on Canvas":
    canvas_result = st_canvas(fill_color="rgba(255, 255, 255, 1)", stroke_width=8, stroke_color="#000000",
                              background_color="#FFFFFF", height=400, width=400, drawing_mode="freedraw", key="canvas")
    if st.button("🔍 Predict", type="primary"):
        if canvas_result.image_data is not None:
            img = canvas_result.image_data.astype(np.uint8)
            processed_tensor, viz_binary = preprocess_image(img, photo_tweaks, thin_mode)
            pred_idx, pred_label, conf, model_used = math_predict(processed_tensor, viz_binary)
            st.session_state.current_binary = viz_binary
            st.session_state.current_pred = pred_label
            st.session_state.current_conf = conf
            st.session_state.current_source = model_used
            prediction_made = True
            col1, col2 = st.columns(2)
            with col1:
                st.image(img, caption="Original", use_container_width=True)
            with col2:
                st.image(viz_binary, caption=f"Processed (Pred: {pred_label}, Conf: {conf:.3f} via {model_used})", use_container_width=True)
            st.success(f"✅ **{pred_label}** | 📈 Conf: {conf:.3f} | 🧠 Source: {model_used}")
        else:
            st.warning("Draw something!")

elif mode == "📁 Upload Image":
    uploaded_file = st.file_uploader("Upload img", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        img_pil = Image.open(uploaded_file).convert("RGB")
        img = np.array(img_pil)
        st.image(img, caption="Uploaded", use_container_width=True)
        if st.button("🔍 Predict", type="primary"):
            processed_tensor, viz_binary = preprocess_image(img, photo_tweaks, thin_mode)
            pred_idx, pred_label, conf, model_used = math_predict(processed_tensor, viz_binary)
            st.session_state.current_binary = viz_binary
            st.session_state.current_pred = pred_label
            st.session_state.current_conf = conf
            st.session_state.current_source = model_used
            prediction_made = True
            col1, col2 = st.columns(2)
            with col1:
                st.image(img, caption="Original", use_container_width=True)
            with col2:
                st.image(viz_binary, caption=f"Processed (Pred: {pred_label}, Conf: {conf:.3f} via {model_used})", use_container_width=True)
                st.success(f"✅ **{pred_label}** | 📈 Conf: {conf:.3f} | 🧠 Source: {model_used}")

# Custom dataset correction
if st.session_state.current_pred is not None:
    st.markdown("---")
    st.header("📝 Custom Dataset Correction")
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        corrected_label = st.text_input("Correct the prediction if wrong:", value=st.session_state.current_pred, key="corrected_label")
    with col2:
        if st.button("💾 Add to Custom Dataset", type="secondary"):
            if st.session_state.current_binary is not None and corrected_label.strip():
                save_to_custom(st.session_state.current_binary, corrected_label.strip())
                if st.checkbox("Re-predict after adding?"):
                    processed_tensor, _ = preprocess_image(st.session_state.current_binary, photo_tweaks, thin_mode)
                    new_idx, new_label, new_conf, new_source = math_predict(processed_tensor, st.session_state.current_binary)
                    st.session_state.current_pred = new_label
                    st.session_state.current_conf = new_conf
                    st.session_state.current_source = new_source
                    st.rerun()
    with col3:
        if st.button("🗑️ Clear Prediction"):
            st.session_state.current_binary = None
            st.session_state.current_pred = None
            st.session_state.current_conf = None
            st.session_state.current_source = None
            st.rerun()

    custom_data = load_custom()
    if custom_data:
        label_counts = Counter([label for label, _ in custom_data])
        st.subheader("Custom Dataset Overview")
        st.json(dict(label_counts))
    else:
        st.info("No custom entries yet. Add some to improve matching!")

st.markdown("---")
st.markdown("*Math Model + Custom Dataset | Digit & Letter Predictions Filtered Out*")
