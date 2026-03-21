# math_expression_recognizer.py
# Combined Streamlit app: digits+characters (Keras) + math symbols (PyTorch)
# Requires: streamlit, streamlit_drawable_canvas, tensorflow/keras, torch, torchvision, opencv-python, pillow, sympy, numpy, h5py, keras
# Run: streamlit run math_expression_recognizer.py

import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import cv2
from PIL import Image
import io
import os
import time
import re

# Keras/TensorFlow imports (digits & characters)
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input
from tensorflow.keras.utils import to_categorical

# PyTorch imports (math symbols)
import torch
import torch.nn as nn
from torchvision import transforms

# For solving/evaluating math
from sympy import sympify, simplify, solve, Eq, Symbol
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application

# ---------------------------
# ---------- CONFIG ----------
# ---------------------------

# Adjust these paths to where your weights/checkpoints are located
DIGIT_WEIGHTS = "/Users/mac/Documents/math recognizer and solver/colab_weights/improved1.weights.h5"
CHAR_WEIGHTS = "/Users/mac/Documents/math recognizer and solver/colab_weights/checkpoint.weights.h5"
CHAR_MAPPING_FILE = "/Users/mac/Documents/math recognizer and solver/colab_weights/emnist-bymerge-mapping (1).txt"

# PyTorch math symbol model checkpoints (from your second script)
MATH_STATE_FILE = "improved_dataset_state.pth"      # contains label_map/class_name_map
MATH_CHECKPOINT = "best_model_optimized.pth"       # contains model_state_dict

# ---------------------------
# ------- Utilities ---------
# ---------------------------

st.set_page_config(page_title="Handwritten Math Expression Recognizer & Solver", layout="wide")
st.title("✏️ Live Handwritten Math Expression Recognizer + Solver")

@st.cache_resource
def load_emnist_mapping(path):
    mapping = {}
    if not os.path.exists(path):
        return {}
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                label, ascii_val = map(int, parts[:2])
                mapping[label] = chr(ascii_val)
    return mapping

# ---------------------------
# --- Load Keras models -----
# ---------------------------

@st.cache_resource
def build_and_load_keras_models():
    # Build digit model architecture (same as your first script)
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization, GlobalAveragePooling2D, Add
    from tensorflow.keras import Input
    from tensorflow.keras.models import Model

    # Digit model (same architecture)
    def build_digit_model():
        input_layer = Input(shape=(28, 28, 1))
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
        x = BatchNormalization()(x)
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(0.25)(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(0.25)(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = Flatten()(x)
        x = Dense(128, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        output_layer = Dense(10, activation='softmax')(x)
        return Model(inputs=input_layer, outputs=output_layer)

    # Character model (same architecture)
    def build_character_model():
        input_layer = Input(shape=(28, 28, 1))
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(input_layer)
        x = BatchNormalization()(x)
        shortcut = x
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = Add()([shortcut, x])
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(0.3)(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        shortcut = x
        x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = Add()([shortcut, x])
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(0.3)(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        shortcut = x
        x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = Add()([shortcut, x])
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(0.4)(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        shortcut = x
        x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = Add()([shortcut, x])
        x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.4)(x)
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        output_layer = Dense(37, activation='softmax')(x)
        return Model(inputs=input_layer, outputs=output_layer)

    digit_model = build_digit_model()
    character_model = build_character_model()

    # Load weights if available (if paths exist)
    if os.path.exists(DIGIT_WEIGHTS):
        try:
            digit_model.load_weights(DIGIT_WEIGHTS)
            st.info("Loaded digit model weights")
        except Exception as e:
            st.warning(f"Failed loading digit weights: {e}")
    else:
        st.warning(f"Digit weights not found at {DIGIT_WEIGHTS}")

    if os.path.exists(CHAR_WEIGHTS):
        try:
            character_model.load_weights(CHAR_WEIGHTS)
            st.info("Loaded character model weights")
        except Exception as e:
            st.warning(f"Failed loading character weights: {e}")
    else:
        st.warning(f"Character weights not found at {CHAR_WEIGHTS}")

    # Prepare mappings for characters (EMNIST)
    mapping = load_emnist_mapping(CHAR_MAPPING_FILE)
    # In your code you removed first 10 labels (digits) and created new_mapping
    new_mapping = {i: mapping[i + 10] for i in range(37)} if mapping else {}

    return digit_model, character_model, new_mapping

digit_model, character_model, new_char_mapping = build_and_load_keras_models()

# ---------------------------
# --- Load PyTorch Math Model
# ---------------------------

@st.cache_resource
def load_math_model():
    # load state that contains class_name_map and label_map
    if not os.path.exists(MATH_STATE_FILE) or not os.path.exists(MATH_CHECKPOINT):
        st.warning("Math model files not found. Please ensure improved_dataset_state.pth and best_model_optimized.pth exist.")
        return None, None, {}

    import torch.serialization

# Allow TensorDataset objects to be loaded safely
    torch.serialization.add_safe_globals([torch.utils.data.dataset.TensorDataset])

# Load the full model (set weights_only=False to restore older behavior)
    state_math = torch.load(MATH_STATE_FILE, map_location="cpu", weights_only=False)

    class_name_map_math = state_math.get("class_name_map", {})
    label_map_math = state_math.get("label_map", {})
    num_classes_math = len(label_map_math)

    # define the same network as your second script (OptimizedSymbolCNN)
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
    checkpoint_math = torch.load(MATH_CHECKPOINT, map_location=device)
    model_math.load_state_dict(checkpoint_math["model_state_dict"])
    model_math.eval()

    return model_math, device, class_name_map_math

model_math, math_device, class_name_map_math = load_math_model()

# ---------------------------
# --- Preprocessing helpers
# ---------------------------

# Prepare transforms for math model (same as your second app)
math_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

def preprocess_for_digit_char(img_crop):
    """
    img_crop: grayscale crop (numpy uint8) with foreground dark on white background expected.
    Returns 28x28 normalized tensor suitable for keras models.
    """
    # ensure grayscale
    if len(img_crop.shape) == 3:
        img_crop = cv2.cvtColor(img_crop, cv2.COLOR_RGB2GRAY)

    # threshold and invert if necessary so that foreground is white on black for MNIST-like models
    _, th = cv2.threshold(img_crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # MNIST digits originally black background, white stroke. Your training pipeline uses inversion sometimes.
    # We'll resize to 28x28 keeping aspect ratio: center in 28x28
    h, w = th.shape
    # pad to square
    size = max(h, w)
    pad_top = (size - h) // 2
    pad_bottom = size - h - pad_top
    pad_left = (size - w) // 2
    pad_right = size - w - pad_left
    squared = cv2.copyMakeBorder(th, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=255)
    resized = cv2.resize(squared, (28, 28), interpolation=cv2.INTER_AREA)

    # Some of your EMNIST preprocessing included flipping/rotating; try to match that if needed:
    # rotated = np.fliplr(np.rot90(resized, k=-1))  # optional if characters appear rotated
    # For now, keep simple:
    final = resized.astype('float32') / 255.0
    # Make model input shape (1, 28, 28, 1)
    return final.reshape(1, 28, 28, 1)

def preprocess_for_math(img_crop):
    """
    Prepare a 45x45-ish padded input for your PyTorch math model (binary_img expected).
    We return the torch tensor and also a 'viz_binary' for custom comparisons.
    """
    if len(img_crop.shape) == 3:
        img_crop = cv2.cvtColor(img_crop, cv2.COLOR_RGB2GRAY)

    # invert if background dark:
    if np.mean(img_crop) < 127:
        img_crop = cv2.bitwise_not(img_crop)

    # threshold to get clean binary
    _, binary = cv2.threshold(img_crop, 200, 255, cv2.THRESH_BINARY)

    # crop to non-zero
    coords = cv2.findNonZero(255 - binary)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        binary = binary[y:y+h, x:x+w]
    else:
        binary = np.ones((45, 45), dtype=np.uint8) * 255

    desired_size = 45
    old_size = binary.shape[:2]
    ratio = float(desired_size - 8) / max(old_size)
    if ratio <= 0:
        ratio = 1.0
    new_size = tuple([int(x * ratio) for x in old_size])
    # avoid zero
    new_size = (max(1, new_size[0]), max(1, new_size[1]))
    binary = cv2.resize(binary, (new_size[1], new_size[0]))
    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    binary = cv2.copyMakeBorder(binary, top, bottom, left, right, cv2.BORDER_CONSTANT, value=255)
    binary = cv2.bitwise_not(binary)  # black-on-white -> white-on-black for model as in your script
    binary = cv2.GaussianBlur(binary, (3,3), 0)

    # convert to tensor for math model
    pil = Image.fromarray(binary.astype(np.uint8))
    tensor = math_transform(pil).unsqueeze(0)  # shape (1,1,H,W)
    return tensor, binary

# ---------------------------
# ---- Segmentation logic ----
# ---------------------------

def segment_image_full(expression_img):
    """
    Input: RGB or grayscale image sized e.g. 400x400 from canvas/upload.
    Output: list of (x,y,w,h,crop) sorted left->right
    """
    # convert to grayscale
    if len(expression_img.shape) == 3:
        gray = cv2.cvtColor(expression_img, cv2.COLOR_RGB2GRAY)
    else:
        gray = expression_img.copy()

    # Smooth & threshold
    blurred = cv2.GaussianBlur(gray, (3,3), 0)
    # Otsu or adaptive threshold to pick foreground
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Some drawn strokes might be narrow — dilate a bit
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    dilated = cv2.dilate(thresh, kernel, iterations=1)

    # find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    height, width = gray.shape
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        # filter out tiny noise
        if w*h < 20:
            continue
        # optionally expand box a little
        pad = 5
        x0 = max(0, x-pad)
        y0 = max(0, y-pad)
        x1 = min(width, x+w+pad)
        y1 = min(height, y+h+pad)
        boxes.append((x0,y0,x1-x0,y1-y0))

    # Sort left-to-right, top-to-bottom fallback (for multi-line)
    boxes = sorted(boxes, key=lambda b: (b[1], b[0]))  # by y then x
    # attempt to group into lines by y coordinate and then sort within lines left->right
    # simple single-line sort by x:
    boxes = sorted(boxes, key=lambda b: b[0])
    crops = []
    for (x,y,w,h) in boxes:
        crop = gray[y:y+h, x:x+w]
        crops.append((x,y,w,h,crop))
    return crops

# ---------------------------
# ---- Unified prediction ----
# ---------------------------

def is_symbol_label(label):
    # consider as "math symbol" anything that is not alphanumeric (digits or letters)
    # adjust this depending on your math model labels (if model uses e.g. "sin" or "sqrt" names)
    return not re.match(r'^[0-9a-zA-Z]$', label)

def predict_crop(crop):
    """
    crop: grayscale image (numpy uint8)
    returns: (chosen_label, source, confidences dict)
    """
    # prepare both inputs
    keras_input = preprocess_for_digit_char(crop)  # shape (1,28,28,1)
    torch_tensor, viz_binary = preprocess_for_math(crop)  # tensor shape (1,1,H,W)

    # Keras predictions
    try:
        # digit model predict
        pred_digit = digit_model.predict(keras_input, verbose=0)
        idx_digit = int(np.argmax(pred_digit[0]))
        conf_digit = float(pred_digit[0][idx_digit])
        # character model predict
        pred_char = character_model.predict(keras_input, verbose=0)
        idx_char = int(np.argmax(pred_char[0]))
        conf_char = float(pred_char[0][idx_char])
        # decode char mapping
        char_label = new_char_mapping.get(idx_char, None)
    except Exception as e:
        idx_digit, conf_digit, idx_char, conf_char, char_label = None, 0.0, None, 0.0, None

    # PyTorch math model predict
    math_label, conf_math, math_index = None, 0.0, None
    if model_math is not None:
        try:
            with torch.no_grad():
                dev = math_device
                in_dev = torch_tensor.to(dev)
                out = model_math(in_dev)
                probs = torch.softmax(out, dim=1).cpu().numpy()[0]
                arg = int(np.argmax(probs))
                conf_math = float(probs[arg])
                math_index = arg
                math_label = class_name_map_math.get(arg, str(arg))
        except Exception as e:
            math_label, conf_math, math_index = None, 0.0, None

    # Decision logic:
    # - If math model returns a label that is non-alphanumeric, prefer it.
    # - Else if char/digit conf is higher than math_conf by margin, pick char/digit
    # - Attempt to disambiguate digits vs letters: choose digit if digit_conf > char_conf and digit_conf > 0.5
    chosen_label = None
    source = None
    confidences = {}

    # prepare candidate labels
    digit_label = str(idx_digit) if idx_digit is not None else None
    char_label_final = char_label if char_label is not None else None

    confidences['digit_conf'] = conf_digit
    confidences['char_conf'] = conf_char
    confidences['math_conf'] = conf_math

    # if math_label exists and is symbol-like (non-alnum) -> use it
    if math_label and is_symbol_label(math_label):
        chosen_label = math_label
        source = 'math_model'
    else:
        # prefer digit if digit_conf notably higher than char_conf
        if conf_digit >= conf_char and conf_digit > 0.35:
            chosen_label = digit_label
            source = 'digit_model'
        elif char_label_final and conf_char > 0.35:
            chosen_label = char_label_final
            source = 'char_model'
        else:
            # fallback: if math_label is alnum and math_conf strong, use it
            if math_label and conf_math > 0.5:
                chosen_label = math_label
                source = 'math_model'
            else:
                # as last resort: pick whichever has highest confidence
                best = max([('digit', conf_digit), ('char', conf_char), ('math', conf_math)], key=lambda x: x[1])
                if best[0] == 'digit' and idx_digit is not None:
                    chosen_label = digit_label; source = 'digit_model'
                elif best[0] == 'char' and char_label_final is not None:
                    chosen_label = char_label_final; source = 'char_model'
                elif math_label:
                    chosen_label = math_label; source = 'math_model'
                else:
                    chosen_label = '?' ; source = 'unknown'

    return chosen_label, source, confidences, viz_binary

# ---------------------------
# ---- Post-process & solve --
# ---------------------------

def build_expression_from_tokens(tokens):
    """
    tokens: list of (label, src)
    returns a string expression suitable for sympy parsing with replacements:
    - common math glyphs replaced to programming equivalents
    - e.g., '×' -> '*', '÷' -> '/', '—' or '−' -> '-'
    - If tokens include multi-char labels like 'sin', they should already be single token labels from math model
    """
    # join tokens with no separator; but add '*' for implicit multiplication like '2x' -> '2*x' (we'll post-process)
    raw = "".join([t for t, s in tokens])

    # replace common glyphs
    repl = {
        '×': '*',
        '✕': '*',
        '·': '*',
        '÷': '/',
        '−': '-',  # different minus signs
        '—': '-',
        '⁄': '/',
        '∗': '*',
        '×': '*',
        '√': 'sqrt',
        '∑': 'sum',
        '^': '**',
        '≤': '<=',
        '≥': '>='
    }
    for k,v in repl.items():
        raw = raw.replace(k, v)

    # Insert explicit multiplication between number and variable if missing: e.g. "2x" -> "2*x", or ")(" -> ")*("
    # Use regex to insert * between digit/letter and letter or '('
    raw = re.sub(r'(?<=\d)(?=[A-Za-z(])', '*', raw)
    raw = re.sub(r'(?<=[A-Za-z0-9\)])(?=\()', '*', raw)
    # also between letter and digit: 'x2' -> 'x*2'
    raw = re.sub(r'(?<=[A-Za-z])(?=\d)', '*', raw)

    # fix double operators
    raw = re.sub(r'\+\+', '+', raw)
    raw = re.sub(r'--', '-', raw)
    raw = raw.strip()
    return raw

def try_solve_expression(expr_str):
    """
    Try to evaluate or solve the recognized expression using sympy.
    Returns (result_text, success_flag)
    """
    expr_str = expr_str.strip()
    if expr_str == "":
        return "Empty expression", False
    try:
        # If it contains '=' treat as equation
        if '=' in expr_str:
            left, right = expr_str.split('=', 1)
            # attempt to parse both sides
            transformations = (standard_transformations + (implicit_multiplication_application,))
            left_parsed = parse_expr(left, transformations=transformations)
            right_parsed = parse_expr(right, transformations=transformations)
            eq = Eq(left_parsed, right_parsed)
            # try to solve for a variable if present, otherwise solve symbolically
            vars_in_eq = list(eq.free_symbols)
            if len(vars_in_eq) == 0:
                # no symbolic variable -> check boolean truth
                truth = bool(left_parsed.equals(right_parsed))
                return f"Equation truth: {truth}", True
            else:
                sol = solve(eq, vars_in_eq)
                return f"Solve result: {sol}", True
        else:
            # no '=', try simplify or evaluate numeric
            transformations = (standard_transformations + (implicit_multiplication_application,))
            parsed = parse_expr(expr_str, transformations=transformations)
            simplified = simplify(parsed)
            # If expression is numeric, evaluate
            if simplified.free_symbols:
                return f"Simplified: {simplified}", True
            else:
                val = simplified.evalf()
                return f"Value: {val}", True
    except Exception as e:
        return f"Sympy failed to parse/solve: {e}", False

# ---------------------------
# --------- Streamlit UI ----
# ---------------------------

st.markdown("Draw a whole expression (e.g. `2x+3=7`), or upload an image. The app will segment, recognize tokens, and attempt to solve or evaluate.")

mode = st.radio("Input mode", ("🖊️ Draw (Canvas)", "📁 Upload Image"))

img_for_processing = None
orig_display = None

if mode.startswith("🖊️"):
    canvas_result = st_canvas(fill_color="rgba(255,255,255,1)",
                              stroke_width=8,
                              stroke_color="#000000",
                              background_color="#FFFFFF",
                              height=400, width=800,
                              drawing_mode="freedraw", key="canvas")
    if canvas_result.image_data is not None:
        orig_display = canvas_result.image_data.astype(np.uint8)
        st.image(orig_display, caption="Canvas image", use_column_width=True)
        img_for_processing = orig_display
else:
    uploaded = st.file_uploader("Upload image", type=["png","jpg","jpeg"])
    if uploaded is not None:
        pil = Image.open(uploaded).convert("RGB")
        arr = np.array(pil)
        orig_display = arr
        st.image(arr, caption="Uploaded image", use_column_width=True)
        img_for_processing = arr

if img_for_processing is not None:
    col1, col2 = st.columns([1,1])
    with col1:
        if st.button("🔍 Segment & Recognize"):
            t0 = time.time()
            crops = segment_image_full(img_for_processing)
            tokens = []  # list of (label, source, confidence)
            viz_imgs = []
            if not crops:
                st.warning("No symbols detected. Try drawing thicker or ensure contrast.")
            else:
                for (x,y,w,h,crop) in crops:
                    label, source, confs, viz_binary = predict_crop(crop)
                    tokens.append((label, source, confs))
                    viz_imgs.append((x,y,w,h,label,source,confs,viz_binary))
                # Show tokens and visual crops
                st.subheader("Detected tokens (Left→Right)")
                cols = st.columns(min(4, len(viz_imgs)))
                for i, (x,y,w,h,label,source,confs,viz) in enumerate(viz_imgs):
                    with cols[i % len(cols)]:
                        st.image(viz, width=120, caption=f"{label} ({source})\n confs: {confs}")
                # Build expression
                token_labels = [(t[0], t[1]) for t in tokens]
                expr = build_expression_from_tokens(token_labels)
                st.success(f"Recognized expression: `{expr}`")
                result_text, ok = try_solve_expression(expr)
                if ok:
                    st.info(result_text)
                else:
                    st.warning(result_text)
            t1 = time.time()
            st.write(f"Processed in {(t1-t0):.3f}s")

    with col2:
        st.header("Manual token correction (optional)")
        st.markdown("If automatic segmentation/prediction got tokens wrong, you can correct them here and re-run the solver.")
        if st.button("✨ Auto-fill detected tokens into correction box"):
            crops = segment_image_full(img_for_processing)
            token_labels = []
            for (x,y,w,h,crop) in crops:
                label, source, confs, viz_binary = predict_crop(crop)
                token_labels.append(label)
            # create a joined expression for editing
            joined = "".join(token_labels)
            st.session_state['manual_expr'] = joined

        manual_expr = st.text_input("Edit the expression (after corrections)", value=st.session_state.get('manual_expr',''), key='manual_expr_input')
        if st.button("🧮 Evaluate / Solve edited expression"):
            if manual_expr.strip() == "":
                st.warning("Provide an expression to evaluate")
            else:
                expr = build_expression_from_tokens([(c,'manual') for c in list(manual_expr)])
                st.success(f"Using expression: `{expr}`")
                res, ok = try_solve_expression(expr)
                if ok:
                    st.info(res)
                else:
                    st.warning(res)

st.markdown("---")
st.markdown("Hints & Tips: Draw thicker strokes if segmentation misses small symbols. For fractions/exponents or complex two-line layout you'll need layout parsing (future enhancement).")

