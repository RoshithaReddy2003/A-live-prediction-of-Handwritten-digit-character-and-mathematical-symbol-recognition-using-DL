import cv2
import numpy as np
from numpy import argmax
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization, GlobalAveragePooling2D, Add
from keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from base64 import b64decode, b64encode
import h5py
import os
import time
import gzip
import struct
from http.server import HTTPServer, SimpleHTTPRequestHandler
import threading
import json
import socket
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from collections import Counter
import re
import pickle

# ===============================
#  Load Digit & Character Models (Keras/TF)
# ===============================
(trainX, trainY), (testX, testY) = mnist.load_data()
trainX = trainX.reshape((trainX.shape[0], 28, 28, 1)).astype('float32') / 255.0
testX = testX.reshape((testX.shape[0], 28, 28, 1)).astype('float32') / 255.0
trainY = to_categorical(trainY, 10)
testY = to_categorical(testY, 10)

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

digit_model = build_digit_model()
digit_model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Check for weights; if not found, train a simple model
digit_weights_file = '/Users/mac/Documents/math recognizer and solver/colab_weights/improved1.weights.h5'
if os.path.exists(digit_weights_file):
    digit_model.load_weights(digit_weights_file)
    print(f"Loaded digit weights from {digit_weights_file}")
else:
    print(f"{digit_weights_file} not found. Training a quick model...")
    digit_model.fit(trainX, trainY, epochs=3, batch_size=128, validation_data=(testX, testY), verbose=1)
    os.makedirs('./data', exist_ok=True)
    digit_model.save_weights(digit_weights_file)
    print(f"Trained and saved digit weights to {digit_weights_file}")

digit_feature_model = Model(inputs=digit_model.input, outputs=digit_model.layers[-3].output)

# --- Characters Model Setup ---
def load_emnist_images(filename):
    with gzip.open(filename, 'rb') as f:
        _, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, rows, cols)
    return images

def load_emnist_labels(filename):
    with gzip.open(filename, 'rb') as f:
        _, num_labels = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels

def load_emnist_mapping(filename):
    mapping = {}
    with open(filename, 'r') as f:
        for line in f:
            try:
                label, ascii_val = map(int, line.split())
                mapping[label] = chr(ascii_val)
            except ValueError:
                continue  # Skip malformed lines
    return mapping

# Load EMNIST from existing paths (no download needed)
emnist_base = '/Users/mac/Documents/math recognizer and solver/colab_weights/'
train_images = load_emnist_images(f"{emnist_base}emnist-bymerge-train-images-idx3-ubyte.gz")
train_labels = load_emnist_labels(f"{emnist_base}emnist-bymerge-train-labels-idx1-ubyte.gz")
test_images = load_emnist_images(f"{emnist_base}emnist-bymerge-test-images-idx3-ubyte.gz")
test_labels = load_emnist_labels(f"{emnist_base}emnist-bymerge-test-labels-idx1-ubyte.gz")
mapping = load_emnist_mapping(f"{emnist_base}emnist-bymerge-mapping (1).txt")
print("Loaded EMNIST files from provided paths.")

character_indices_train = train_labels >= 10
character_indices_test = test_labels >= 10
train_images = train_images[character_indices_train]
train_labels = train_labels[character_indices_train]
test_images = test_images[character_indices_test]
test_labels = test_labels[character_indices_test]
train_labels = train_labels - 10
test_labels = test_labels - 10
new_mapping = {i: mapping[i + 10] for i in range(37)}

train_images = train_images.reshape((train_images.shape[0], 28, 28, 1)).astype('float32') / 255.0
test_images = test_images.reshape((test_images.shape[0], 28, 28, 1)).astype('float32') / 255.0
train_images = np.array([np.fliplr(np.rot90(img, k=-1)) for img in train_images])
test_images = np.array([np.fliplr(np.rot90(img, k=-1)) for img in test_images])
train_labels = to_categorical(train_labels, 37)
test_labels = to_categorical(test_labels, 37)

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

character_model = build_character_model()
character_model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Check for weights; if not found, train a simple model
character_weights_file = '/Users/mac/Documents/math recognizer and solver/colab_weights/checkpoint.weights.h5'
if os.path.exists(character_weights_file):
    character_model.load_weights(character_weights_file)
    print(f"Loaded character weights from {character_weights_file}")
else:
    print(f"{character_weights_file} not found. Training a quick model...")
    character_model.fit(train_images, train_labels, epochs=3, batch_size=128, validation_data=(test_images, test_labels), verbose=1)
    os.makedirs('./data', exist_ok=True)
    character_model.save_weights(character_weights_file)
    print(f"Trained and saved character weights to {character_weights_file}")

character_feature_model = Model(inputs=character_model.input, outputs=character_model.layers[-3].output)

# ===============================
#  Load Math Symbol Model (PyTorch) - FIXED
# ===============================
@torch.no_grad()
def load_math_model():
    try:
        # Add safe globals for TensorDataset
        from torch.utils.data import TensorDataset
        import torch.serialization
        torch.serialization.add_safe_globals([TensorDataset])

        # Use absolute paths
        state_path = "/Users/mac/Documents/math recognizer and solver/improved_dataset_state.pth"
        checkpoint_path = "/Users/mac/Documents/math recognizer and solver/best_model_optimized.pth"  # Adjust if different

        state_math = torch.load(state_path, weights_only=False)
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
        checkpoint_math = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model_math.load_state_dict(checkpoint_math["model_state_dict"])
        model_math.eval()

        return model_math, device, class_name_map_math, checkpoint_math
    except Exception as e:
        print(f"Error while loading math model: {e}")
        return None, None, {}, {}

model_math, math_device, class_name_map_math, checkpoint_math = load_math_model()
if model_math:
    print(f"Math Model on {math_device} (Val Acc: {checkpoint_math.get('val_acc', 'N/A'):.2f}%)")
else:
    print("Math model failed to load.")

# Math-specific transform
math_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# ===============================
#  Custom Datasets (Unified: HDF5 for digits/chars, Pickle for math)
# ===============================
digit_custom_images, digit_custom_labels, digit_custom_features = np.array([]), np.array([]), np.array([])
char_custom_images, char_custom_labels, char_custom_features = np.array([]), np.array([]), np.array([])

MATH_CUSTOM_FILE = "/Users/mac/Documents/math recognizer and solver/colab_weights/custom_symbols.pkl"

def load_math_custom():
    if os.path.exists(MATH_CUSTOM_FILE):
        with open(MATH_CUSTOM_FILE, 'rb') as f:
            return pickle.load(f)
    return []

def save_to_math_custom(binary_img, label):
    try:
        custom = load_math_custom()
        custom.append((label, binary_img))
        with open(MATH_CUSTOM_FILE, 'wb') as f:
            pickle.dump(custom, f)
        return True
    except Exception as e:
        print(f"Error saving to math custom dataset: {e}")
        return False

def save_to_custom_dataset(image, label, filename, is_digit=True):
    global digit_custom_images, digit_custom_labels, digit_custom_features, char_custom_images, char_custom_labels, char_custom_features
    if len(image.shape) == 2:
        image = image.reshape(28, 28, 1)
    if image.max() > 1.0:
        image = image.astype('float32') / 255.0
    features = digit_feature_model if is_digit else character_feature_model
    custom_images = digit_custom_images if is_digit else char_custom_images
    custom_labels = digit_custom_labels if is_digit else char_custom_labels
    custom_features = digit_custom_features if is_digit else char_custom_features
    try:
        if os.path.exists(filename):
            with h5py.File(filename, "r") as f:
                current_size = f["images"].shape[0]
                print(f"Before appending, {filename} has {current_size} entries")
        else:
            current_size = 0
            print(f"{filename} does not exist yet, creating new file")

        if not os.path.exists(filename):
            with h5py.File(filename, "w") as f:
                f.create_dataset("images", data=[image], maxshape=(None, 28, 28, 1), dtype='float32')
                dtype = 'int32' if is_digit else h5py.string_dtype(encoding='utf-8')
                f.create_dataset("labels", data=[label], maxshape=(None,), dtype=dtype)
        else:
            with h5py.File(filename, "a") as f:
                f["images"].resize((f["images"].shape[0] + 1), axis=0)
                f["labels"].resize((f["labels"].shape[0] + 1), axis=0)
                f["images"][-1] = image
                f["labels"][-1] = label

        with h5py.File(filename, "r") as f:
            new_size = f["images"].shape[0]
            print(f"After appending, {filename} has {new_size} entries")

        custom_images = np.append(custom_images, [image], axis=0) if custom_images.size else np.array([image])
        custom_labels = np.append(custom_labels, [label], axis=0) if custom_labels.size else np.array([label])
        new_feature = features.predict(image.reshape(1, 28, 28, 1), verbose=0)
        custom_features = np.append(custom_features, new_feature, axis=0) if custom_features.size else new_feature
        if is_digit:
            digit_custom_images, digit_custom_labels, digit_custom_features = custom_images, custom_labels, custom_features
        else:
            char_custom_images, char_custom_labels, char_custom_features = custom_images, custom_labels, custom_features
        print(f"Added to {filename}: {label}")
    except Exception as e:
        print(f"Error saving to custom dataset: {e}")

def load_custom_dataset(filename, is_digit=True):
    global digit_custom_images, digit_custom_labels, digit_custom_features, char_custom_images, char_custom_labels, char_custom_features
    features = digit_feature_model if is_digit else character_feature_model
    if not os.path.exists(filename):
        print(f"{filename} not found.")
        return np.array([]), np.array([])
    with h5py.File(filename, "r") as f:
        images = f["images"][:]
        labels = f["labels"][:]
        if not is_digit:
            labels = np.array([label.decode('utf-8') if isinstance(label, bytes) else label for label in labels], dtype=object)
    if len(images) > 0:
        custom_features = features.predict(images, verbose=0)
        if is_digit:
            digit_custom_images, digit_custom_labels, digit_custom_features = images, labels, custom_features
        else:
            char_custom_images, char_custom_labels, char_custom_features = images, labels, custom_features
        print(f"Loaded {len(images)} items from {filename}")
    return images, labels

# Load existing custom datasets
load_custom_dataset("/Users/mac/Documents/math recognizer and solver/colab_weights/custom_digits.h5", is_digit=True)
load_custom_dataset("/Users/mac/Documents/math recognizer and solver/colab_weights/custom_characters.h5", is_digit=False)

# ===============================
#  Preprocessing (Unified with Math-specific)
# ===============================
def load_image(data, mode='digits'):
    try:
        img_data = b64decode(data.split(',')[1])
        img_array = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("Image decoding failed")
        print(f"Raw image shape: {img.shape}, mean: {np.mean(img)}")
        kernel = np.ones((2, 2), np.uint8)
        img = cv2.erode(img, kernel, iterations=1)
        
        if mode == 'math_symbols':
            # Math-specific preprocessing (45x45, aspect ratio, etc.)
            if np.mean(img) < 127:
                img = cv2.bitwise_not(img)
            _, binary = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)
            coords = cv2.findNonZero(255 - binary)
            if coords is not None:
                x, y, w, h = cv2.boundingRect(coords)
                binary = binary[y:y+h, x:x+w]
            else:
                binary = np.ones((45, 45), np.uint8) * 255
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
            binary = cv2.bitwise_not(binary)
            binary = cv2.GaussianBlur(binary, (3, 3), 0)
            pil_img = Image.fromarray(binary.astype(np.uint8))
            processed_tensor = math_transform(pil_img).unsqueeze(0)
            viz_img = binary  # For visualization
            size = (45, 45)
        else:
            # Original for digits/chars (28x28)
            img = cv2.resize(img, (28, 28))
            _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            if np.mean(img) > 127:
                img = 255 - img
            print(f"Processed image mean: {np.mean(img)}")
            img = img.astype('float32') / 255.0
            processed_tensor = img.reshape(1, 28, 28, 1)
            viz_img = (img * 255).astype(np.uint8).reshape(28, 28)
            size = (28, 28)

        return processed_tensor, viz_img, size
    except Exception as e:
        print(f"Error in load_image: {e}")
        return None, None, (28, 28)

# ===============================
#  Prediction Logic (Extended for Math)
# ===============================
def predict(data_url, mode):
    img_tensor, viz_img, img_size = load_image(data_url, mode)
    if img_tensor is None:
        print(f"Predict failed: Invalid image data for mode {mode}")
        return {"prediction": None, "confidence": None, "source": None, "error": "Image processing failed", "image": None}
    start_time = time.time()
    
    if mode == 'digits':
        model = digit_model
        feature_model = digit_feature_model
        custom_features = digit_custom_features
        custom_labels = digit_custom_labels
        pred_value = model.predict(img_tensor, verbose=0)
        pred_idx = argmax(pred_value)
        model_confidence = min(pred_value[0][pred_idx], 0.99)
        custom_pred, custom_confidence = check_custom_match(img_tensor[0], custom_features, custom_labels, mode)
        if custom_pred is None or custom_confidence + 0.15 < model_confidence:
            pred = pred_idx
            confidence = model_confidence
            source = digit_weights_file
        else:
            pred = custom_pred
            confidence = min(0.99, custom_confidence + 0.15)
            source = "Custom Dataset"
        pred_label = str(pred)
        
    elif mode == 'characters':
        model = character_model
        feature_model = character_feature_model
        custom_features = char_custom_features
        custom_labels = char_custom_labels
        pred_value = model.predict(img_tensor, verbose=0)
        pred_idx = argmax(pred_value)
        model_confidence = min(pred_value[0][pred_idx], 0.99)
        custom_pred, custom_confidence = check_custom_match(img_tensor[0], custom_features, custom_labels, mode)
        if custom_pred is None or custom_confidence + 0.15 < model_confidence:
            pred = new_mapping[pred_idx]
            confidence = model_confidence
            source = character_weights_file
        else:
            pred = custom_pred
            confidence = min(0.99, custom_confidence + 0.15)
            source = "Custom Dataset"
        pred_label = str(pred)
        
    elif mode == 'math_symbols':
        if model_math is None:
            return {"prediction": None, "confidence": None, "source": None, "error": "Math model not loaded", "image": None}
        def is_symbol(label):
            return not re.match(r'^[0-9a-zA-Z]$', label)
        symbol_label_indices = [i for i, lbl in class_name_map_math.items() if is_symbol(lbl)]
        math_preds, math_confs = [], []
        for _ in range(5):  # Ensemble
            outputs_math = model_math(img_tensor.to(math_device))
            probs_math = torch.softmax(outputs_math, dim=1)
            symbol_probs = probs_math[0][symbol_label_indices]
            pred_relative = symbol_probs.argmax().item()
            pred_abs = symbol_label_indices[pred_relative]
            conf_math = symbol_probs[pred_relative].item()
            math_preds.append(pred_abs)
            math_confs.append(conf_math)
        final_math = Counter(math_preds).most_common(1)[0][0]
        final_conf_math = np.mean([c for p, c in zip(math_preds, math_confs) if p == final_math])
        pred_label = class_name_map_math[final_math]
        source = "math_model"
        
        # Check math custom dataset
        custom = load_math_custom()
        if custom and viz_img is not None:
            flat_query = viz_img.flatten().astype(np.float32)
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
                    pred_label = custom_label
                    final_conf_math = max_sim
                    source = "custom"
        confidence = final_conf_math
        pred_label = str(pred_label)
        
    else:
        return {"prediction": None, "confidence": None, "source": None, "error": "Invalid mode", "image": None}
    
    pred_time = time.time() - start_time
    _, img_encoded = cv2.imencode('.png', viz_img)
    img_base64 = b64encode(img_encoded.tobytes()).decode('utf-8')
    return {
        "prediction": pred_label,
        "confidence": float(confidence),
        "source": source,
        "time": pred_time,
        "image": img_base64
    }

def check_custom_match(input_image, custom_features, custom_labels, mode, threshold=0.75):
    if len(custom_features) == 0:
        return None, 0.0
    try:
        feature_model = digit_feature_model if mode == 'digits' else character_feature_model
        input_features = feature_model.predict(input_image.reshape(1, 28, 28, 1), verbose=0)
        best_similarity = -1.0
        best_label = None
        best_confidence = 0.0
        for i in range(len(custom_features)):
            similarity = np.dot(input_features.flatten(), custom_features[i].flatten()) / (np.linalg.norm(input_features) * np.linalg.norm(custom_features[i]))
            if similarity > threshold and similarity > best_similarity:
                best_similarity = similarity
                best_label = custom_labels[i]
                best_confidence = similarity
        return best_label, best_confidence
    except Exception as e:
        print(f"Error in check_custom_match: {e}")
        return None, 0.0

def correct(data_url, true_label, mode):
    img_tensor, viz_img, _ = load_image(data_url, mode)
    if img_tensor is None:
        return {"status": "error", "message": "Image processing failed"}
    
    if mode == 'math_symbols':
        if model_math is None:
            return {"status": "error", "message": "Math model not loaded"}
        saved = save_to_math_custom(viz_img, true_label.strip())
        if not saved:
            return {"status": "error", "message": "Failed to save to custom dataset"}
    else:
        is_digit = mode == 'digits'
        if is_digit:
            try:
                true_label = int(true_label)
                if not (0 <= true_label <= 9):
                    return {"status": "error", "message": "Invalid digit. Enter 0-9."}
            except ValueError:
                return {"status": "error", "message": "Invalid input. Enter a digit (0-9)."}
        else:
            if not true_label.isalpha() or len(true_label) != 1:
                return {"status": "error", "message": "Invalid character. Enter a single letter (A-Z, a-z)."}
        filename = "/Users/mac/Documents/math recognizer and solver/colab_weights/custom_digits.h5" if is_digit else "/Users/mac/Documents/math recognizer and solver/colab_weights/custom_characters.h5"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        save_to_custom_dataset(img_tensor[0], true_label, filename, is_digit)
    
    return predict(data_url, mode)

def get_custom_stats():
    digits_count = len(digit_custom_labels) if len(digit_custom_labels) > 0 else 0
    chars_count = len(char_custom_labels) if len(char_custom_labels) > 0 else 0
    math_count = len(load_math_custom())
    return {
        "digits": digits_count,
        "chars": chars_count,
        "math": math_count
    }

# ===============================
#  Server Setup (Extended)
# ===============================
def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]

port = find_free_port()
print(f"Using free port: {port}")

class CustomHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            with open('index.html', 'rb') as f:
                self.wfile.write(f.read())
        elif self.path.startswith('/static/'):
            filename = self.path[8:]
            static_path = os.path.join('static', filename)
            print(f"GET request to /static/{filename}")
            if os.path.exists(static_path):
                try:
                    with open(static_path, 'rb') as f:
                        self.send_response(200)
                        if filename.endswith(('.jpg', '.jpeg')):
                            self.send_header('Content-type', 'image/jpeg')
                        elif filename.endswith('.png'):
                            self.send_header('Content-type', 'image/png')
                        else:
                            self.send_header('Content-type', 'application/octet-stream')
                        self.end_headers()
                        self.wfile.write(f.read())
                        return
                except Exception as e:
                    print(f"Error serving static file {static_path}: {e}")
                    self.send_error(500)
            else:
                print(f"Static file not found: {static_path}")
            self.send_error(404)
        elif self.path == '/favicon.ico':
            self.send_response(204)
            self.end_headers()
            return
        else:
            print(f"GET request to {self.path} - 404")
            self.send_error(404)

    def do_POST(self):
        if self.path == '/stats':
            stats = get_custom_stats()
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(stats).encode('utf-8'))
            return

        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length).decode('utf-8')
        print(f"POST request to {self.path} with data: {post_data}")
        data = json.loads(post_data)

        if self.path == '/predict':
            result = predict(data['data_url'], data['mode'])
            print(f"Predict result: {result}")
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(result).encode('utf-8'))
        elif self.path == '/correct':
            result = correct(data['data_url'], data['true_label'], data['mode'])
            print(f"Correct result: {result}")
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(result).encode('utf-8'))
        else:
            print(f"POST request to {self.path} - 404")
            self.send_error(404)

# Print loading status for debugging
print("digit_model loaded:", digit_model is not None)
print("character_model loaded:", character_model is not None)
print("math_model loaded:", model_math is not None)
print("math_class_map size:", len(class_name_map_math))

# Deploy locally
server = HTTPServer(('0.0.0.0', port), CustomHandler)

def run_server():
    try:
        print(f"Starting server at http://0.0.0.0:{port}")
        server.serve_forever()
    except Exception as e:
        print(f"Server failed to start: {e}")

thread = threading.Thread(target=run_server)
thread.daemon = True
thread.start()

print(f"Access the web interface at: http://localhost:{port}")
input("Press Enter to stop the server...")