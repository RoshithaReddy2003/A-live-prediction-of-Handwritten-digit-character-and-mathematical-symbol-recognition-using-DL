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

# --- Digits Model Setup ---
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

# Check for weights; if not found, train a simple model (uncomment to train once)
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
            label, ascii_val = map(int, line.split())
            mapping[label] = chr(ascii_val)
    return mapping

# Load EMNIST data (assume in ./data/)
train_images = load_emnist_images("/Users/mac/Documents/math recognizer and solver/colab_weights/emnist-bymerge-train-images-idx3-ubyte.gz")
train_labels = load_emnist_labels("/Users/mac/Documents/math recognizer and solver/colab_weights/emnist-bymerge-train-labels-idx1-ubyte.gz")
test_images = load_emnist_images("/Users/mac/Documents/math recognizer and solver/colab_weights/emnist-bymerge-test-images-idx3-ubyte.gz")
test_labels = load_emnist_labels("/Users/mac/Documents/math recognizer and solver/colab_weights/emnist-bymerge-test-labels-idx1-ubyte.gz")
mapping = load_emnist_mapping("/Users/mac/Documents/math recognizer and solver/colab_weights/emnist-bymerge-mapping (1).txt")

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

# Check for weights; if not found, train a simple model (uncomment to train once)
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

# --- Global Variables for Custom Datasets ---
digit_custom_images, digit_custom_labels, digit_custom_features = np.array([]), np.array([]), np.array([])
char_custom_images, char_custom_labels, char_custom_features = np.array([]), np.array([]), np.array([])

# --- Custom Dataset Management ---
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

# --- Image Preprocessing ---
def load_image(data):
    try:
        img_data = b64decode(data.split(',')[1])
        img_array = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("Image decoding failed")
        print(f"Raw image shape: {img.shape}, mean: {np.mean(img)}")
        kernel = np.ones((2, 2), np.uint8)
        img = cv2.erode(img, kernel, iterations=1)
        img = cv2.resize(img, (28, 28))
        _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if np.mean(img) > 127:
            img = 255 - img
        print(f"Processed image mean: {np.mean(img)}")
        img = img.astype('float32') / 255.0
        return img.reshape(1, 28, 28, 1)
    except Exception as e:
        print(f"Error in load_image: {e}")
        return None

# --- Prediction and Correction Logic ---
def predict(data_url, mode):
    img = load_image(data_url)
    if img is None:
        print(f"Predict failed: Invalid image data for mode {mode}")
        return {"prediction": None, "confidence": None, "source": None, "error": "Image processing failed"}
    start_time = time.time()
    is_digit = mode == 'digits'
    model = digit_model if is_digit else character_model
    feature_model = digit_feature_model if is_digit else character_feature_model
    custom_features = digit_custom_features if is_digit else char_custom_features
    custom_labels = digit_custom_labels if is_digit else char_custom_labels
    try:
        pred_value = model.predict(img, verbose=0)
        pred_idx = argmax(pred_value)
        model_confidence = min(pred_value[0][pred_idx], 0.99)
        custom_pred, custom_confidence = check_custom_match(img[0], custom_features, custom_labels, mode)
        if custom_pred is None or custom_confidence + 0.15 < model_confidence:
            pred = pred_idx if is_digit else new_mapping[pred_idx]
            confidence = model_confidence
            source = digit_weights_file if is_digit else character_weights_file
        else:
            pred = custom_pred
            confidence = min(0.99, custom_confidence + 0.15)
            source = "Custom Dataset"
    except Exception as e:
        print(f"Error in predict for mode {mode}: {e}")
        return {"prediction": None, "confidence": None, "source": None, "error": str(e)}
    pred_time = time.time() - start_time
    img_data = (img[0] * 255).astype(np.uint8)
    _, img_encoded = cv2.imencode('.png', img_data)
    img_base64 = b64encode(img_encoded.tobytes()).decode('utf-8')
    return {
        "prediction": str(pred),
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
    img = load_image(data_url)
    if img is None:
        return {"status": "error", "message": "Image processing failed"}
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
    os.makedirs('./data', exist_ok=True)
    save_to_custom_dataset(img[0], true_label, filename, is_digit)
    return predict(data_url, mode)

# --- Load existing custom datasets ---
load_custom_dataset("/Users/mac/Documents/math recognizer and solver/colab_weights/custom_digits.h5", is_digit=True)
load_custom_dataset("/Users/mac/Documents/math recognizer and solver/colab_weights/custom_characters.h5", is_digit=False)

# --- Find an Available Port ---
def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]

port = find_free_port()
print(f"Using free port: {port}")

# --- Custom HTTP Handler ---
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
            # Serve files from static/ directory
            filename = self.path[8:]  # Remove '/static/' prefix
            static_path = os.path.join('static', filename)
            print(f"GET request to /static/{filename}")
            if os.path.exists(static_path):
                try:
                    with open(static_path, 'rb') as f:
                        self.send_response(200)
                        if filename.endswith('.jpg') or filename.endswith('.jpeg'):
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
            # Ignore favicon requests
            self.send_response(204)
            self.end_headers()
            return
        else:
            print(f"GET request to {self.path} - 404")
            self.send_error(404)

    def do_POST(self):
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

# --- Create Enhanced HTML File ---
html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Handwritten Recognition</title>
    <style>
        body {
            font-family: 'Segoe UI', Arial, sans-serif;
            background-image: url('static/background_image.jpg');
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            margin: 0;
            padding: 20px;
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
        }

        .start-screen {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background-image: url('static/background_image.jpg');
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            z-index: 1000;
            transition: opacity 0.5s ease;
        }

        .start-screen h1 {
            color: white;
            text-shadow: 0 2px 4px rgba(0, 0, 0, 0.5);
            font-size: 2.5em;
            margin-bottom: 20px;
        }

        .container {
            background: rgba(255, 255, 255, 0.95);
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
            max-width: 600px;
            width: 100%;
            display: none;
            flex-direction: column;
            align-items: center;
            box-sizing: border-box;
        }

        .container.active {
            display: flex;
        }

        h2 {
            margin: 10px 0;
            color: #333;
            font-size: 1.8em;
        }

        .mode-buttons {
            display: flex;
            justify-content: center;
            gap: 10px;
            margin-bottom: 10px;
        }

        .action-buttons {
            display: flex;
            justify-content: center;
            gap: 12px;
            margin-top: 10px;
            align-items: center;
            flex-wrap: wrap;
        }

        canvas {
            border: 2px solid #e5e7eb;
            border-radius: 10px;
            background: white;
            box-shadow: inset 0 2px 10px rgba(0, 0, 0, 0.05);
            margin: 10px 0;
        }

        button {
            padding: 14px 30px;
            font-size: 18px;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.3s ease;
            background: linear-gradient(135deg,
                rgba(109, 122, 253, 0.8),
                rgba(76, 175, 80, 0.8));
            color: white;
            font-weight: 500;
            position: relative;
            overflow: hidden;
        }

        button:hover {
            background: linear-gradient(135deg,
                rgba(90, 103, 216, 0.8),
                rgba(61, 139, 64, 0.8));
        }

        button.active {
            background: linear-gradient(135deg,
                rgba(124, 179, 66, 0.9),
                rgba(0, 153, 255, 0.9));
            box-shadow: 0 0 15px rgba(0, 153, 255, 0.5);
            transform: scale(1.05);
        }

        button::before {
            content: '';
            position: absolute;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: linear-gradient(45deg,
                transparent,
                rgba(255, 255, 255, 0.3),
                transparent);
            transform: rotate(45deg);
            animation: hologram 3s infinite linear;
        }

        @keyframes hologram {
            0% { transform: translateX(-100%) rotate(45deg); }
            100% { transform: translateX(100%) rotate(45deg); }
        }

        #startBtn, #digitBtn, #charBtn, #predictBtn, #clearBtn, #correctBtn {
            background: linear-gradient(135deg,
                rgba(109, 122, 253, 0.8),
                rgba(76, 175, 80, 0.8));
        }

        #startBtn:hover, #digitBtn:hover, #charBtn:hover, #predictBtn:hover, #clearBtn:hover, #correctBtn:hover {
            background: linear-gradient(135deg,
                rgba(90, 103, 216, 0.8),
                rgba(61, 139, 64, 0.8));
        }

        #correctBtn {
            display: none;
        }

        #prediction {
            margin-top: 15px;
            padding: 10px;
            border-radius: 10px;
            background: #f9fafb;
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.05);
            max-width: 500px;
            width: 100%;
            position: relative;
        }

        .quantum-loader {
            width: 30px;
            height: 30px;
            border-radius: 50%;
            background: conic-gradient(#6d7afd, #4caf50, #6d7afd);
            animation: spin 1.5s linear infinite;
            margin: 10px auto;
            display: none;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="start-screen">
        <h1>Welcome To Handwritten Recognition</h1>
        <button id="startBtn">Start</button>
    </div>

    <div class="container">
        <h2>Handwritten Recognition</h2>

        <!-- Mode selection above canvas -->
        <div class="mode-buttons">
            <button id="digitBtn">Digits</button>
            <button id="charBtn">Characters</button>
        </div>

        <canvas id="canvas" width="280" height="280"></canvas>

        <!-- Action buttons below canvas -->
        <div class="action-buttons">
            <button id="predictBtn">Predict</button>
            <button id="clearBtn">Clear</button>
            <button id="correctBtn">Correct</button>
        </div>

        <div id="prediction">
            <div class="quantum-loader" id="processingLoader"></div>
        </div>
    </div>

    <script>
        const canvas = document.getElementById('canvas');
        const ctx = canvas.getContext('2d');
        let drawing = false;
        let mode = 'digits';
        let drawingEnabled = false;

        // Initialize canvas
        function initCanvas() {
            ctx.fillStyle = 'white';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            ctx.lineWidth = 8;
            ctx.strokeStyle = 'black';
            ctx.lineCap = 'round';
            ctx.lineJoin = 'round';
        }

        // Start screen handler
        document.getElementById('startBtn').addEventListener('click', () => {
            console.log('Start button clicked');
            document.querySelector('.start-screen').style.opacity = '0';
            setTimeout(() => {
                document.querySelector('.start-screen').style.display = 'none';
                document.querySelector('.container').classList.add('active');
                initCanvas();
                enableDrawing();
            }, 500);
        });

        // Initialize Correct button as hidden
        document.getElementById('correctBtn').style.display = 'none';

        // Mode button handlers
        document.getElementById('digitBtn').addEventListener('click', () => {
            mode = 'digits';
            document.getElementById('digitBtn').classList.add('active');
            document.getElementById('charBtn').classList.remove('active');
            quantumReset();
        });

        document.getElementById('charBtn').addEventListener('click', () => {
            mode = 'characters';
            document.getElementById('charBtn').classList.add('active');
            document.getElementById('digitBtn').classList.remove('active');
            quantumReset();
        });

        // Set initial active mode
        document.getElementById('digitBtn').click();

        // Drawing logic
        function enableDrawing() {
            canvas.style.cursor = 'crosshair';
            drawingEnabled = true;

            canvas.addEventListener('mousedown', startDrawing);
            canvas.addEventListener('mousemove', draw);
            canvas.addEventListener('mouseup', stopDrawing);
            canvas.addEventListener('mouseout', stopDrawing);

            canvas.addEventListener('touchstart', startDrawing);
            canvas.addEventListener('touchmove', draw);
            canvas.addEventListener('touchend', stopDrawing);
        }

        function startDrawing(e) {
            if (!drawingEnabled) return;
            drawing = true;
            const pos = getCanvasPosition(e);
            ctx.beginPath();
            ctx.moveTo(pos.x, pos.y);
            console.log(`Started drawing at (${pos.x}, ${pos.y})`);
            e.preventDefault();
        }

        function draw(e) {
            if (!drawing) return;
            const pos = getCanvasPosition(e);
            ctx.lineTo(pos.x, pos.y);
            ctx.stroke();
            console.log(`Drawing to (${pos.x}, ${pos.y})`);
            e.preventDefault();
        }

        function stopDrawing() {
            if (drawing) {
                ctx.closePath();
                drawing = false;
                console.log('Stopped drawing');
            }
        }

        function getCanvasPosition(e) {
            const rect = canvas.getBoundingClientRect();
            const scaleX = canvas.width / rect.width;
            const scaleY = canvas.height / rect.height;
            const clientX = e.clientX || (e.touches && e.touches[0].clientX);
            const clientY = e.clientY || (e.touches && e.touches[0].clientY);
            return {
                x: (clientX - rect.left) * scaleX,
                y: (clientY - rect.top) * scaleY
            };
        }

        // Predict button handler
        document.getElementById('predictBtn').addEventListener('click', async () => {
            if (!drawingEnabled) return;
            const loader = document.getElementById('processingLoader');
            if (loader) loader.style.display = 'block';
            try {
                const dataURL = canvas.toDataURL('image/png');
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ data_url: dataURL, mode })
                });
                const data = await response.json();
                if (data.error) {
                    showHolographicError(new Error(data.error));
                } else {
                    displayQuantumPrediction(data);
                }
            } catch (error) {
                showHolographicError(error);
            } finally {
                if (loader) loader.style.display = 'none';
            }
        });

        // Correct button handler with pop-up
        document.getElementById('correctBtn').addEventListener('click', async () => {
            if (!drawingEnabled) return;
            const promptText = mode === 'digits' ? 'Enter the correct digit (0-9):' : 'Enter the correct character (A-Z, a-z):';
            const trueLabel = prompt(promptText);
            if (trueLabel === null || trueLabel.trim() === '') {
                showHolographicError(new Error('No correction label provided'));
                return;
            }
            const loader = document.getElementById('processingLoader');
            if (loader) loader.style.display = 'block';
            try {
                const dataURL = canvas.toDataURL('image/png');
                const response = await fetch('/correct', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ data_url: dataURL, true_label: trueLabel.trim(), mode })
                });
                const data = await response.json();
                if (data.status === 'error') {
                    showHolographicError(new Error(data.message));
                } else {
                    displayQuantumPrediction(data);
                }
            } catch (error) {
                showHolographicError(error);
            } finally {
                if (loader) loader.style.display = 'none';
            }
        });

        // Clear button handler
        document.getElementById('clearBtn').addEventListener('click', quantumReset);

        function quantumReset() {
            ctx.fillStyle = 'white';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            ctx.lineWidth = 8;
            ctx.strokeStyle = 'black';
            ctx.lineCap = 'round';
            ctx.lineJoin = 'round';
            document.getElementById('prediction').innerHTML = '<div class="quantum-loader" id="processingLoader"></div>';
            document.getElementById('correctBtn').style.display = 'none';
        }

        function displayQuantumPrediction(data) {
            const predictionHTML = `
                <div class="quantum-loader" id="processingLoader"></div>
                <div class="result">
                    <img src="data:image/png;base64,${data.image}" alt="Prediction">
                    <div>
                        <p><strong>Predicted Value:</strong> ${data.prediction}</p>
                        <p><strong>Confidence:</strong> ${(data.confidence * 100).toFixed(2)}%</p>
                        <p><strong>Processing Time:</strong> ${data.time.toFixed(3)}s</p>
                        <p><strong>Source:</strong> ${data.source}</p>
                    </div>
                </div>`;
            document.getElementById('prediction').innerHTML = predictionHTML;
            document.getElementById('correctBtn').style.display = 'inline-block';
            document.getElementById('processingLoader').style.display = 'none';
        }

        function showHolographicError(error) {
            console.error('Error:', error);
            document.getElementById('prediction').innerHTML = `
                <div class="quantum-loader" id="processingLoader"></div>
                <div class="result">
                    <p><strong>Error:</strong> Unable to process: ${error.message}</p>
                </div>`;
            document.getElementById('correctBtn').style.display = 'none';
            document.getElementById('processingLoader').style.display = 'none';
        }
    </script>
</body>
</html>
"""

with open('index.html', 'w') as f:
    f.write(html_content)

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