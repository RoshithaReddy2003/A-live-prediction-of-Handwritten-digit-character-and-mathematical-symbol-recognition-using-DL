import torch
import torch.nn as nn
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
import os
import matplotlib.pyplot as plt

# Load dataset metadata
state = torch.load('improved_dataset_state.pth', weights_only=False)
label_map = state['label_map']
class_name_map = state['class_name_map']
num_classes = len(label_map)

# Model class (exact from train.py)
class OptimizedSymbolCNN(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.3):
        super(OptimizedSymbolCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2), nn.Dropout2d(0.1),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2), nn.Dropout2d(0.1),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2), nn.Dropout2d(0.2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2), nn.Dropout2d(0.2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
            nn.Linear(256, 512), nn.BatchNorm1d(512), nn.ReLU(inplace=True), nn.Dropout(dropout_rate),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True), nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None: nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01); nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
model = OptimizedSymbolCNN(num_classes, dropout_rate=0.3).to(device)
checkpoint = torch.load('best_model_optimized.pth', map_location=device, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print(f"Model loaded on {device} (Best Val Acc: {checkpoint['val_acc']:.2f}%)")

# Exact transform from training
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

def preprocess_exact(image_path, photo_mode=False):
    # Load as in training: Grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not load {image_path}")
    
    # Resize early
    image = cv2.resize(image, (45, 45))
    
    if photo_mode:
        # Light tweaks for photos: Auto-invert if needed + mild denoise
        if np.mean(image) > 127:  # Dark bg?
            image = cv2.bitwise_not(image)
        image = cv2.medianBlur(image, 3)  # Mild only
        kernel = np.ones((2, 2), np.uint8)
        image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    
    # Exact Otsu INV from training
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Exact enhancements
    binary = cv2.GaussianBlur(binary, (3, 3), 0)
    binary = cv2.convertScaleAbs(binary, alpha=1.2, beta=10)
    
    pil_img = Image.fromarray(binary.astype(np.uint8))
    return transform(pil_img), binary

def predict_symbol(img_tensor):
    with torch.no_grad():
        img_tensor = img_tensor.unsqueeze(0).to(device)
        output = model(img_tensor)
        pred_idx = output.argmax(1).item()
        confidence = torch.softmax(output, dim=1).max().item()
        pred_name = class_name_map[pred_idx]
        return pred_idx, pred_name, confidence

# Test on clean 'sin' from dataset
dataset_root = '/Users/mac/Documents/math recognizer and solver/Archieve1/extracted_images/sin'
sample_files = [f for f in os.listdir(dataset_root) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
if sample_files:
    sample_path = os.path.join(dataset_root, sample_files[0])  # First file
    print(f"Testing on clean sample: {sample_path}")
    processed, viz_img = preprocess_exact(sample_path, photo_mode=False)
    pred_idx, pred_name, conf = predict_symbol(processed)
    print(f"Clean Prediction: {pred_name} (conf: {conf:.3f})")  # Now 'sin' >0.99!
    
    # Quick viz for clean
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.imread(sample_path, cv2.IMREAD_GRAYSCALE), cmap='gray')
    plt.title('Original Dataset Image')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(viz_img, cmap='gray')
    plt.title(f'Processed (Pred: {pred_name}, Conf: {conf:.3f})')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('clean_sin_viz.png', dpi=150)
    plt.show()
    print("Clean viz saved as 'clean_sin_viz.png'")
else:
    print("No sin samples found—check path.")

# Now test handwritten (enter path)
handwritten_path = input("\nEnter path to handwritten single symbol (e.g., your_sin_photo.jpg): ").strip()
if os.path.exists(handwritten_path):
    photo_mode = input("Use photo tweaks? (y/n, default n): ").strip().lower() == 'y'
    processed, viz_img = preprocess_exact(handwritten_path, photo_mode=photo_mode)
    pred_idx, pred_name, conf = predict_symbol(processed)
    print(f"Handwritten Prediction: {pred_name} (conf: {conf:.3f})")
    
    # Viz
    original = cv2.imread(handwritten_path)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB) if len(original.shape)==3 else original, cmap='gray' if len(original.shape)==2 else None)
    ax1.set_title('Original Handwritten')
    ax1.axis('off')
    ax2.imshow(viz_img, cmap='gray')
    ax2.set_title(f'Processed (Pred: {pred_name}, Conf: {conf:.3f})')
    ax2.axis('off')
    plt.tight_layout()
    plt.savefig('handwritten_viz.png', dpi=150)
    plt.show()
    print("Handwritten viz saved as 'handwritten_viz.png'")
else:
    print("Path not found—double-check.")