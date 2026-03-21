import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from collections import Counter
import pickle  # For safer saving

# Dataset Setup
data_root = '/Users/mac/Documents/math recognizer and solver/Archieve1/extracted_images'
if not os.path.exists(data_root):
    print(f"Error: Directory {data_root} does not exist.")
    exit(1)

classes = [d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d)) and d != 'exists']
label_map = {cls: idx for idx, cls in enumerate(classes)}
class_name_map = {idx: cls for cls, idx in label_map.items()}
print(f"Detected classes: {len(classes)} classes")

class MathSymbolDataset(Dataset):
    def __init__(self, root_dir, transform=None, use_otsu=True):
        self.root_dir = root_dir
        self.transform = transform
        self.use_otsu = use_otsu
        self.images = []
        self.labels = []
        for cls in classes:
            cls_dir = os.path.join(root_dir, cls)
            if os.path.exists(cls_dir) and cls != 'exists':
                for img_file in os.listdir(cls_dir):
                    if img_file.lower().endswith(('.jpeg', '.jpg', '.png')):
                        self.images.append(os.path.join(cls_dir, img_file))
                        self.labels.append(label_map[cls])
        print(f"Loaded {len(self.images)} images across {len(classes)} classes")
    
    def __len__(self): 
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None: 
            raise ValueError(f"Failed to load image: {img_path}")
        image = cv2.resize(image, (45, 45))
        # Otsu for binary enhancement (toggleable)
        if self.use_otsu:
            _, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        image = cv2.GaussianBlur(image, (3, 3), 0)
        image = cv2.convertScaleAbs(image, alpha=1.2, beta=10)
        pil_image = Image.fromarray(image.astype(np.uint8))
        if self.transform: 
            image = self.transform(pil_image)
        return image, self.labels[idx]

# Transforms (Otsu in __getitem__; Grayscale fix for jitter)
base_transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize((0.5,), (0.5,))
])

aug_transform = transforms.Compose([
    transforms.RandomRotation(degrees=10),
    transforms.RandomAffine(degrees=0, shear=10, translate=(0.05, 0.05), fill=255),
    transforms.RandomResizedCrop(45, scale=(0.9, 1.1)),
    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
    transforms.Grayscale(num_output_channels=1),  # Fix: Ensure grayscale post-jitter
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load & Analyze
full_dataset = MathSymbolDataset(data_root, transform=base_transform, use_otsu=True)
class_counts = Counter(full_dataset.labels)
print("Original Class Distribution:")
for cls_idx, count in sorted(class_counts.items()):
    print(f"  Class {cls_idx} ({class_name_map[cls_idx]}): {count} samples")

# Hardcoded F1 (from your report; filter to dataset classes)
class_to_f1 = {
    'beta': 0.9832, 'pm': 0.9952, 'Delta': 0.9733, 'gamma': 0.9579, 'infty': 0.9766,
    'rightarrow': 0.9805, 'div': 0.9616, 'gt': 0.9886, 'forward_slash': 0.6466,
    'leq': 0.9858, 'mu': 0.9778, 'in': 0.9444, 'times': 0.4735, 'sin': 0.9954,
    'R': 0.9057, 'u': 0.9317, '9': 0.8430, '0': 0.9082, '{': 0.9567,
    '7': 0.9215, 'i': 0.9742, 'N': 0.9244, 'G': 0.8693, '+': 0.9870,
    ',': 0.5458, '6': 0.9442, 'z': 0.8587, '}': 0.9393, '1': 0.8165,
    '8': 0.9617, 'T': 0.9241, 'S': 0.8927, 'cos': 0.9985, 'A': 0.9627,
    '-': 0.9918, 'f': 0.9488, 'o': 0.7692, 'H': 0.9394, 'sigma': 0.9772,
    'sqrt': 0.9755, 'pi': 0.9647, 'int': 0.9652, 'sum': 0.9810, 'lim': 0.9877,
    'lambda': 0.9835, 'neq': 0.9941, 'log': 0.9921, 'ldots': 1.0000,
    'forall': 0.9808, 'lt': 0.9906, 'theta': 0.9547, 'ascii_124': 0.4080,
    'M': 0.8722, '!': 0.9960, 'alpha': 0.9638, 'j': 0.9652, 'C': 0.9462,
    ']': 0.9483, '(': 0.9333, 'd': 0.9774, 'v': 0.9537, 'prime': 0.5730,
    'q': 0.8460, '=': 0.9860, '4': 0.9688, 'X': 0.8321, 'phi': 0.9945,  
    '3': 0.9892, 'tan': 0.9968, 'e': 0.9495, ')': 0.9548, '[': 0.9472,
    'b': 0.9688, 'k': 0.9771, 'l': 0.8168, 'geq': 0.9811, '2': 0.9518,
    'y': 0.9546, '5': 0.9288, 'p': 0.9768, 'w': 0.9803
}
class_to_f1 = {name: f1 for name, f1 in class_to_f1.items() if name in label_map}

f1_values = np.array(list(class_to_f1.values()))
dynamic_threshold = np.percentile(f1_values, 25)
print(f"\nDynamic F1 Threshold: {dynamic_threshold:.4f}")
low_class_names = [name for name, f1 in class_to_f1.items() if f1 < dynamic_threshold]
print(f"Low-performing classes: {low_class_names}")
low_classes = [label_map[name] for name in low_class_names if name in label_map]
print(f"Low classes indices: {low_classes}")

all_counts = list(class_counts.values())
percentile_70 = int(np.percentile(all_counts, 70))
percentile_80 = int(np.percentile(all_counts, 80))
print(f"Percentiles: 70th={percentile_70}, 80th={percentile_80}")

target_count = {}
for cls_idx in class_counts:
    if cls_idx in low_classes:
        target_count[cls_idx] = percentile_80
        print(f"Low class {cls_idx} ({class_name_map[cls_idx]}): target {percentile_80}")
    else:
        target_count[cls_idx] = percentile_70
        print(f"Class {cls_idx} ({class_name_map[cls_idx]}): target {percentile_70}")

def smart_balance_dataset(dataset, class_counts, target_counts):
    print(f"\nSmart Balancing:")
    balanced_images, balanced_labels = [], []
    
    # Original samples
    for img_path, label in zip(dataset.images, dataset.labels):
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None: continue
        image = cv2.resize(image, (45, 45))
        if dataset.use_otsu:
            _, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        image = cv2.GaussianBlur(image, (3, 3), 0)
        image = cv2.convertScaleAbs(image, alpha=1.2, beta=10)
        pil_image = Image.fromarray(image.astype(np.uint8))
        tensor_img = base_transform(pil_image)
        balanced_images.append(tensor_img)
        balanced_labels.append(label)
    
    total_generated = 0
    for cls_idx, count in class_counts.items():
        target = target_counts[cls_idx]
        if count < target:
            shortage = target - count
            cls_indices = [i for i in range(len(dataset)) if dataset.labels[i] == cls_idx]
            aug_per_sample = min(10, max(1, shortage // len(cls_indices)))
            generated = 0
            for idx in cls_indices:
                if generated >= shortage: break
                # Reload original for aug
                img_path = dataset.images[idx]
                image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if image is None: continue
                image = cv2.resize(image, (45, 45))
                if dataset.use_otsu:
                    _, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                image = cv2.GaussianBlur(image, (3, 3), 0)
                image = cv2.convertScaleAbs(image, alpha=1.2, beta=10)
                pil_image = Image.fromarray(image.astype(np.uint8))
                for _ in range(aug_per_sample):
                    if generated >= shortage: break
                    aug_img = aug_transform(pil_image)
                    balanced_images.append(aug_img)
                    balanced_labels.append(cls_idx)
                    generated += 1
            total_generated += generated
            print(f"Class {cls_idx} ({class_name_map[cls_idx]}): {count} -> {count + generated} (+{generated})")
    
    print(f"Total augmented: {total_generated}")
    return torch.stack(balanced_images), torch.tensor(balanced_labels)

balanced_tensors, balanced_labels = smart_balance_dataset(full_dataset, class_counts, target_count)
balanced_dataset = TensorDataset(balanced_tensors, balanced_labels)

final_counts = Counter(balanced_labels.numpy())
print(f"\nFinal Balanced Distribution:")
for cls_idx, count in sorted(final_counts.items()):
    print(f"  Class {cls_idx} ({class_name_map[cls_idx]}): {count} samples")

# Splits
train_size = int(0.8 * len(balanced_dataset))
val_size = int(0.1 * len(balanced_dataset))
test_size = len(balanced_dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
    balanced_dataset, [train_size, val_size, test_size]
)

print(f"\nSplits: Train {len(train_dataset)}, Val {len(val_dataset)}, Test {len(test_dataset)}")

def visualize_balanced_samples(balanced_dataset, num_samples=18):
    fig, axes = plt.subplots(3, 6, figsize=(15, 8))
    classes_to_show = list(range(min(num_samples, len(classes))))
    for i, cls_idx in enumerate(classes_to_show):
        row, col = i // 6, i % 6
        if row >= 3: break
        # Find sample
        for img, label in balanced_dataset:
            if label.item() == cls_idx:
                axes[row, col].imshow(img.squeeze().numpy(), cmap='gray')
                axes[row, col].set_title(f'Class {cls_idx}\n{class_name_map[cls_idx][:8]}')
                axes[row, col].axis('off')
                break
    plt.tight_layout()
    plt.suptitle('Balanced Dataset Samples (Otsu-Enhanced)', y=1.02)
    plt.savefig('balanced_samples.png')  # For thesis
    plt.close()  # Non-blocking
    print("Visualization saved as 'balanced_samples.png'")

if __name__ == '__main__':
    visualize_balanced_samples(balanced_dataset)
    
    # Save datasets (not loaders) + metadata
    state = {
        'balanced_dataset': balanced_dataset,
        'train_dataset': train_dataset,
        'val_dataset': val_dataset,
        'test_dataset': test_dataset,
        'label_map': label_map, 
        'class_name_map': class_name_map,
        'final_class_counts': final_counts
    }
    torch.save(state, 'improved_dataset_state.pth')
    print(f"\nDataset saved as 'improved_dataset_state.pth' ({len(balanced_dataset)} samples)")
    print("MTech Project Complete! Run train.py next. 🚀")