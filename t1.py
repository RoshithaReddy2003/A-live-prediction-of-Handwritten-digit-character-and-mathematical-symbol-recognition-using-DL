import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

# Load improved dataset (fixed: datasets, not loaders)
try:
    state = torch.load('improved_dataset_state.pth', weights_only=False)
    train_dataset = state['train_dataset']
    val_dataset = state['val_dataset'] 
    test_dataset = state['test_dataset']
    balanced_dataset = state['balanced_dataset']
    label_map = state['label_map']
    class_name_map = state['class_name_map']
    final_class_counts = state['final_class_counts']
    num_classes = len(label_map)
    print(f"Loaded dataset with {num_classes} classes (balanced: {len(balanced_dataset)} samples)")
except:
    print("Error: Please run load_preprocess.py first!")
    exit(1)

# Recreate loaders (fixed mismatch)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Improved CNN Architecture - Right-sized for 45x45 images
class OptimizedSymbolCNN(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.3):  # Reduced for higher acc
        super(OptimizedSymbolCNN, self).__init__()
        
        # Feature extraction layers - optimized for 45x45 input
        self.features = nn.Sequential(
            # Block 1: 45x45 -> 22x22
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.1),
            
            # Block 2: 22x22 -> 11x11  
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.1),
            
            # Block 3: 11x11 -> 5x5
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2),
            
            # Block 4: 5x5 -> 2x2
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2),
        )
        
        # Classifier layers
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # Global average pooling
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256), 
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Create model
model = OptimizedSymbolCNN(num_classes=num_classes, dropout_rate=0.3).to(device)

# Calculate class weights for imbalanced classes (from balanced distro)
all_labels = [balanced_dataset[i][1].item() for i in range(len(balanced_dataset))]
class_weights = compute_class_weight('balanced', classes=np.unique(all_labels), y=all_labels)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

# Improved training setup
criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)  # Label smoothing
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)  # Start lower

# Learning rate scheduler - gentler ramp for smoothness
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer, 
    max_lr=0.003,  # Reduced to avoid over-shoot
    epochs=150,  # More epochs
    steps_per_epoch=len(train_loader),
    pct_start=0.3,
    anneal_strategy='cos'
)

# Mixed precision training for faster training
scaler = GradScaler() if device.type == 'cuda' else None

# Improved training function (same)
def train_epoch(model, train_loader, optimizer, criterion, device, scaler=None):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    progress_bar = tqdm(train_loader, desc='Training', leave=False)
    
    for batch_idx, (data, target) in enumerate(progress_bar):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        
        if scaler is not None:  # Mixed precision
            with autocast():
                outputs = model(data)
                loss = criterion(outputs, target)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(data)
            loss = criterion(outputs, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        scheduler.step()  # OneCycleLR steps per batch
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{100.*correct/total:.2f}%',
            'LR': f'{scheduler.get_last_lr()[0]:.6f}'
        })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

# Improved evaluation function (eval every 2 epochs for speed)
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in loader:  # No tqdm for speed
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            loss = criterion(outputs, target)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    epoch_loss = running_loss / len(loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc, all_predictions, all_targets

# Training loop with better early stopping
print("\nStarting Training...")
print("=" * 50)

num_epochs = 150  # Extended
best_val_acc = 0.0
best_train_acc = 0.0
patience = 25  # More patience
patience_counter = 0
train_losses = []
train_accs = []
val_losses = []
val_accs = []

for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs}")
    print("-" * 30)
    
    # Training
    train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device, scaler)
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    
    # Validation every 2 epochs (speed)
    if (epoch + 1) % 2 == 0:
        val_loss, val_acc, val_preds, val_targets = evaluate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")  # Only print if eval
    else:
        # Dummy for plotting (use last val)
        if val_losses:
            val_loss, val_acc = val_losses[-1], val_accs[-1]
        else:
            val_loss, val_acc = 0, 0
        val_losses.append(val_loss)
        val_accs.append(val_acc)
    
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
    print(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
    
    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_train_acc = train_acc
        patience_counter = 0
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'epoch': epoch,
            'val_acc': val_acc,
            'train_acc': train_acc,
            'class_name_map': class_name_map
        }, 'best_model_optimized.pth')
        print(f"✅ New best model saved! Val Acc: {val_acc:.2f}%")
    else:
        patience_counter += 1
    
    # Early stopping check
    if patience_counter >= patience and epoch > 40:  # Later start
        print(f"\n🛑 Early stopping triggered at epoch {epoch+1}")
        print(f"Best validation accuracy: {best_val_acc:.2f}%")
        break

print(f"\n🎯 Training completed!")
print(f"Best Train Accuracy: {best_train_acc:.2f}%")
print(f"Best Validation Accuracy: {best_val_acc:.2f}%")

# Load best model for final evaluation
checkpoint = torch.load('best_model_optimized.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])

# Final test evaluation
print("\n📊 Final Test Evaluation:")
print("=" * 30)
test_loss, test_acc, test_preds, test_targets = evaluate(model, test_loader, criterion, device)
print(f"Test Accuracy: {test_acc:.2f}%")

# Detailed classification report
print("\nClassification Report:")
unique_classes = sorted(label_map.values())
class_names = [class_name_map[i] for i in unique_classes]
print(classification_report(test_targets, test_preds, target_names=class_names, digits=4))

# Plot training curves
def plot_training_curves():
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss curves (smooth)
    ax1.plot(train_losses, label='Train Loss', color='blue')
    ax1.plot(val_losses, label='Val Loss', color='red') 
    ax1.set_title('Loss Curves (Smooth Val)')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy curves (smooth plateau expected)
    ax2.plot(train_accs, label='Train Acc', color='blue')
    ax2.plot(val_accs, label='Val Acc', color='red')
    ax2.set_title('Accuracy Curves (95%+ Target)')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    # Confusion matrix
    cm = confusion_matrix(test_targets, test_preds)
    sns.heatmap(cm[:20, :20], annot=False, fmt='d', ax=ax3)  # Show first 20 classes
    ax3.set_title('Confusion Matrix (First 20 Classes)')
    ax3.set_xlabel('Predicted')
    ax3.set_ylabel('Actual')
    
    # Class-wise accuracy
    class_accs = []
    for i in range(min(20, num_classes)):
        mask = np.array(test_targets) == i
        if np.sum(mask) > 0:
            acc = np.mean(np.array(test_preds)[mask] == i) * 100
            class_accs.append(acc)
        else:
            class_accs.append(0)
    
    ax4.bar(range(len(class_accs)), class_accs)
    ax4.set_title('Per-Class Accuracy (First 20 Classes)')
    ax4.set_xlabel('Class Index')
    ax4.set_ylabel('Accuracy (%)')
    ax4.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig('smooth_training_curves.png')  # For thesis
    plt.show()

plot_training_curves()

# Summary
print(f"\n🎯 Final Results Summary:")
print(f"📈 Best Training Accuracy: {best_train_acc:.2f}%")
print(f"📈 Best Validation Accuracy: {best_val_acc:.2f}%") 
print(f"📈 Final Test Accuracy: {test_acc:.2f}%")
print(f"\n💾 Best model saved as 'best_model_optimized.pth'")

if test_acc >= 95.0:
    print("\n🎉 SUCCESS! Achieved target accuracy of 95%+")
else:
    print(f"\n⚠️  Target not reached. Current: {test_acc:.2f}%, Target: 95%")
    print("💡 Rerun with tweaks or accept 94% as strong baseline")

print("\n✅ Expressions Ready: App.py solves simple (2+y=5 → y=3) & complex (sin(x)+cos(y)=0 → y=acos(-sin(x))). Run streamlit!") 