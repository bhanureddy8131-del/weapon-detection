from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import random
from src.config import LSTM_MODEL_PATH, ROOT_DIR, SEQUENCE_LENGTH
from src.modeling.classifiers.lstm import (
    ACTION_ARCHITECTURE,
    ACTION_CLASSES,
    POSE_FEATURE_LAYOUT,
    POSE_FEATURE_SIZE,
    BidirectionalLSTM,
)

# --- CONFIGURATION ---
DATA_PATH = Path(ROOT_DIR) / "data" / "processed" / "lstm_data"
MODEL_SAVE_PATH = Path(LSTM_MODEL_PATH)
EPOCHS = 80              # More epochs for the scheduler to work
BATCH_SIZE = 64          # Bigger batch for stable gradients
LEARNING_RATE = 0.001
HIDDEN_SIZE = 128        # Doubled brain size
NUM_LAYERS = 2
DROPOUT_RATE = 0.4       # Higher dropout to prevent overfitting

# TARGET SAMPLES per class (We force all classes to match this count)
TARGET_SAMPLES = 6000 

# --- 1. DATASET WITH NOISE AUGMENTATION ---
class ActionDataset(Dataset):
    def __init__(self, features, labels, augment=False):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.augment = augment

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = self.features[idx]
        y = self.labels[idx]
        
        # On-the-fly Data Augmentation (Only for Training)
        if self.augment:
            # Add random jitter (simulate shaky camera/sensor noise)
            noise = torch.randn_like(x) * 0.005
            x = x + noise
            
        return x, y

# --- 3. TRAINING FUNCTION ---
def train_model():
    print(" Starting Professional Training Pipeline...")

    if MODEL_SAVE_PATH.exists():
        raise FileExistsError(
            "Refusing to overwrite the existing LSTM checkpoint: "
            f"{MODEL_SAVE_PATH}"
        )

    if not DATA_PATH.is_dir():
        raise FileNotFoundError(
            f"Action training data is missing: {DATA_PATH}. "
            "Run scripts/create_lstm_data.py with labeled action clips first."
        )
    
    features = []
    labels = []
    
    class_directories = {
        path.name.strip().lower(): path
        for path in DATA_PATH.iterdir()
        if path.is_dir()
    }
    if set(class_directories) != ACTION_CLASSES:
        raise ValueError(
            "Action training folders must be exactly "
            f"{sorted(ACTION_CLASSES)}; found {sorted(class_directories)}."
        )
    class_names = sorted(ACTION_CLASSES)
    class_map = {name: i for i, name in enumerate(class_names)}
    print(f"   Classes Map: {class_map}")

    # --- ADVANCED BALANCING STRATEGY ---
    for class_name in class_names:
        class_dir = class_directories[class_name]
        files = sorted(class_dir.glob("*.npy"))
        
        print(f"   Processing '{class_name}': Found {len(files)} files.")
        
        if not files:
            raise ValueError(f"Action class '{class_name}' contains no .npy sequences.")

        selected_files = []
        if len(files) >= TARGET_SAMPLES:
            selected_files = random.sample(files, TARGET_SAMPLES)
        else:
            # Oversampling (Duplication)
            repeats = TARGET_SAMPLES // len(files)
            remainder = TARGET_SAMPLES % len(files)
            selected_files = files * repeats + files[:remainder]
            print(f"      ↳ Oversampled: Copied data {repeats}x times to match size.")

        # Load Data
        for file in selected_files:
            data = np.load(file)
            expected_shape = (SEQUENCE_LENGTH, POSE_FEATURE_SIZE)
            if data.shape != expected_shape:
                raise ValueError(
                    f"Training sequence {file} has shape {data.shape}; "
                    f"expected {expected_shape}."
                )
            if not np.isfinite(data).all():
                raise ValueError(f"Training sequence {file} contains non-finite values.")
            features.append(data)
            labels.append(class_map[class_name])

    X = np.array(features)
    y = np.array(labels)
    
    print(f"   Final Dataset Size: {len(X)} samples (Perfectly Balanced)")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Enable augmentation ONLY for training data
    train_dataset = ActionDataset(X_train, y_train, augment=True)
    test_dataset = ActionDataset(X_test, y_test, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')

    # Model Init
    if X.ndim != 3 or X.shape[1:] != (SEQUENCE_LENGTH, POSE_FEATURE_SIZE):
        raise ValueError(
            "Loaded training data does not match the configured sequence contract: "
            f"received {X.shape}."
        )
    input_size = POSE_FEATURE_SIZE
    model = BidirectionalLSTM(
        input_size,
        HIDDEN_SIZE,
        NUM_LAYERS,
        len(class_names),
        dropout_rate=DROPOUT_RATE,
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5) 
    
    # FIX: Removed 'verbose=True' to fix TypeError
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    best_acc = 0.0

    print(f" Training on {device} with Bidirectional LSTM...")

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Evaluate
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        acc = 100 * correct / total
        avg_loss = running_loss/len(train_loader)
        
        # Step the scheduler
        scheduler.step(acc)
        
        # We manually print the LR here, so we don't need verbose=True in the scheduler
        print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {avg_loss:.4f} | Acc: {acc:.2f}% | LR: {optimizer.param_groups[0]['lr']:.6f}")

        if acc > best_acc:
            best_acc = acc
            MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'class_map': class_map,
                'input_size': input_size,
                'hidden_size': HIDDEN_SIZE,
                'num_layers': NUM_LAYERS,
                'bidirectional': True,
                'architecture': ACTION_ARCHITECTURE,
                'sequence_length': SEQUENCE_LENGTH,
                'feature_layout': POSE_FEATURE_LAYOUT,
                'dropout_rate': DROPOUT_RATE,
            }, str(MODEL_SAVE_PATH))

    print(f"\n Best Accuracy: {best_acc:.2f}%")
    print(f" Saved to {MODEL_SAVE_PATH}")
    
if __name__ == "__main__":
    train_model()