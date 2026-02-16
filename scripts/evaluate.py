import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from torch.utils.data import TensorDataset, DataLoader
from utils.matrices import compute_confusion_matrix
from utils.plotting import plot_confusion_matrix
from models.cnn.cnn_model import SleepEEGCNN
from utils.data_loader import load_sleepedf

# --- Config ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 8
MODEL_PATH = "sleep_cnn.pth"

# --- Load dataset ---
X, y = load_sleepedf()
print(f"[INFO] Final dataset shape: X={X.shape}, y={y.shape}")
print(f"[INFO] Unique classes: {set(y)}")

X_tensor = torch.tensor(X, dtype=torch.float32).permute(0, 2, 1)  # shape: [samples, channels, length]
y_tensor = torch.tensor(y, dtype=torch.long)

dataset = TensorDataset(X_tensor, y_tensor)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# --- Load model ---
model = SleepEEGCNN().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# --- Evaluation ---
all_preds = []
all_labels = []

with torch.no_grad():
    for batch_X, batch_y in dataloader:
        batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
        outputs = model(batch_X)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch_y.cpu().numpy())

# --- Confusion matrix ---
cm = compute_confusion_matrix(all_labels, all_preds)
print("[INFO] Confusion matrix:")
print(cm)
plot_confusion_matrix(cm, classes=[0, 1], title="SleepEEG CNN Confusion Matrix")
