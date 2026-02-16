import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from torch.utils.data import TensorDataset, DataLoader
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

X_tensor = torch.tensor(X, dtype=torch.float32).permute(0, 2, 1)
y_tensor = torch.tensor(y, dtype=torch.long)

dataset = TensorDataset(X_tensor, y_tensor)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# --- Load model ---
model = SleepEEGCNN().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# --- Make predictions ---
predictions = []

with torch.no_grad():
    for batch_X, _ in dataloader:
        batch_X = batch_X.to(DEVICE)
        outputs = model(batch_X)
        preds = torch.argmax(outputs, dim=1)
        predictions.extend(preds.cpu().numpy())

print("[INFO] Predictions:")
print(predictions)
