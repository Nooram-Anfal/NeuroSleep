import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1. Load Data
# -----------------------------
def load_data():
    # Dummy data: 2 channels, 40 samples
    X = np.random.rand(2, 40)
    y = np.random.randint(0, 2, 40)
    return X, y

X, y = load_data()
print(f"[INFO] Dataset shape: X={X.shape}, y={y.shape}")
print(f"[INFO] Unique classes: {set(y)}")

# -----------------------------
# 2. Prepare Data
# -----------------------------
X_tensor = torch.tensor(X.T, dtype=torch.float32).unsqueeze(2)  # (samples, channels, length)
y_tensor = torch.tensor(y, dtype=torch.long)
dataset = TensorDataset(X_tensor, y_tensor)
loader = DataLoader(dataset, batch_size=8, shuffle=True)

# -----------------------------
# 3. Define CNN Model
# -----------------------------
class SleepEEGCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(SleepEEGCNN, self).__init__()
        self.conv1 = nn.Conv1d(2, 16, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(16)
        self.pool1 = nn.MaxPool1d(1)

        self.conv2 = nn.Conv1d(16, 32, kernel_size=1)
        self.bn2 = nn.BatchNorm1d(32)
        self.pool2 = nn.MaxPool1d(1)

        # fc1 input size will be computed dynamically
        self.fc1 = None
        self.fc2 = None
        self.num_classes = num_classes

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        if self.fc1 is None:
            # Initialize fully connected layers dynamically
            in_features = x.size(1)
            self.fc1 = nn.Linear(in_features, 64).to(x.device)
            self.fc2 = nn.Linear(64, self.num_classes).to(x.device)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# -----------------------------
# 4. Train the Model
# -----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SleepEEGCNN().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
EPOCHS = 5

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch_X, batch_y in loader:
        batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {total_loss/len(loader):.4f}")

# Save the trained model
torch.save(model.state_dict(), "sleep_cnn.pth")
print("[SUCCESS] Model trained and saved as sleep_cnn.pth")

# -----------------------------
# 5. Evaluate
# -----------------------------
model.eval()
with torch.no_grad():
    predictions = []
    for batch_X, _ in loader:
        batch_X = batch_X.to(DEVICE)
        outputs = model(batch_X)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        predictions.extend(preds)

y_true = y
y_pred = np.array(predictions)

# -----------------------------
# 6. Confusion Matrix
# -----------------------------
def compute_confusion_matrix(y_true, y_pred, num_classes=2):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm

cm = compute_confusion_matrix(y_true, y_pred)
print("[INFO] Confusion Matrix:")
print(cm)

# -----------------------------
# 7. Plot Confusion Matrix
# -----------------------------
def plot_confusion_matrix(cm, classes=['0','1'], title='Confusion Matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    fmt = 'd'
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

plot_confusion_matrix(cm)
