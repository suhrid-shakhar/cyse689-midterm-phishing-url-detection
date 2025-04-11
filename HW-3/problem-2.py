import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import time
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("Android_Feats.csv", header=None)
X = df.iloc[:, :-1].values
y = pd.to_numeric(df.iloc[:, -1], errors='coerce').fillna(0).clip(0, 1).values

# Normalize
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# Train-test split (70/30)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Convert to tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test.reshape(-1, 1), dtype=torch.float32)

# Define network
class AndroidNet(nn.Module):
    def __init__(self):
        super(AndroidNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(X.shape[1], 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

model = AndroidNet()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Lists to store loss and accuracy for visualization
train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []

# Training
start = time.time()
for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    y_pred = model(X_train)
    loss = criterion(y_pred, y_train)
    loss.backward()
    optimizer.step()

    # Record loss and accuracy
    train_losses.append(loss.item())
    
    # Calculate training accuracy
    train_acc = ((y_pred > 0.5) == y_train).float().mean().item()
    train_accuracies.append(train_acc)
    
    # Calculate test accuracy and loss
    model.eval()
    with torch.no_grad():
        y_test_pred = model(X_test)
        test_loss = criterion(y_test_pred, y_test)
        test_losses.append(test_loss.item())
        test_acc = ((y_test_pred > 0.5) == y_test).float().mean().item()
        test_accuracies.append(test_acc)

end = time.time()

# Print training stats
print(f"Training Time: {end - start:.2f}s")
print(f"Training Accuracy: {train_accuracies[-1]:.4f}")
print(f"Testing Accuracy: {test_accuracies[-1]:.4f}")

# Plot Loss and Accuracy over epochs
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Loss plot
ax1.plot(range(100), train_losses, label='Training Loss')
ax1.plot(range(100), test_losses, label='Testing Loss')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss')
ax1.set_title('Loss over Epochs')
ax1.legend()

# Accuracy plot
ax2.plot(range(100), train_accuracies, label='Training Accuracy')
ax2.plot(range(100), test_accuracies, label='Testing Accuracy')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Accuracy')
ax2.set_title('Accuracy over Epochs')
ax2.legend()

plt.tight_layout()
plt.show()
