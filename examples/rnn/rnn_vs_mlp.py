import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(0)

# -----------------------
# Synthetic sequence data
# -----------------------
def make_data(n=4000, T=20):
    # X: (n, T, 1) binary
    X = torch.randint(0, 2, (n, T, 1)).float()
    # y depends on last timestep
    y = X[:, -1, 0].long()
    return X, y

T = 20
X, y = make_data(n=5000, T=T)

# Train/val split
n_train = 4000
X_train, y_train = X[:n_train], y[:n_train]
X_val, y_val = X[n_train:], y[n_train:]

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
val_loader   = DataLoader(TensorDataset(X_val, y_val), batch_size=256)

# -----------------------
# Models
# -----------------------
class MLP(nn.Module):
    def __init__(self, T, d=1, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(T * d, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2)
        )

    def forward(self, x):
        # x: (B, T, d) -> flatten
        x = x.view(x.size(0), -1)
        return self.net(x)

class SimpleRNNClassifier(nn.Module):
    def __init__(self, d=1, hidden=32):
        super().__init__()
        self.rnn = nn.RNN(input_size=d, hidden_size=hidden, batch_first=True)
        self.fc = nn.Linear(hidden, 2)

    def forward(self, x):
        # x: (B, T, d)
        out, hT = self.rnn(x)          # hT: (1, B, hidden)
        h_last = hT.squeeze(0)         # (B, hidden)
        return self.fc(h_last)

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

mlp = MLP(T=T, d=1, hidden=64)
rnn = SimpleRNNClassifier(d=1, hidden=32)

print("MLP params:", count_params(mlp))
print("RNN params:", count_params(rnn))

# -----------------------
# Training utilities
# -----------------------
def accuracy(logits, y):
    return (logits.argmax(dim=1) == y).float().mean().item()

def train_one(model, epochs=5, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for ep in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val_accs = []
            for xb, yb in val_loader:
                val_accs.append(accuracy(model(xb), yb))
        print(f"epoch {ep:02d} | val acc: {sum(val_accs)/len(val_accs):.4f}")

print("\nTraining MLP")
train_one(mlp, epochs=6, lr=1e-3)

print("\nTraining RNN")
train_one(rnn, epochs=6, lr=1e-3)
