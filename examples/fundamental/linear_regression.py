import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Define the model
x_train = torch.tensor([[1], [2], [3]], dtype=torch.float32)
y_train = torch.tensor([[2], [4], [6]], dtype=torch.float32)

# Define weight and bias
W = torch.zeros(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# Define the cost function
hypothesys = x_train * W + b
cost = torch.mean((hypothesys - y_train) ** 2)
optimizer = optim.SGD([W, b], lr=0.01)

# Training
NUM_EPOCHS = 1000

for epoch in range(NUM_EPOCHS + 1):
    # Calculate the cost
    hypothesys = x_train * W + b
    cost = torch.mean((hypothesys - y_train) ** 2)

    # Gradient descent
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print('Epoch {:4d}/{} W: {:.3f}, b: {:.3f} Cost: {:.6f}'.format(
            epoch, NUM_EPOCHS, W.item(), b.item(), cost.item()
        ))

### High level implementation with nn.Module
# Define the model
x_train = torch.tensor([[1], [2], [3]], dtype=torch.float32)
y_train = torch.tensor([[2], [4], [6]], dtype=torch.float32)

# Define weight and bias
W = torch.zeros(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

model = LinearRegressionModel()
hypothesis = model(x_train)

cost = torch.nn.functional.mse_loss(hypothesis, y_train)
optimizer = optim.SGD(model.parameters(), lr=0.01)

NUM_EPOCHS = 1000
for epoch in range(NUM_EPOCHS + 1):
    # Calculate the cost
    hypothesys = model(x_train)
    cost = torch.nn.functional.mse_loss(hypothesys, y_train)

    # Gradient descent
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print('High level implement - Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch, NUM_EPOCHS, cost.item()
        ))
