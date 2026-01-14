import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[1], [2], [3]])

W = torch.zeros(1)
lr = 0.1

NUM_EPOCHS = 10
for epoch in range(NUM_EPOCHS + 1):
    hyppthesis = x_train * W
    cost = torch.mean((hyppthesis - y_train) ** 2)
    gradient = torch.sum((W * x_train - y_train) * x_train)
    
    print('Epoch {:4d}/{} W: {:.3f}, Cost: {:.6f}'.format(
        epoch, NUM_EPOCHS, W.item(), cost.item()
    ))
    
    W -= lr * gradient

### High-level Implementation with SGD ###
x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[1], [2], [3]])
W = torch.zeros(1, requires_grad=True)
optimizer = optim.SGD([W], lr=0.1)

NUM_EPOCHS = 10
for epoch in range(NUM_EPOCHS + 1):
    hypothesis = x_train * W

    cost = torch.mean((hypothesis - y_train) ** 2)

    print('High level implement - Epoch {:4d}/{} W: {:.3f} Cost: {:.6f}'.format(
        epoch, NUM_EPOCHS, W.item(), cost.item()
    ))

    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
