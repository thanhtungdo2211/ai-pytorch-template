import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]
y_data = [[0], [0], [0], [1], [1], [1]]
x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)

W = torch.zeros((2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)
# optimizer 설정
optimizer = optim.SGD([W, b], lr=1)

nb_epochs = 1000
for epoch in range(nb_epochs + 1):

    # Cost 계산
    hypothesis = torch.sigmoid(x_train.matmul(W) + b) # or .mm or @
    cost = -(y_train * torch.log(hypothesis) + 
             (1 - y_train) * torch.log(1 - hypothesis)).mean()

    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    with torch.no_grad():
        predicted = hypothesis >= 0.5
        accuracy = (predicted.float() == y_train).float().mean()

    # 100번마다 로그 출력
    if epoch % 100 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f} Accuracy: {:.2f}%'.format(
            epoch, nb_epochs, cost.item(), accuracy.item() * 100
        ))

with torch.no_grad():
    logits = x_train.matmul(W) + b
    probs = torch.sigmoid(logits)

    print("Logits:\n", logits)
    print("Probabilities:\n", probs)
    print("Predicted labels (prob >= 0.5):\n", (probs >= 0.5).int())
    print("Actual labels:\n", y_train.int())
    print("Difference (y_train - probs):\n", y_train - probs)

# High level implementation with nn.Module
x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]
y_data = [[0], [0], [0], [1], [1], [1]]
x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)

W = torch.zeros((2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

class BinaryClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.linear(x))

model = BinaryClassifier()

optimizer = optim.SGD(model.parameters(), lr=1)

nb_epochs = 100

for epoch in range(nb_epochs + 1):
    
    hypothesis = model(x_train)
    cost = F.binary_cross_entropy(hypothesis, y_train)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    with torch.no_grad():
        predicted = hypothesis >= torch.FloatTensor([0.5])
        correct_prediction = predicted.float() == y_train
        accuracy = correct_prediction.sum().item() / len(correct_prediction)

    if epoch % 10 == 0:
        print('High level implement - Epoch {:4d}/{} Cost: {:.6f} Accuracy: {:2.2f}%'.format(
            epoch, nb_epochs, cost.item(), accuracy * 100
        ))