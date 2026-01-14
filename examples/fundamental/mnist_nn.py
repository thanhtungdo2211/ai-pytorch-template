import torch
import torchvision.datasets as dsets
import torchvision.transforms as tranforms

import random

device = 'cuda' if torch.cuda.is_available() else 'cpu'

random.seed(111)
torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)
    
LEARNING_RATE = 0.001
TRANING_EPOCHS = 15
BATCH_SIZE = 100

mnist_train = dsets.MNIST(
    root='../Pytorch-master/MNIST_data/',
    train=True,
    transform=tranforms.ToTensor(),
    download=True
)

mnist_test = dsets.MNIST(
    root='../Pytorch-master/MNIST_data/',
    train=False,
    transform=tranforms.ToTensor(),
    download=True
)

data_loader = torch.utils.data.DataLoader(
    dataset=mnist_train,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True
)
    
linear1 = torch.nn.Linear(784, 256, bias=True)
linear2 = torch.nn.Linear(256, 256, bias=True)
linear3 = torch.nn.Linear(256, 10, bias=True)

relu = torch.nn.ReLU()

torch.nn.init.normal_(linear1.weight)
torch.nn.init.normal_(linear2.weight)
torch.nn.init.normal_(linear3.weight)

model = torch.nn.Sequential(linear1, relu, linear2, relu, linear3).to(device)

criterion = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)

total_batch = len(data_loader)

for epoch in range(TRANING_EPOCHS):
    avg_cost = 0
    
    for X, Y in data_loader:
        X = X.view(-1, 28 * 28).to(device)
        Y = Y.to(device)
        
        optimizer.zero_grad()
        hypothesis = model(X)
        
        cost = criterion(hypothesis, Y)
        cost.backward()
        optimizer.step()
        
        avg_cost = avg_cost + cost / total_batch 
    
    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))
    
torch.save(model.state_dict(), 'mnist_model_weights.pth')
