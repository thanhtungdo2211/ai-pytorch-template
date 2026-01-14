import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import random

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# for reproducibility
random.seed(777)
torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)
    
# parameters
training_epochs = 15
batch_size = 100

# MNIST dataset
mnist_train = dsets.MNIST(root='../Pytorch-master/MNIST_data/',
                          train=True,
                          transform=transforms.ToTensor(),
                          download=True)

mnist_test = dsets.MNIST(root='../Pytorch-master/MNIST_data/',
                         train=False,
                         transform=transforms.ToTensor(),
                         download=True)

# dataset loaders
train_loader = torch.utils.data.DataLoader(dataset=mnist_train,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          drop_last=True)

test_loader = torch.utils.data.DataLoader(dataset=mnist_test,
                                         batch_size=batch_size,
                                         shuffle=False,
                                         drop_last=True)

# Model components
linear1 = torch.nn.Linear(784, 512, bias=True)
linear2 = torch.nn.Linear(512, 512, bias=True)
linear3 = torch.nn.Linear(512, 512, bias=True)
linear4 = torch.nn.Linear(512, 512, bias=True)
linear5 = torch.nn.Linear(512, 10, bias=True)
selu = torch.nn.SELU()
bn1 = torch.nn.BatchNorm1d(512)
bn2 = torch.nn.BatchNorm1d(512)
bn3 = torch.nn.BatchNorm1d(512)
bn4 = torch.nn.BatchNorm1d(512)

# Model with BatchNorm
model = torch.nn.Sequential(linear1, bn1, selu,
                           linear2, bn2, selu,
                           linear3, bn3, selu,
                           linear4, bn4, selu,
                           linear5).to(device)

# Loss and optimizer
criterion = torch.nn.CrossEntropyLoss().to(device)  
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Save Losses and Accuracies every epoch
train_losses = []
train_accs = []
valid_losses = []
valid_accs = []

train_total_batch = len(train_loader)
test_total_batch = len(test_loader)

for epoch in range(training_epochs):
    model.train()  # set the model to train mode

    # Training phase
    for X, Y in train_loader:
        # reshape input image into [batch_size by 784]
        X = X.view(-1, 28 * 28).to(device)
        Y = Y.to(device)

        optimizer.zero_grad()
        prediction = model(X)
        loss = criterion(prediction, Y)
        loss.backward()
        optimizer.step()

    # Evaluation phase
    with torch.no_grad():
        model.eval()     # set the model to evaluation mode

        # Evaluate on training set
        train_loss, train_acc = 0, 0
        for i, (X, Y) in enumerate(train_loader):
            X = X.view(-1, 28 * 28).to(device)
            Y = Y.to(device)

            prediction = model(X)
            correct_prediction = torch.argmax(prediction, 1) == Y
            train_loss += criterion(prediction, Y)
            train_acc += correct_prediction.float().mean()

        train_loss = train_loss / train_total_batch
        train_acc = train_acc / train_total_batch

        # Save train metrics
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        print('[Epoch %d-TRAIN] Loss: %.5f, Accuracy: %.2f%%' % 
              (epoch + 1, train_loss.item(), train_acc.item() * 100))

        # Evaluate on test set
        test_loss, test_acc = 0, 0
        for i, (X, Y) in enumerate(test_loader):
            X = X.view(-1, 28 * 28).to(device)
            Y = Y.to(device)

            prediction = model(X)
            correct_prediction = torch.argmax(prediction, 1) == Y
            test_loss += criterion(prediction, Y)
            test_acc += correct_prediction.float().mean()

        test_loss = test_loss / test_total_batch
        test_acc = test_acc / test_total_batch

        # Save validation metrics
        valid_losses.append(test_loss)
        valid_accs.append(test_acc)
        
        print('[Epoch %d-VALID] Loss: %.5f, Accuracy: %.2f%%' % 
              (epoch + 1, test_loss.item(), test_acc.item() * 100))
        print()

print('Learning finished')

# Save the trained model
torch.save(model.state_dict(), 'mnist_model_with_bn.pth')
print('Model saved successfully!')

# Plot training curves
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot([loss.cpu() for loss in train_losses], label='Train Loss')
plt.plot([loss.cpu() for loss in valid_losses], label='Valid Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot([acc.cpu() for acc in train_accs], label='Train Accuracy')
plt.plot([acc.cpu() for acc in valid_accs], label='Valid Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.tight_layout()
plt.show()