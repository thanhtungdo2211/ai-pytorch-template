import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np

class ManyToOneRNN(nn.Module):
    """
    Many-to-One RNN: Takes a sequence and outputs a single prediction
    Input: (batch_size, seq_len, input_size)
    Output: (batch_size, output_size)
    """
    def __init__(self, input_size, hidden_size, output_size, n_layers=1, dropout=0.2):
        super(ManyToOneRNN, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, input_size)
        Returns:
            output: (batch_size, output_size)
        """
        # RNN forward pass
        # output: (batch_size, seq_len, hidden_size)
        # hidden: (n_layers, batch_size, hidden_size)
        rnn_output, hidden = self.rnn(x)
        
        # Use last timestep output (many-to-one)
        last_output = rnn_output[:, -1, :]  # (batch_size, hidden_size)
        last_output = self.dropout(last_output)
        
        # Fully connected layer
        output = self.fc(last_output)  # (batch_size, output_size)
        return output

def train_epoch(model, dataloader, optimizer, loss_fn, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = 0
    
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        predictions = model(X)
        loss = loss_fn(predictions, y)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches

def evaluate(model, dataloader, loss_fn, device):
    """Evaluate model on validation/test set"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    num_batches = 0
    
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            
            predictions = model(X)
            loss = loss_fn(predictions, y)
            total_loss += loss.item()
            num_batches += 1
            
            # Calculate accuracy
            pred_labels = torch.argmax(predictions, dim=1)
            correct += (pred_labels == y).sum().item()
            total += y.size(0)
    
    accuracy = correct / total
    return total_loss / num_batches, accuracy

def create_synthetic_data(num_samples=100, seq_len=10, input_size=5, num_classes=2):
    """Create synthetic sequential data for testing"""
    X = torch.randn(num_samples, seq_len, input_size)
    y = torch.randint(0, num_classes, (num_samples,))
    return X, y

# Example usage
if __name__ == "__main__":
    # Parameters
    input_size = 5
    hidden_size = 32
    output_size = 2
    n_layers = 2
    batch_size = 16
    epochs = 20
    learning_rate = 0.001
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print(f"Using device: {device}")
    
    # Create synthetic data
    # print("\nCreating synthetic data...")
    X, y = create_synthetic_data(num_samples=200, seq_len=15, input_size=input_size, num_classes=output_size)
    
    # Split into train/test (80/20)
    train_size = int(0.8 * len(X))
    test_size = len(X) - train_size
    
    train_X, test_X = X[:train_size], X[train_size:]
    train_y, test_y = y[:train_size], y[train_size:]
    
    # Create DataLoaders
    train_dataset = TensorDataset(train_X, train_y)
    test_dataset = TensorDataset(test_X, test_y)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print(train_loader)
    # Initialize model
    print(f"\nInitializing RNN model...")
    model = ManyToOneRNN(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        n_layers=n_layers,
        dropout=0.2
    ).to(device)
    
    print(f"Model:\n{model}")
    
    # Training setup
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    
    # Training loop
    print(f"\nTraining for {epochs} epochs...")
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        test_loss, test_acc = evaluate(model, test_loader, loss_fn, device)
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:2d}/{epochs} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")
    
    # Inference
    print(f"\n--- Inference ---")
    sample_input = torch.randn(1, 15, input_size).to(device)
    with torch.no_grad():
        output = model(sample_input)
        pred = torch.argmax(output, dim=1)
        probabilities = torch.softmax(output, dim=1)
    
    print(f"Sample input shape: {sample_input.shape}")
    print(f"Model output shape: {output.shape}")
    print(f"Predicted class: {pred.item()}")
    print(f"Class probabilities: {probabilities[0].cpu().numpy()}")