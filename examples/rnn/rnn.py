import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from collections import Counter
import re

class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab_size=5000, max_len=100):
        # Build vocabulary
        words = []
        for text in texts:
            words.extend(re.findall(r'\w+', text.lower()))
        
        word_counts = Counter(words)
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        for idx, (word, _) in enumerate(sorted(word_counts.items(), 
                                               key=lambda x: x[1], 
                                               reverse=True)[:vocab_size-2], start=2):
            self.word2idx[word] = idx
        
        self.vocab_size = len(self.word2idx)
        self.max_len = max_len
        self.texts = texts
        self.labels = labels
    
    def text_to_indices(self, text):
        words = re.findall(r'\w+', text.lower())
        indices = [self.word2idx.get(w, self.word2idx['<UNK>']) for w in words]
        
        # Pad or truncate
        if len(indices) < self.max_len:
            indices += [self.word2idx['<PAD>']] * (self.max_len - len(indices))
        else:
            indices = indices[:self.max_len]
        
        return torch.tensor(indices, dtype=torch.long)
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text_tensor = self.text_to_indices(self.texts[idx])
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return text_tensor, label

class SimpleRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers=1, dropout=0.3):
        super(SimpleRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, n_layers, 
                         batch_first=True, dropout=dropout if n_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, text):
        # text shape: (batch_size, seq_len)
        embedded = self.dropout(self.embedding(text))  # (batch_size, seq_len, embedding_dim)
        output, hidden = self.rnn(embedded)  # output: (batch_size, seq_len, hidden_dim)
        # Use last hidden state
        hidden = self.dropout(hidden[-1])  # (batch_size, hidden_dim)
        return self.fc(hidden)  # (batch_size, output_dim)

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers=2, dropout=0.3):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, 
                           batch_first=True, dropout=dropout if n_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, text):
        embedded = self.dropout(self.embedding(text))  # (batch_size, seq_len, embedding_dim)
        output, (hidden, cell) = self.lstm(embedded)  # hidden: (n_layers, batch_size, hidden_dim)
        hidden = self.dropout(hidden[-1])  # Use last layer's hidden state
        return self.fc(hidden)

class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers=2, dropout=0.3):
        super(GRUClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.gru = nn.GRU(embedding_dim, hidden_dim, n_layers, 
                         batch_first=True, dropout=dropout if n_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, text):
        embedded = self.dropout(self.embedding(text))
        output, hidden = self.gru(embedded)  # hidden: (n_layers, batch_size, hidden_dim)
        hidden = self.dropout(hidden[-1])
        return self.fc(hidden)

def train_epoch(model, dataloader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    for text, labels in dataloader:
        text, labels = text.to(device), labels.to(device)
        
        optimizer.zero_grad()
        predictions = model(text)
        loss = loss_fn(predictions, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

def evaluate(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for text, labels in dataloader:
            text, labels = text.to(device), labels.to(device)
            
            predictions = model(text)
            loss = loss_fn(predictions, labels)
            total_loss += loss.item()
            
            pred_labels = torch.argmax(predictions, dim=1)
            correct += (pred_labels == labels).sum().item()
            total += labels.size(0)
    
    accuracy = correct / total
    return total_loss / len(dataloader), accuracy

# Example usage
if __name__ == "__main__":
    # Sample data
    texts = [
        "this movie is absolutely fantastic and amazing",
        "i loved this film so much",
        "worst movie ever made",
        "terrible and boring waste of time",
        "great cinematography and acting",
        "horrible plot and dialogue",
        "best film i've seen all year",
        "unwatchable garbage",
    ]
    labels = [1, 1, 0, 0, 1, 0, 1, 0]  # 1 = positive, 0 = negative
    
    # Create dataset
    dataset = TextDataset(texts, labels, vocab_size=1000, max_len=20)
    
    # Split into train/test
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=2)
    
    # Initialize model (choose one)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LSTMClassifier(vocab_size=dataset.vocab_size, embedding_dim=100, 
                          hidden_dim=128, output_dim=2, n_layers=2).to(device)
    
    # Training
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    
    epochs = 20
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        test_loss, test_acc = evaluate(model, test_loader, loss_fn, device)
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
    
    # Make prediction
    sample_text = "this movie was fantastic"
    text_tensor = dataset.text_to_indices(sample_text).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(text_tensor)
        pred = torch.argmax(output, dim=1)
        print(f"\nSample: '{sample_text}' -> Sentiment: {'Positive' if pred.item() == 1 else 'Negative'}")