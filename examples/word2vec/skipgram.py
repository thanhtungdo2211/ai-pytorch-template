import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from collections import Counter
import re

class SkipGramDataset(Dataset):
    def __init__(self, text, vocab_size=5000, window_size=2):
        # Preprocess text
        words = re.findall(r'\w+', text.lower())
        
        # Build vocabulary
        word_counts = Counter(words)
        self.word2idx = {word: idx for idx, (word, _) in 
                         enumerate(sorted(word_counts.items(), 
                                        key=lambda x: x[1], reverse=True)[:vocab_size])}
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        self.vocab_size = len(self.word2idx)
        
        # Convert to indices, filter unknown words
        self.indices = [self.word2idx[w] for w in words if w in self.word2idx]
        self.window_size = window_size
        self.pairs = self._generate_pairs()
    
    def _generate_pairs(self):
        pairs = []
        for i, center_word in enumerate(self.indices):
            # Get context words within window
            start = max(0, i - self.window_size)
            end = min(len(self.indices), i + self.window_size + 1)
            
            for j in range(start, end):
                if i != j:  # Skip the center word itself
                    pairs.append((center_word, self.indices[j]))
        return pairs
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        center, context = self.pairs[idx]
        return torch.tensor(center), torch.tensor(context)

class SkipGram(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(SkipGram, self).__init__()
        self.center_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        # Initialize embeddings
        nn.init.uniform_(self.center_embeddings.weight, -0.5/embedding_dim, 0.5/embedding_dim)
        nn.init.uniform_(self.context_embeddings.weight, -0.5/embedding_dim, 0.5/embedding_dim)
    
    def forward(self, center, context):
        center_emb = self.center_embeddings(center)  # (batch_size, embedding_dim)
        context_emb = self.context_embeddings(context)  # (batch_size, embedding_dim)
        
        # Compute dot product
        dot_product = torch.sum(center_emb * context_emb, dim=1)  # (batch_size,)
        return dot_product

# Training
def train_skipgram(text, embedding_dim=100, batch_size=64, epochs=5, learning_rate=0.01):
    # Create dataset
    dataset = SkipGramDataset(text, vocab_size=5000, window_size=2)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SkipGram(dataset.vocab_size, embedding_dim).to(device)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    
    # Loss function for negative sampling (simplified using binary cross-entropy)
    # In practice, you'd implement negative sampling separately
    loss_fn = nn.BCEWithLogitsLoss()
    
    # Training loop
    for epoch in range(epochs):
        total_loss = 0
        for center, context in dataloader:
            center, context = center.to(device), context.to(device)
            
            # Forward pass
            output = model(center, context)
            
            # Create positive labels (1 for actual context pairs)
            labels = torch.ones(output.shape[0]).to(device)
            
            # Compute loss
            loss = loss_fn(output, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    return model, dataset

# Example usage
if __name__ == "__main__":
    sample_text = """
    the quick brown fox jumps over the lazy dog
    machine learning is a subset of artificial intelligence
    natural language processing helps computers understand human language
    word embeddings capture semantic meaning of words
    skip gram is a popular word embedding technique
    """ * 10  # Repeat to have more training data
    
    model, dataset = train_skipgram(sample_text, embedding_dim=100, epochs=5)
    
    # Extract word embeddings
    embeddings = model.center_embeddings.weight.data
    
    # Get embedding for a specific word
    word = "machine"
    if word in dataset.word2idx:
        word_idx = dataset.word2idx[word]
        word_embedding = embeddings[word_idx]
        print(f"\nEmbedding for '{word}': {word_embedding[:10]}...")  # Print first 10 dims