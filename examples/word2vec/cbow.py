import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from collections import Counter
import re

class CBOWDataset(Dataset):
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
        """Generate (context_words, target_word) pairs"""
        pairs = []
        for i, target_word in enumerate(self.indices):
            # Get context words within window
            start = max(0, i - self.window_size)
            end = min(len(self.indices), i + self.window_size + 1)
            
            context = []
            for j in range(start, end):
                if i != j:  # Skip the target word itself
                    context.append(self.indices[j])
            
            if context:  # Only add if we have context words
                pairs.append((context, target_word))
        
        return pairs
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        context, target = self.pairs[idx]
        # Pad context to fixed size (window_size * 2)
        max_context_size = self.window_size * 2
        context = context + [0] * (max_context_size - len(context))  # Pad with 0s
        context = context[:max_context_size]  # Truncate if needed
        
        return torch.tensor(context, dtype=torch.long), torch.tensor(target, dtype=torch.long)

class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        """
        CBOW model that predicts target word from context words
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of word embeddings
        """
        super(CBOW, self).__init__()
        # Input embeddings (for context words)
        self.input_embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        # Output embeddings (for target word)
        self.output_embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Initialize embeddings
        nn.init.uniform_(self.input_embeddings.weight, -0.5/embedding_dim, 0.5/embedding_dim)
        nn.init.uniform_(self.output_embeddings.weight, -0.5/embedding_dim, 0.5/embedding_dim)
    
    def forward(self, context_words):
        """
        Forward pass
        
        Args:
            context_words: (batch_size, window_size * 2)
        
        Returns:
            output logits: (batch_size, vocab_size)
        """
        # Get embeddings for context words
        context_embeds = self.input_embeddings(context_words)  # (batch_size, context_len, embedding_dim)
        
        # Average the context embeddings
        context_avg = torch.mean(context_embeds, dim=1)  # (batch_size, embedding_dim)
        
        # Compute scores for all vocabulary words
        # Using matrix multiplication: context_avg @ output_embeddings.weight.T
        output = torch.matmul(context_avg, self.output_embeddings.weight.T)  # (batch_size, vocab_size)
        
        return output

class CBOWNegativeSampling(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        """CBOW with negative sampling (more efficient)"""
        super(CBOWNegativeSampling, self).__init__()
        self.input_embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.output_embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        nn.init.uniform_(self.input_embeddings.weight, -0.5/embedding_dim, 0.5/embedding_dim)
        nn.init.uniform_(self.output_embeddings.weight, -0.5/embedding_dim, 0.5/embedding_dim)
    
    def forward(self, context_words, target_word, negative_samples=None):
        """
        Args:
            context_words: (batch_size, context_len)
            target_word: (batch_size,)
            negative_samples: (batch_size, num_negatives)
        
        Returns:
            positive_score, negative_scores
        """
        # Average context embeddings
        context_embeds = self.input_embeddings(context_words)
        context_avg = torch.mean(context_embeds, dim=1)  # (batch_size, embedding_dim)
        
        # Positive sample score
        target_embeds = self.output_embeddings(target_word)  # (batch_size, embedding_dim)
        positive_score = torch.sum(context_avg * target_embeds, dim=1)  # (batch_size,)
        
        # Negative samples score
        if negative_samples is not None:
            negative_embeds = self.output_embeddings(negative_samples)  # (batch_size, num_neg, embedding_dim)
            negative_scores = torch.matmul(context_avg.unsqueeze(1), 
                                          negative_embeds.transpose(1, 2))  # (batch_size, 1, num_neg)
            negative_scores = negative_scores.squeeze(1)  # (batch_size, num_neg)
            return positive_score, negative_scores
        
        return positive_score

def train_cbow(text, embedding_dim=100, batch_size=32, epochs=10, learning_rate=0.01):
    """Train CBOW model"""
    # Create dataset
    dataset = CBOWDataset(text, vocab_size=5000, window_size=2)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CBOW(dataset.vocab_size, embedding_dim).to(device)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    
    print(f"Vocabulary size: {dataset.vocab_size}")
    print(f"Total pairs: {len(dataset)}")
    print(f"Device: {device}\n")
    
    # Training loop
    for epoch in range(epochs):
        total_loss = 0
        for context, target in dataloader:
            context, target = context.to(device), target.to(device)
            
            # Forward pass
            logits = model(context)  # (batch_size, vocab_size)
            loss = loss_fn(logits, target)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    return model, dataset

def get_similar_words(model, dataset, word, top_k=5):
    """Find similar words using cosine similarity"""
    if word not in dataset.word2idx:
        print(f"Word '{word}' not in vocabulary")
        return
    
    word_idx = dataset.word2idx[word]
    word_embed = model.output_embeddings.weight[word_idx].unsqueeze(0)  # (1, embedding_dim)
    
    # Compute cosine similarity with all words
    all_embeds = model.output_embeddings.weight  # (vocab_size, embedding_dim)
    similarities = torch.nn.functional.cosine_similarity(word_embed, all_embeds)
    
    # Get top-k similar words (excluding the word itself)
    top_indices = torch.topk(similarities, top_k + 1)[1]
    
    print(f"\nTop {top_k} similar words to '{word}':")
    for i, idx in enumerate(top_indices):
        if idx != word_idx:
            similar_word = dataset.idx2word[idx.item()]
            similarity_score = similarities[idx].item()
            print(f"  {i}. {similar_word}: {similarity_score:.4f}")

# Example usage
if __name__ == "__main__":
    sample_text = """
    the quick brown fox jumps over the lazy dog
    machine learning is a subset of artificial intelligence
    natural language processing helps computers understand human language
    word embeddings capture semantic meaning of words
    cbow predicts target words from context words
    deep learning models can learn representations
    neural networks are inspired by biological neurons
    word vectors represent words in vector space
    semantic relationships are captured by embeddings
    context window determines surrounding words
    """ * 20  # Repeat to have more training data
    
    # Train model
    model, dataset = train_cbow(sample_text, embedding_dim=100, batch_size=16, epochs=15)
    
    # Get word embeddings
    embeddings = model.output_embeddings.weight.data
    
    # Find similar words
    get_similar_words(model, dataset, "learning", top_k=5)
    get_similar_words(model, dataset, "neural", top_k=5)
    
    # Get embedding for a specific word
    word = "machine"
    if word in dataset.word2idx:
        word_idx = dataset.word2idx[word]
        word_embedding = embeddings[word_idx]
        print(f"\nEmbedding for '{word}': {word_embedding[:10]}...")  # Print first 10 dims