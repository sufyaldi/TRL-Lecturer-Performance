import torch
import torch.nn as nn

class LecturerLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim=1):
        super(LecturerLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        embedded = self.embedding(x)
        output, (hidden, cell) = self.lstm(embedded)
        # Use the final hidden state for prediction
        out = self.fc(hidden[-1])
        return out, hidden[-1]

class TransformerAutoencoder(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, output_dim=1):
        super(TransformerAutoencoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, output_dim)
        
    def forward(self, x):
        embedded = self.embedding(x)
        encoded = self.transformer_encoder(embedded)
        # Global average pooling
        pooled = encoded.mean(dim=1)
        out = self.fc(pooled)
        return out, pooled
