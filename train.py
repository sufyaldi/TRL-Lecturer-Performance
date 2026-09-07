import torch
import torch.nn as nn
import torch.optim as optim
from models import LecturerLSTM

def train_model(model, dataloader, epochs=50, lr=0.001):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for sequences, labels in dataloader:
            optimizer.zero_grad()
            
            predictions, _ = model(sequences)
            loss = criterion(predictions.squeeze(), labels)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{epochs} | Loss: {epoch_loss/len(dataloader):.4f}')
            
if __name__ == "__main__":
    print("Training sequence model...")
    # Example: model = LecturerLSTM(vocab_size=10, embedding_dim=64, hidden_dim=128)
    # train_model(model, train_dataloader)
