"""
Author: Sufyaldy
"""
import torch
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def evaluate_model(model, dataloader):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for sequences, labels in dataloader:
            predictions, _ = model(sequences)
            all_preds.extend(predictions.squeeze().numpy())
            all_labels.extend(labels.numpy())
            
    rmse = mean_squared_error(all_labels, all_preds, squared=False)
    mae = mean_absolute_error(all_labels, all_preds)
    r2 = r2_score(all_labels, all_preds)
    
    print(f"Test RMSE: {rmse:.2f}")
    print(f"Test MAE: {mae:.2f}")
    print(f"Test R2: {r2:.2f}")
    
    return rmse, mae, r2

if __name__ == "__main__":
    print("Evaluating model...")
    # Example: evaluate_model(model, test_dataloader)
