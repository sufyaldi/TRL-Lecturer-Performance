import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

def plot_latent_space(latent_vectors, labels):
    """
    Projects latent style vectors to 2D using t-SNE and plots them.
    """
    tsne = TSNE(n_components=2, random_state=42)
    latent_2d = tsne.fit_transform(latent_vectors)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(latent_2d[:, 0], latent_2d[:, 1], c=labels, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, label='Lecturer Performance Score')
    plt.title("t-SNE Projection of Teaching Style Latent Space")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()

def plot_actual_vs_predicted(actual, predicted, model_name="Model"):
    plt.figure(figsize=(8, 8))
    plt.scatter(actual, predicted, alpha=0.6)
    plt.plot([min(actual), max(actual)], [min(actual), max(actual)], 'r--')
    plt.xlabel("Actual Score")
    plt.ylabel("Predicted Score")
    plt.title(f"{model_name} - Actual vs Predicted LPS")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()
