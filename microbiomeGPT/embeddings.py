"""
Microbiome Embedding Utilities
============================
This module provides utilities for generating and visualizing embeddings
from the MicrobiomeTransformer model.

Functions:
- get_embeddings: Extract embeddings from the model
- reduce_dimensions: Reduce embedding dimensionality using PCA or t-SNE
- plot_embeddings: Visualize reduced embeddings with labels
"""

import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

def get_embeddings(model, data_loader, device):
    """
    Extract embeddings from the model for all samples in the data loader.
    
    Args:
        model: The trained model
        data_loader: DataLoader containing the samples
        device: Device to run the model on
        
    Returns:
        torch.Tensor: Concatenated embeddings for all samples
    """
    model.eval()
    all_embeddings = []

    with torch.no_grad():
        for batch in data_loader:
            taxa_padded, expressions_padded = batch
            taxa_padded = taxa_padded.to(device)
            expressions_padded = expressions_padded.to(device)
            
            # Get embeddings and apply mean pooling
            embeddings = model.get_embeddings(taxa_padded, expressions_padded)
            embeddings = embeddings.mean(dim=1)
            all_embeddings.append(embeddings.cpu())

    return torch.cat(all_embeddings, dim=0)

def reduce_dimensions(embeddings, method='pca', n_components=2):
    """
    Reduce dimensionality of embeddings using PCA or t-SNE.
    
    Args:
        embeddings: High-dimensional embeddings
        method: 'pca' or 'tsne'
        n_components: Number of dimensions in output
        
    Returns:
        numpy.ndarray: Reduced-dimensional embeddings
    """
    if method == 'pca':
        reducer = PCA(n_components=n_components)
    elif method == 'tsne':
        reducer = TSNE(n_components=n_components, random_state=42)
    else:
        raise ValueError("Method must be 'pca' or 'tsne'")

    reduced_embeddings = reducer.fit_transform(embeddings)
    return reduced_embeddings

def plot_embeddings(reduced_embeddings, study_names, title):
    """
    Create a scatter plot of reduced embeddings with labels.
    
    Args:
        reduced_embeddings: Low-dimensional embeddings
        study_names: Labels for each point
        title: Plot title
    """
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        x=reduced_embeddings[:, 0], 
        y=reduced_embeddings[:, 1], 
        hue=study_names, 
        palette='deep'
    )
    
    plt.title(title)
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.legend(
        bbox_to_anchor=(1.05, 1), 
        loc='upper left', 
        ncol=1 + len(set(study_names)) // 20
    )
    plt.tight_layout()
    plt.show()
