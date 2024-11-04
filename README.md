# MicrobiomeGPT

A deep learning framework for microbiome data analysis using transformer-based architectures. This toolkit provides methods for embedding generation, classification, and regression tasks on microbiome abundance data.

## Features

- Pre-trained transformer model for microbiome data representation
- Fine-tuning capabilities for:
  - Regression tasks (e.g., predicting age from microbiome composition)
  - Classification tasks (e.g., disease state prediction)
- Data preprocessing utilities
- Embedding visualization tools
- Performance evaluation metrics

## Installation

```bash
# Installation instructions coming soon
pip install microbiomeGPT
```

## Quick Start

### Loading and Preprocessing Data

```python
import pandas as pd
from microbiomeGPT.preprocessing import matrix_rclr

# Load data
metadata = pd.read_csv('data/sampleMetadata.csv', index_col='sample_id')
raw_df = pd.read_csv('data/relative_abundance.csv', index_col=0)

# Preprocess using robust centered log-ratio transformation
df = pd.DataFrame(
    matrix_rclr(raw_df.iloc[:,3:].fillna(0)), 
    index=raw_df.index, 
    columns=raw_df.iloc[:,3:].columns
).fillna(0)
```

### Training a Base Transformer Model

```python
from microbiomeGPT.model import MicrobiomeTransformer
from microbiomeGPT import utils

# Initialize model
model = MicrobiomeTransformer(
    vocab_size=len(vocab),
    d_model=2048,
    nhead=8,
    num_encoder_layers=1,
    max_features=128,
    taxa_to_idx=taxa_to_idx
).to(device)

# Train model
utils.train(model, train_loader, test_loader, epochs=100, lr=0.00005, device=device)
```

### Regression Example: Age Prediction

```python
from microbiomeGPT.model import MicrobiomeTransformerRegression

# Create regression model using pre-trained transformer
regression_model = MicrobiomeTransformerRegression(
    pretrained_model,
    freeze_pretrained=True
).to(device)

# Train regression model
utils.train_regression(
    regression_model,
    train_loader,
    test_loader,
    epochs=1000,
    lr=0.005,
    device=device
)
```

### Classification Example: Health Status Prediction

```python
from microbiomeGPT.model import MicrobiomeTransformerClassification

# Create classification model
classification_model = MicrobiomeTransformerClassification(
    pretrained_model,
    num_classes,
    freeze_pretrained=True
).to(device)

# Train classification model
utils.train_classification(
    classification_model,
    train_loader,
    test_loader,
    epochs=25,
    lr=0.000005,
    device=device
)
```

### Visualizing Embeddings

```python
from microbiomeGPT import embeddings

# Extract embeddings
all_embeddings = embeddings.get_embeddings(model, full_loader, device)

# Reduce dimensions and visualize
pca_embeddings = embeddings.reduce_dimensions(all_embeddings.numpy(), method='pca')
tsne_embeddings = embeddings.reduce_dimensions(all_embeddings.numpy(), method='tsne')

# Plot embeddings
embeddings.plot_embeddings(pca_embeddings, metadata_labels, 'PCA of Sample Embeddings')
embeddings.plot_embeddings(tsne_embeddings, metadata_labels, 't-SNE of Sample Embeddings')
```

## Performance

### Regression Performance
The model achieves competitive performance on age prediction tasks when compared to traditional methods like Random Forest regression:
- Mean Absolute Error (MAE)
- R² Score

### Classification Performance
For health status classification, the model demonstrates strong predictive capability with:
- Accuracy metrics
- Confusion matrix visualization

## Requirements

- Python 3.8+
- PyTorch
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- gemelli
