"""
Microbiome Analysis Utilities (Part 1)
====================================
Core utilities for microbiome data processing and model training.
Includes dataset classes and training functions.

Classes:
- MicrobiomeDataset: Base dataset for sequence modeling
- MicrobiomeRegressionDataset: Dataset for regression tasks
- MicrobiomeClassificationDataset: Dataset for classification tasks

Functions:
- Training loops for different model types
- Collate functions for batching
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# =====================
# Dataset Classes
# =====================

class MicrobiomeDataset(Dataset):
    """Dataset for sequence modeling of microbiome data."""
    
    def __init__(self, data, taxa_to_idx, max_features=None):
        """
        Args:
            data (pd.DataFrame): Input microbiome abundance data
            taxa_to_idx (dict): Mapping from taxa names to indices
            max_features (int, optional): Maximum number of features to use
        """
        self.data = data
        self.taxa_to_idx = taxa_to_idx
        self.max_features = max_features if max_features is not None else len(data.columns)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get non-zero abundances sorted by value
        row = self.data.iloc[idx]
        non_zero = row[row != 0].sort_values(ascending=False)
        non_zero = non_zero.iloc[:self.max_features]

        # Create sequence with special tokens
        taxa = ['<START>'] + list(non_zero.index) + ['<END>']
        expressions = [0] + list(non_zero.values) + [0]

        # Convert to tensors
        taxa_indices = [self.taxa_to_idx[t] for t in taxa]
        taxa_tensor = torch.tensor(taxa_indices, dtype=torch.long)
        expressions_tensor = torch.tensor(expressions, dtype=torch.float)

        return taxa_tensor, expressions_tensor

class MicrobiomeRegressionDataset(Dataset):
    """Dataset for regression tasks using microbiome data."""
    
    def __init__(self, data, taxa_to_idx, target_column, max_features=None):
        """
        Args:
            data (pd.DataFrame): Input data with target column
            taxa_to_idx (dict): Mapping from taxa names to indices
            target_column (str): Name of regression target column
            max_features (int, optional): Maximum number of features to use
        """
        self.data = data.drop(columns=[target_column])
        self.targets = data[target_column]
        self.taxa_to_idx = taxa_to_idx
        self.max_features = max_features if max_features is not None else len(self.data.columns)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        non_zero = row[row != 0].sort_values(ascending=False)
        non_zero = non_zero.iloc[:self.max_features]

        taxa = ['<START>'] + list(non_zero.index) + ['<END>']
        expressions = [0] + list(non_zero.values) + [0]

        taxa_indices = [self.taxa_to_idx[t] for t in taxa]
        taxa_tensor = torch.tensor(taxa_indices, dtype=torch.long)
        expressions_tensor = torch.tensor(expressions, dtype=torch.float)
        target_tensor = torch.tensor(self.targets.iloc[idx], dtype=torch.float)

        return taxa_tensor, expressions_tensor, target_tensor

class MicrobiomeClassificationDataset(Dataset):
    """Dataset for classification tasks using microbiome data."""
    
    def __init__(self, data, taxa_to_idx, target_column, max_features=None):
        """
        Args:
            data (pd.DataFrame): Input data with target column
            taxa_to_idx (dict): Mapping from taxa names to indices
            target_column (str): Name of classification target column
            max_features (int, optional): Maximum number of features to use
        """
        self.data = data.drop(columns=[target_column])
        self.targets = data[target_column]
        self.taxa_to_idx = taxa_to_idx
        self.max_features = max_features if max_features is not None else len(self.data.columns)
        
        self.label_encoder = LabelEncoder()
        self.encoded_targets = self.label_encoder.fit_transform(self.targets)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        non_zero = row[row != 0].sort_values(ascending=False)
        non_zero = non_zero.iloc[:self.max_features]

        taxa = ['<START>'] + list(non_zero.index) + ['<END>']
        expressions = [0] + list(non_zero.values) + [0]

        taxa_indices = [self.taxa_to_idx[t] for t in taxa]
        taxa_tensor = torch.tensor(taxa_indices, dtype=torch.long)
        expressions_tensor = torch.tensor(expressions, dtype=torch.float)
        target_tensor = torch.tensor(self.encoded_targets[idx], dtype=torch.long)

        return taxa_tensor, expressions_tensor, target_tensor

# =====================
# Collate Functions
# =====================

def collate_fn(batch):
    """Collate function for sequence data."""
    taxa, expressions = zip(*batch)
    taxa_padded = pad_sequence(taxa, batch_first=True, padding_value=0)
    expressions_padded = pad_sequence(expressions, batch_first=True, padding_value=0)
    return taxa_padded, expressions_padded

def regression_collate_fn(batch):
    """Collate function for regression data."""
    taxa, expressions, targets = zip(*batch)
    taxa_padded = pad_sequence(taxa, batch_first=True, padding_value=0)
    expressions_padded = pad_sequence(expressions, batch_first=True, padding_value=0)
    targets = torch.stack(targets)
    return taxa_padded, expressions_padded, targets

def classification_collate_fn(batch):
    """Collate function for classification data."""
    taxa, expressions, targets = zip(*batch)
    taxa_padded = pad_sequence(taxa, batch_first=True, padding_value=0)
    expressions_padded = pad_sequence(expressions, batch_first=True, padding_value=0)
    targets = torch.stack(targets)
    return taxa_padded, expressions_padded, targets

# =====================
# Training Functions
# =====================

def train(model, train_loader, val_loader, epochs, lr, device):
    """Train the base sequence model."""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=0)
    taxa_criterion = nn.CrossEntropyLoss(ignore_index=0)
    expr_criterion = nn.MSELoss()
    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        for taxa, expressions in train_loader:
            taxa, expressions = taxa.to(device), expressions.to(device)
            optimizer.zero_grad()

            src_key_padding_mask = (taxa == 0)
            taxa_logits, expr_pred = model(taxa[:, :-1], expressions[:, :-1], src_key_padding_mask[:, :-1])

            taxa_loss = taxa_criterion(taxa_logits.reshape(-1, taxa_logits.size(-1)), taxa[:, 1:].reshape(-1))
            expr_loss = expr_criterion(expr_pred.reshape(-1), expressions[:, 1:].reshape(-1))
            loss = 50 * taxa_loss + expr_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        val_loss = validate(model, val_loader, taxa_criterion, expr_criterion, device)
        print(f'Epoch {epoch+1}, Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss:.4f}')
        
        scheduler.step()
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')

def train_regression(model, train_loader, val_loader, epochs, lr, device):
    """Train the regression model."""
    optimizer = optim.SGD(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=0)
    criterion = nn.L1Loss()
    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        for taxa, expressions, targets in train_loader:
            taxa = taxa.to(device)
            expressions = expressions.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            src_key_padding_mask = (taxa == 0)
            predictions = model(taxa, expressions, src_key_padding_mask)
            loss = criterion(predictions, targets)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        val_loss = validate_regression(model, val_loader, criterion, device)
        print(f'Epoch {epoch+1}, Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss:.4f}')
        
        scheduler.step()
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_regression_model.pth')

def train_classification(model, train_loader, val_loader, epochs, lr, device):
    """Train the classification model."""
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=0)
    criterion = nn.CrossEntropyLoss()
    best_val_accuracy = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        for taxa, expressions, targets in train_loader:
            taxa = taxa.to(device)
            expressions = expressions.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            src_key_padding_mask = (taxa == 0)
            predictions = model(taxa, expressions, src_key_padding_mask)
            loss = criterion(predictions, targets)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        val_accuracy = validate_classification(model, val_loader, device)
        print(f'Epoch {epoch+1}, Train Loss: {train_loss/len(train_loader):.4f}, Val Accuracy: {val_accuracy:.4f}')
        
        scheduler.step()
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), 'best_classification_model.pth')

"""
Microbiome Analysis Utilities (Part 2)
====================================
Validation and visualization utilities for microbiome models.
Includes functions for model evaluation and result visualization.

Functions:
- Validation functions for different model types
- Visualization utilities for model predictions
- Performance analysis tools
"""

import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression

# =====================
# Validation Functions
# =====================

def validate(model, val_loader, taxa_criterion, expr_criterion, device):
    """Validate the base sequence model."""
    model.eval()
    val_loss = 0
    
    with torch.no_grad():
        for taxa, expressions in val_loader:
            taxa = taxa.to(device)
            expressions = expressions.to(device)
            
            src_key_padding_mask = (taxa == 0)
            taxa_logits, expr_pred = model(taxa[:, :-1], expressions[:, :-1], src_key_padding_mask[:, :-1])
            
            taxa_loss = taxa_criterion(taxa_logits.reshape(-1, taxa_logits.size(-1)), taxa[:, 1:].reshape(-1))
            expr_loss = expr_criterion(expr_pred.reshape(-1), expressions[:, 1:].reshape(-1))
            loss = 50 * taxa_loss + expr_loss
            
            val_loss += loss.item()
    
    return val_loss / len(val_loader)

def validate_regression(model, val_loader, criterion, device, plot_reg=False):
    """
    Validate the regression model.
    
    Args:
        model: Model to validate
        val_loader: Validation data loader
        criterion: Loss criterion
        device: Computation device
        plot_reg: Whether to plot regression results
    """
    model.eval()
    val_loss = 0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for taxa, expressions, targets in val_loader:
            taxa = taxa.to(device)
            expressions = expressions.to(device)
            targets = targets.to(device)

            src_key_padding_mask = (taxa == 0)
            predictions = model(taxa, expressions, src_key_padding_mask)
            loss = criterion(predictions, targets)
            val_loss += loss.item()

            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    if plot_reg:
        plot_regression_results(np.array(all_predictions), np.array(all_targets))

    return val_loss / len(val_loader)

def validate_classification(model, val_loader, device, plot_conf_matrix=False):
    """
    Validate the classification model.
    
    Args:
        model: Model to validate
        val_loader: Validation data loader
        device: Computation device
        plot_conf_matrix: Whether to plot confusion matrix
    """
    model.eval()
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for taxa, expressions, targets in val_loader:
            taxa = taxa.to(device)
            expressions = expressions.to(device)
            targets = targets.to(device)

            src_key_padding_mask = (taxa == 0)
            predictions = model(taxa, expressions, src_key_padding_mask)
            all_predictions.extend(predictions.argmax(dim=1).cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    accuracy = accuracy_score(all_targets, all_predictions)
    
    if plot_conf_matrix:
        print("\nClassification Report:")
        print(classification_report(all_targets, all_predictions))

    return accuracy

# =====================
# Visualization Functions
# =====================

def plot_regression_results(predictions, targets):
    """
    Plot regression results with trend line and metrics.
    
    Args:
        predictions: Model predictions
        targets: True target values
    """
    plt.figure(figsize=(10, 10))
    plt.scatter(targets, predictions, alpha=0.5)

    # Fit and plot regression line
    model = LinearRegression()
    model.fit(targets.reshape(-1, 1), predictions)
    predicted_line = model.predict(targets.reshape(-1, 1))
    plt.plot(targets, predicted_line, color='blue', linewidth=2, label='Best Fit Line')

    # Plot identity line
    plt.plot([min(targets), max(targets)], [min(targets), max(targets)], 
             'r--', lw=2, label='y = x')

    # Calculate and display metrics
    mae = mean_absolute_error(targets, predictions)
    r_squared = r2_score(targets, predictions)
    plt.text(0.05, 0.95, 
             f'R-squared = {r_squared:.4f}\nMAE = {mae:.4f}',
             transform=plt.gca().transAxes, 
             fontsize=12, 
             verticalalignment='top')

    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Predicted vs Actual Values')
    plt.legend()
    plt.tight_layout()
    plt.show()

def inspect_predictions(model, test_loader, vocab, device):
    """
    Inspect model predictions in detail.
    
    Args:
        model: Model to evaluate
        test_loader: Test data loader
        vocab: Vocabulary mapping
        device: Computation device
    """
    model.eval()
    all_taxa_true = []
    all_taxa_pred = []
    all_expr_true = []
    all_expr_pred = []

    with torch.no_grad():
        for taxa, expressions in test_loader:
            taxa = taxa.to(device)
            expressions = expressions.to(device)

            src_key_padding_mask = (taxa == 0)
            taxa_logits, expr_pred = model(taxa[:, :-1], expressions[:, :-1], src_key_padding_mask[:, :-1])

            taxa_pred = taxa_logits.argmax(dim=-1)

            # Collect non-padding predictions
            mask = taxa[:, 1:] != 0
            all_taxa_true.extend(taxa[:, 1:][mask].cpu().numpy())
            all_taxa_pred.extend(taxa_pred[mask].cpu().numpy())
            all_expr_true.extend(expressions[:, 1:][mask].cpu().numpy())
            all_expr_pred.extend(expr_pred[mask].cpu().numpy())

    # Calculate and display metrics
    taxa_accuracy = accuracy_score(all_taxa_true, all_taxa_pred)
    expr_mse = F.mse_loss(torch.tensor(all_expr_pred), torch.tensor(all_expr_true))

    print(f"Taxa Prediction Accuracy: {taxa_accuracy:.4f}")
    print(f"Expression Prediction MSE: {expr_mse:.4f}")

    # Visualization
    plt.figure(figsize=(15, 5))

    # Taxa confusion matrix
    plt.subplot(1, 2, 1)
    cm = pd.crosstab(pd.Series(all_taxa_true, name='Actual'),
                     pd.Series(all_taxa_pred, name='Predicted'))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues')
    plt.title('Taxa Prediction Confusion Matrix')

    # Expression scatter plot
    plt.subplot(1, 2, 2)
    plt.scatter(all_expr_true, all_expr_pred, alpha=0.5)
    plt.plot([min(all_expr_true), max(all_expr_true)], 
             [min(all_expr_true), max(all_expr_true)], 'r--')
    plt.xlabel('True Expression')
    plt.ylabel('Predicted Expression')
    plt.title('Expression Prediction')

    plt.tight_layout()
    plt.show()

def visualize_step_by_step_predictions(model, test_loader, vocab, device, n_examples=5, k_tokens=5):
    """
    Visualize model predictions step by step.
    
    Args:
        model: Model to evaluate
        test_loader: Test data loader
        vocab: Vocabulary mapping
        device: Computation device
        n_examples: Number of examples to visualize
        k_tokens: Number of tokens to show per example
    """
    model.eval()

    for i, (taxa, expressions) in enumerate(test_loader):
        if i >= n_examples:
            break

        taxa = taxa.to(device)
        expressions = expressions.to(device)

        print(f"\nExample {i+1}:")
        print("Initial sequence:")
        for j in range(min(k_tokens, len(taxa[0]))):
            print(f"{vocab[taxa[0][j].item()]}: {expressions[0][j].item():.4f}")

        print("\nStep-by-step predictions:")
        for step in range(k_tokens, len(taxa[0])):
            src_key_padding_mask = (taxa[0][:step] == 0).unsqueeze(0)
            taxa_input = taxa[0][:step].unsqueeze(0)
            expr_input = expressions[0][:step].unsqueeze(0)

            with torch.no_grad():
                taxa_logits, expr_pred = model(taxa_input, expr_input, src_key_padding_mask)

            predicted_taxa = taxa_logits[0, -1].argmax().item()
            predicted_expr = expr_pred[0, -1].item()

            print(f"Step {step}:")
            print(f"Predicted: {vocab[predicted_taxa]}: {predicted_expr:.4f}")
            print(f"Actual: {vocab[taxa[0][step].item()]}: {expressions[0][step].item():.4f}")
            print()

        print("-" * 50)
