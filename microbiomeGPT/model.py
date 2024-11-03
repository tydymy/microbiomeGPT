"""
Microbiome Transformer Models
============================
This module contains transformer-based models for analyzing microbiome data.
It includes models for sequence prediction, regression, and classification tasks.

Main components:
- MicrobiomeTransformer: Base transformer model for microbiome sequence modeling
- MicrobiomeTransformerRegression: Model for regression tasks using microbiome data
- MicrobiomeTransformerClassification: Model for classification tasks using microbiome data
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    """
    Implements positional encoding for transformer models.
    Adds position-dependent features to the input embeddings.
    """
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """Add positional encoding to input tensor."""
        return x + self.pe[:, :x.size(1)]

class MicrobiomeTransformer(nn.Module):
    """
    Base transformer model for microbiome sequence analysis.
    Combines taxa embeddings with expression values and processes them through a transformer.
    
    Args:
        vocab_size (int): Size of the taxonomy vocabulary
        d_model (int): Dimension of the model's embeddings
        nhead (int): Number of attention heads
        num_encoder_layers (int): Number of transformer encoder layers
        taxa_to_idx (dict): Mapping from taxonomy names to indices
        max_features (int): Maximum number of features to process
        max_len (int): Maximum sequence length
    """
    def __init__(self, vocab_size, d_model, nhead, num_encoder_layers, taxa_to_idx, max_features=128, max_len=5000):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_features = max_features
        self.idx_to_taxa = {idx: taxa for taxa, idx in taxa_to_idx.items()}
        
        # Embedding layers
        self.taxa_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.embedding_dropout = nn.Dropout(0.1)
        self.expression_proj = nn.Linear(1, d_model)
        self.expression_scale = nn.Parameter(torch.ones(1))
        self.pos_encoder = PositionalEncoding(d_model, max_len)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, 
            dim_feedforward=512, 
            dropout=0.2,
            batch_first=True,
            norm_first=True  # Apply normalization before attention and feedforward
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # Output layers
        self.fc_taxa = nn.Linear(d_model, vocab_size)
        self.fc_expression = nn.Linear(d_model, 1)

    def forward(self, taxa, expressions, src_key_padding_mask=None, return_embeddings=False, input_embedding=None):
        """
        Forward pass of the model.
        
        Args:
            taxa (torch.Tensor): Input taxonomy indices
            expressions (torch.Tensor): Input expression values
            src_key_padding_mask (torch.Tensor): Mask for padding tokens
            return_embeddings (bool): Whether to return embeddings instead of predictions
            input_embedding (torch.Tensor): Optional pre-computed embeddings
            
        Returns:
            tuple: (taxa_logits, expr_pred) or embeddings if return_embeddings=True
        """
        if input_embedding is None:
            # Process input sequences
            taxa_emb = self.embedding_dropout(self.taxa_embedding(taxa))
            expr_emb = self.expression_scale * self.expression_proj(expressions.unsqueeze(-1))
            x = taxa_emb + expr_emb

            if src_key_padding_mask is None:
                src_key_padding_mask = (taxa == 0)
            
            x = x.masked_fill(src_key_padding_mask.unsqueeze(-1), 0.0)
            x = x + self.pos_encoder(x)
        else:
            x = input_embedding
            if src_key_padding_mask is None:
                src_key_padding_mask = torch.zeros(x.size(0), x.size(1), dtype=torch.bool, device=x.device)
        
        # Apply transformer
        output = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        if return_embeddings:
            return output

        # Generate predictions
        taxa_logits = self.fc_taxa(output)
        expr_pred = self.fc_expression(output).squeeze(-1)

        return taxa_logits, expr_pred

    def get_embeddings(self, taxa, expressions, src_key_padding_mask=None):
        """Get embeddings for input sequences."""
        return self.forward(taxa, expressions, src_key_padding_mask, return_embeddings=True)

    def decode_embedding(self, embedding):
        """Decode embeddings back to taxa predictions and expression values."""
        taxa_logits = self.fc_taxa(embedding)
        expr_pred = self.fc_expression(embedding)

        # Get the top max_features predictions
        top_taxa_values, top_taxa_indices = torch.topk(taxa_logits, k=self.max_features, dim=-1)
        taxa_names = [[self.idx_to_taxa[idx.item()] for idx in sample] for sample in top_taxa_indices]
        expr_values = expr_pred

        return taxa_names, expr_values

class MicrobiomeTransformerRegression(nn.Module):
    """
    Regression model based on the MicrobiomeTransformer.
    Uses attention mechanism to process sequence data for regression tasks.
    
    Args:
        pretrained_model (MicrobiomeTransformer): Pre-trained base model
        freeze_pretrained (bool): Whether to freeze the pre-trained model's parameters
    """
    def __init__(self, pretrained_model, freeze_pretrained=True):
        super().__init__()
        self.pretrained_model = pretrained_model
        self.attention = nn.MultiheadAttention(
            pretrained_model.fc_taxa.in_features, 
            num_heads=8, 
            batch_first=True
        )
        
        # Regression head
        self.regression_head = nn.Sequential(
            nn.Linear(pretrained_model.fc_taxa.in_features, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        if freeze_pretrained:
            self._freeze_pretrained_model()

    def _freeze_pretrained_model(self):
        """Freeze pre-trained model parameters."""
        for param in self.pretrained_model.parameters():
            param.requires_grad = False

    def forward(self, taxa, expressions, src_key_padding_mask=None):
        """Forward pass for regression prediction."""
        # Get embeddings from pre-trained model
        taxa_emb = self.pretrained_model.taxa_embedding(taxa)
        expr_emb = self.pretrained_model.expression_proj(expressions.unsqueeze(-1))
        x = taxa_emb + expr_emb
        x = x + self.pretrained_model.pos_encoder(x)

        if src_key_padding_mask is None:
            src_key_padding_mask = (taxa == 0)

        # Apply transformer and attention
        output = self.pretrained_model.transformer(x, src_key_padding_mask=src_key_padding_mask)
        query = output.mean(dim=1, keepdim=True)
        context, _ = self.attention(query, output, output, key_padding_mask=src_key_padding_mask)
        context = context.squeeze(1)

        # Generate regression prediction
        regression_output = self.regression_head(context)
        return regression_output.squeeze(-1)

class MicrobiomeTransformerClassification(nn.Module):
    """
    Classification model based on the MicrobiomeTransformer.
    Uses attention mechanism to process sequence data for classification tasks.
    
    Args:
        pretrained_model (MicrobiomeTransformer): Pre-trained base model
        num_classes (int): Number of classification classes
        freeze_pretrained (bool): Whether to freeze the pre-trained model's parameters
    """
    def __init__(self, pretrained_model, num_classes, freeze_pretrained=True):
        super().__init__()
        self.pretrained_model = pretrained_model
        self.attention = nn.MultiheadAttention(
            pretrained_model.fc_taxa.in_features, 
            num_heads=8, 
            batch_first=True
        )
        
        # Classification head
        self.classification_head = nn.Sequential(
            nn.Linear(pretrained_model.fc_taxa.in_features, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

        if freeze_pretrained:
            self._freeze_pretrained_model()

    def _freeze_pretrained_model(self):
        """Freeze pre-trained model parameters."""
        for param in self.pretrained_model.parameters():
            param.requires_grad = False

    def forward(self, taxa, expressions, src_key_padding_mask=None):
        """Forward pass for classification prediction."""
        # Get embeddings from pre-trained model
        taxa_emb = self.pretrained_model.taxa_embedding(taxa)
        expr_emb = self.pretrained_model.expression_proj(expressions.unsqueeze(-1))
        x = taxa_emb + expr_emb
        x = x + self.pretrained_model.pos_encoder(x)

        if src_key_padding_mask is None:
            src_key_padding_mask = (taxa == 0)

        # Apply transformer and attention
        output = self.pretrained_model.transformer(x, src_key_padding_mask=src_key_padding_mask)
        query = output.mean(dim=1, keepdim=True)
        context, _ = self.attention(query, output, output, key_padding_mask=src_key_padding_mask)
        context = context.squeeze(1)

        # Generate classification prediction
        classification_output = self.classification_head(context)
        return classification_output
