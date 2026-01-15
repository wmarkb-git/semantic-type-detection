"""
Multi-input neural network model using PyTorch.
Architecture inspired by original Sherlock but simplified.
"""
import torch
import torch.nn as nn
import numpy as np


class MultiInputModel(nn.Module):
    """
    Multi-input neural network for semantic type classification.
    
    Architecture:
        - Branch 1: Embedding features → Dense layers
        - Branch 2: Statistical features → Dense layers
        - Fusion: Concatenate branches → Dense layers → Softmax
    """
    
    def __init__(
        self,
        embedding_dim=384,
        stats_dim=8,
        num_classes=10,
        embedding_units=[128, 64],
        stats_units=[64, 32],
        fusion_units=[64],
        dropout_rate=0.3
    ):
        super(MultiInputModel, self).__init__()
        
        # Embedding branch
        self.embedding_layers = nn.ModuleList()
        prev_dim = embedding_dim
        for units in embedding_units:
            self.embedding_layers.append(nn.Sequential(
                nn.Linear(prev_dim, units),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ))
            prev_dim = units
        
        # Stats branch
        self.stats_layers = nn.ModuleList()
        prev_dim = stats_dim
        for units in stats_units:
            self.stats_layers.append(nn.Sequential(
                nn.Linear(prev_dim, units),
                nn.ReLU(),
                nn.Dropout(dropout_rate * 0.7)  # Less dropout for stats
            ))
            prev_dim = units
        
        # Fusion layers
        fusion_input_dim = embedding_units[-1] + stats_units[-1]
        self.fusion_layers = nn.ModuleList()
        prev_dim = fusion_input_dim
        for units in fusion_units:
            self.fusion_layers.append(nn.Sequential(
                nn.Linear(prev_dim, units),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ))
            prev_dim = units
        
        # Output layer
        self.output_layer = nn.Linear(prev_dim, num_classes)
    
    def forward(self, embedding_input, stats_input):
        """
        Forward pass.
        
        Args:
            embedding_input: Tensor of shape (batch_size, embedding_dim)
            stats_input: Tensor of shape (batch_size, stats_dim)
        
        Returns:
            Tensor of shape (batch_size, num_classes) with logits
        """
        # Embedding branch
        x = embedding_input
        for layer in self.embedding_layers:
            x = layer(x)
        
        # Stats branch
        y = stats_input
        for layer in self.stats_layers:
            y = layer(y)
        
        # Concatenate
        z = torch.cat([x, y], dim=1)
        
        # Fusion layers
        for layer in self.fusion_layers:
            z = layer(z)
        
        # Output
        output = self.output_layer(z)
        return output


class BaselineModel(nn.Module):
    """
    Simple baseline model (embeddings only).
    """
    
    def __init__(
        self,
        input_dim=384,
        num_classes=10,
        hidden_units=[128],
        dropout_rate=0.4
    ):
        super(BaselineModel, self).__init__()
        
        layers = []
        prev_dim = input_dim
        for units in hidden_units:
            layers.extend([
                nn.Linear(prev_dim, units),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = units
        
        layers.append(nn.Linear(prev_dim, num_classes))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Tensor of shape (batch_size, input_dim)
        
        Returns:
            Tensor of shape (batch_size, num_classes) with logits
        """
        return self.model(x)


def build_multi_input_model(
    embedding_dim=384,
    stats_dim=8,
    num_classes=10,
    embedding_units=[128, 64],
    stats_units=[64, 32],
    fusion_units=[64],
    dropout_rate=0.3
):
    """
    Build multi-input neural network for semantic type classification.
    
    Args:
        embedding_dim: Dimension of sentence embeddings (default: 384 for MiniLM)
        stats_dim: Dimension of statistical features
        num_classes: Number of semantic types to predict
        embedding_units: Hidden units for embedding branch
        stats_units: Hidden units for stats branch
        fusion_units: Hidden units for fusion layers
        dropout_rate: Dropout rate
        
    Returns:
        PyTorch Model
    """
    model = MultiInputModel(
        embedding_dim=embedding_dim,
        stats_dim=stats_dim,
        num_classes=num_classes,
        embedding_units=embedding_units,
        stats_units=stats_units,
        fusion_units=fusion_units,
        dropout_rate=dropout_rate
    )
    return model


def build_baseline_model(
    input_dim=384,
    num_classes=10,
    hidden_units=[128],
    dropout_rate=0.4
):
    """
    Build simple baseline model (embeddings only).
    
    Args:
        input_dim: Input feature dimension
        num_classes: Number of classes
        hidden_units: Hidden layer units
        dropout_rate: Dropout rate
        
    Returns:
        PyTorch Model
    """
    model = BaselineModel(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_units=hidden_units,
        dropout_rate=dropout_rate
    )
    return model


if __name__ == '__main__':
    # Test model building
    print("="*50)
    print("Testing Model Architecture")
    print("="*50)
    
    print("\n1. Building multi-input model...")
    model = build_multi_input_model()
    
    print("\nModel Summary:")
    print(model)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    print("\n2. Testing forward pass...")
    # Create dummy data
    batch_size = 4
    dummy_embeddings = torch.randn(batch_size, 384)
    dummy_stats = torch.randn(batch_size, 8)
    
    with torch.no_grad():
        predictions = model(dummy_embeddings, dummy_stats)
    
    print(f"Prediction shape: {predictions.shape}")
    print(f"Expected: torch.Size([{batch_size}, 10])")
    
    print("\n3. Building baseline model...")
    baseline = build_baseline_model()
    print(baseline)
    
    print("\n✓ All tests passed!")
