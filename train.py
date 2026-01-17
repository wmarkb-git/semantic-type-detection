"""
Training Script for Sherlock Modern - Semantic Type Detection

=============================================================================
PURPOSE:
=============================================================================
This script trains a neural network to classify semantic types of table columns.
It uses a modern approach combining sentence embeddings (SBERT) with statistical
features to achieve robust semantic type detection on the Sherlock dataset.

The training pipeline extracts features from preprocessed data, builds a neural
network architecture, trains with early stopping and learning rate scheduling,
and generates comprehensive evaluation metrics and visualizations.

=============================================================================
PROCESS OVERVIEW:
=============================================================================
1. LOAD: Read preprocessed parquet files (train/val/test splits)
2. EXTRACT: Generate SBERT embeddings + statistical features from columns
3. BUILD: Construct multi-input neural network or baseline model
4. TRAIN: Fit model with early stopping and LR scheduling
5. EVALUATE: Test set predictions with confusion matrix and metrics
6. VISUALIZE: Generate training curves, confusion matrix, per-class performance
7. SAVE: Store model weights, training history, config, and visualizations

=============================================================================
INPUTS:
=============================================================================
Preprocessed Data Files:
  - Path: data/prototype/ (or custom via --data_path)
  - Files: train_values.parquet, train_labels.parquet
           val_values.parquet, val_labels.parquet
           test_values.parquet, test_labels.parquet
  - Format: Parquet files with 'values' column (lists of strings) and 'type' labels

Embedding Model:
  - Default: 'all-MiniLM-L6-v2' (384-dim sentence-transformer)
  - Customizable via --embedding_model flag

Hyperparameters:
  - Batch size: 32 (default)
  - Epochs: 50 (with early stopping)
  - Learning rate: 1e-3 (with ReduceLROnPlateau)
  - Dropout: 0.3 (default)
  - Patience: 10 epochs for early stopping

=============================================================================
OUTPUTS:
=============================================================================
Model Files (outputs/ directory):
  - final_model.pth - Final trained model
  - best_model.pth - Best model by validation accuracy
  - classes.npy - Label encoder class names
  - config.json - Training configuration and final metrics

Training Data:
  - training_history.csv - Epoch-by-epoch loss and accuracy

Evaluation Results:
  - evaluation_report.txt - Classification report on test set

Visualization Files:
  - training_curves.png - Loss and accuracy over epochs
  - confusion_matrix.png - Test set confusion matrix heatmap
  - per_class_metrics.png - Precision, recall, F1 per semantic type

=============================================================================
ARCHITECTURE OPTIONS:
=============================================================================
Multi-Input Model (default, --model_type multi_input):
  - Input 1: SBERT embeddings (384-dim) → Dense layers
  - Input 2: Statistical features (8-dim) → Dense layers  
  - Concatenate → Dense layers → Softmax output
  - Combines semantic understanding with statistical patterns

Baseline Model (--model_type baseline):
  - Input: SBERT embeddings only (384-dim)
  - Dense layers → Softmax output
  - Simpler architecture for comparison

=============================================================================
USAGE:
=============================================================================
Basic training:
  python train.py

With verbose progress output:
  python train.py --verbose

Custom hyperparameters:
  python train.py --epochs 100 --batch_size 64 --learning_rate 0.001

Baseline model comparison:
  python train.py --model_type baseline

Skip visualization generation:
  python train.py --no-viz

=============================================================================
AUTHOR: Sherlock Modernization Project
DATE: 2026-01-15
FRAMEWORK: PyTorch
=============================================================================
"""
import sys
import argparse
import json
import io
from pathlib import Path

# Fix Windows console encoding for Unicode characters
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Configure UTF-8 encoding for Windows console
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
warnings.filterwarnings('ignore', category=FutureWarning)

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from features import ColumnEmbedder, extract_stats_batch
from models import build_multi_input_model, build_baseline_model

# Global verbose flag
VERBOSE = False


def vprint(*args, **kwargs):
    """Print only if verbose mode is enabled."""
    if VERBOSE:
        print(*args, **kwargs)


def set_seed(seed):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_data(data_path, load_test=True):
    """Load preprocessed data."""
    print(f"Loading data from {data_path}...")
    vprint(f"  Reading parquet files...")
    
    train_values = pd.read_parquet(data_path / 'train_values.parquet')
    train_labels = pd.read_parquet(data_path / 'train_labels.parquet')
    vprint(f"    ✓ Train: {len(train_labels):,} samples")
    
    val_values = pd.read_parquet(data_path / 'val_values.parquet')
    val_labels = pd.read_parquet(data_path / 'val_labels.parquet')
    vprint(f"    ✓ Validation: {len(val_labels):,} samples")
    
    if load_test:
        test_values = pd.read_parquet(data_path / 'test_values.parquet')
        test_labels = pd.read_parquet(data_path / 'test_labels.parquet')
        vprint(f"    ✓ Test: {len(test_labels):,} samples")
        print(f"✓ Loaded {len(train_labels):,} train, {len(val_labels):,} val, {len(test_labels):,} test samples")
        return (train_values, train_labels), (val_values, val_labels), (test_values, test_labels)
    else:
        print(f"✓ Loaded {len(train_labels):,} train samples, {len(val_labels):,} val samples")
        return (train_values, train_labels), (val_values, val_labels)


def extract_features(values_df, labels_df, embedder, label_encoder=None):
    """
    Extract features from raw column data.
    
    Args:
        values_df: DataFrame with 'values' column containing column data
        labels_df: DataFrame with 'type' column containing semantic type labels
        embedder: ColumnEmbedder instance for generating embeddings
        label_encoder: Optional pre-fitted LabelEncoder (for val/test sets)
    
    Returns:
        (embeddings, stats, labels, label_encoder)
    """
    vprint("\nExtracting features...")
    vprint(f"  Input shape: {len(values_df):,} samples")
    
    # Convert values column to list of lists
    vprint("  Converting values to list format...")
    columns_list = values_df['values'].apply(eval if isinstance(values_df['values'].iloc[0], str) else lambda x: x).tolist()
    
    # Extract embeddings
    print("  - Generating sentence embeddings...")
    embeddings = embedder.embed_columns_batch(columns_list, show_progress=True)
    vprint(f"    ✓ Generated embeddings: {embeddings.shape}")
    
    # Extract statistical features
    print("  - Computing statistical features...")
    stats = extract_stats_batch(columns_list, show_progress=True)
    vprint(f"    ✓ Computed statistics: {stats.shape}")
    
    # Encode labels
    print("  - Encoding labels...")
    if label_encoder is None:
        label_encoder = LabelEncoder()
        encoded_labels = label_encoder.fit_transform(labels_df['type'].values)
        vprint(f"    ✓ Fitted label encoder with {len(label_encoder.classes_)} classes")
    else:
        encoded_labels = label_encoder.transform(labels_df['type'].values)
        vprint(f"    ✓ Transformed labels using existing encoder")
    
    print(f"✓ Features extracted:")
    print(f"    Embeddings shape: {embeddings.shape}")
    print(f"    Statistics shape: {stats.shape}")
    print(f"    Labels shape: {encoded_labels.shape}")
    print(f"    Number of classes: {len(label_encoder.classes_)}")
    
    return embeddings, stats, encoded_labels, label_encoder


def plot_training_curves(history, output_path):
    """Generate and save training curves (loss and accuracy)."""
    vprint("\nGenerating training curves...")
    
    sns.set_style('whitegrid')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle('Training History', fontsize=16, fontweight='bold')
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Find best epoch (highest val_accuracy)
    best_epoch = np.argmax(history['val_accuracy']) + 1
    
    # Plot loss
    ax1.plot(epochs, history['train_loss'], 'o-', label='Training Loss', color='#2E86AB', linewidth=2, markersize=4)
    ax1.plot(epochs, history['val_loss'], 's-', label='Validation Loss', color='#A23B72', linewidth=2, markersize=4)
    ax1.axvline(x=best_epoch, color='green', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Best Epoch ({best_epoch})')
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Loss', fontsize=11)
    ax1.set_title('Model Loss', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)
    
    # Plot accuracy
    ax2.plot(epochs, history['train_accuracy'], 'o-', label='Training Accuracy', color='#2E86AB', linewidth=2, markersize=4)
    ax2.plot(epochs, history['val_accuracy'], 's-', label='Validation Accuracy', color='#A23B72', linewidth=2, markersize=4)
    ax2.axvline(x=best_epoch, color='green', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Best Epoch ({best_epoch})')
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Accuracy', fontsize=11)
    ax2.set_title('Model Accuracy', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    
    output_file = output_path / 'training_curves.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    vprint(f"  ✓ Saved: {output_file}")
    print(f"✓ Training curves saved: {output_file}")
    
    plt.close()


def plot_confusion_matrix(y_true, y_pred, classes, output_path):
    """Generate and save confusion matrix."""
    vprint("\nGenerating confusion matrix...")
    
    sns.set_style('white')
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes,
                cbar_kws={'label': 'Count'},
                linewidths=0.5, linecolor='gray',
                ax=ax)
    
    ax.set_title('Confusion Matrix - Test Set', fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    
    # Rotate labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=9)
    
    plt.tight_layout()
    
    output_file = output_path / 'confusion_matrix.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    vprint(f"  ✓ Saved: {output_file}")
    print(f"✓ Confusion matrix saved: {output_file}")
    
    plt.close()


def plot_per_class_metrics(y_true, y_pred, classes, output_path):
    """Generate and save per-class performance metrics."""
    vprint("\nGenerating per-class metrics chart...")
    
    sns.set_style('whitegrid')
    
    # Calculate metrics
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None)
    
    # Create DataFrame for plotting
    metrics_df = pd.DataFrame({
        'Class': classes,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Support': support
    })
    
    # Sort by F1-score for better visualization
    metrics_df = metrics_df.sort_values('F1-Score', ascending=True)
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Per-Class Performance Metrics', fontsize=16, fontweight='bold')
    
    # Plot 1: Precision, Recall, F1-Score
    y_pos = np.arange(len(metrics_df))
    width = 0.25
    
    ax1.barh(y_pos - width, metrics_df['Precision'], width, label='Precision', color='#2E86AB', alpha=0.8)
    ax1.barh(y_pos, metrics_df['Recall'], width, label='Recall', color='#A23B72', alpha=0.8)
    ax1.barh(y_pos + width, metrics_df['F1-Score'], width, label='F1-Score', color='#F18F01', alpha=0.8)
    
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(metrics_df['Class'], fontsize=9)
    ax1.set_xlabel('Score', fontsize=11)
    ax1.set_title('Classification Metrics by Class', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.set_xlim(0, 1.0)
    ax1.grid(axis='x', alpha=0.3)
    
    # Plot 2: Sample support
    ax2.barh(y_pos, metrics_df['Support'], color='#6A994E', alpha=0.8)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(metrics_df['Class'], fontsize=9)
    ax2.set_xlabel('Number of Samples', fontsize=11)
    ax2.set_title('Test Set Support by Class', fontsize=12, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    
    # Add value labels on bars
    for i, v in enumerate(metrics_df['Support']):
        ax2.text(v + 1, i, str(int(v)), va='center', fontsize=8)
    
    plt.tight_layout()
    
    output_file = output_path / 'per_class_metrics.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    vprint(f"  ✓ Saved: {output_file}")
    print(f"✓ Per-class metrics saved: {output_file}")
    
    plt.close()


def generate_evaluation_report(model, test_loader, label_encoder, device, output_path, model_type, generate_viz=True):
    """Evaluate model on test set and generate comprehensive report."""
    print("\n" + "="*50)
    print("EVALUATING ON TEST SET")
    print("="*50)
    
    model.eval()
    all_preds = []
    all_labels = []
    
    vprint("\nMaking predictions on test set...")
    with torch.no_grad():
        for batch in test_loader:
            if model_type == 'multi_input':
                emb, stats, labels = batch
                emb, stats = emb.to(device), stats.to(device)
                outputs = model(emb, stats)
            else:
                emb, labels = batch
                emb = emb.to(device)
                outputs = model(emb)
            
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    vprint(f"  ✓ Generated {len(all_preds):,} predictions")
    
    # Calculate test accuracy
    test_accuracy = np.mean(all_preds == all_labels)
    print(f"\nTest Set Accuracy: {test_accuracy:.4f}")
    
    # Generate classification report
    vprint("\nGenerating classification report...")
    report = classification_report(all_labels, all_preds, target_names=label_encoder.classes_, digits=4)
    print("\nClassification Report:")
    print(report)
    
    # Save text report
    report_path = output_path / 'evaluation_report.txt'
    with open(report_path, 'w') as f:
        f.write("="*50 + "\n")
        f.write("SHERLOCK MODERN - TEST SET EVALUATION REPORT\n")
        f.write("="*50 + "\n\n")
        f.write(f"Test Set Accuracy: {test_accuracy:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
    print(f"\n✓ Evaluation report saved: {report_path}")
    
    # Generate visualizations
    if generate_viz:
        print("\n" + "="*50)
        print("GENERATING EVALUATION VISUALIZATIONS")
        print("="*50)
        
        try:
            plot_confusion_matrix(all_labels, all_preds, label_encoder.classes_, output_path)
            plot_per_class_metrics(all_labels, all_preds, label_encoder.classes_, output_path)
            
            if VERBOSE:
                print("\nVisualization files created:")
                print(f"  - {output_path / 'confusion_matrix.png'}")
                print(f"  - {output_path / 'per_class_metrics.png'}")
        except Exception as e:
            print(f"\n⚠ Warning: Visualization generation failed: {e}")
            print("  Evaluation report was saved successfully.")
    
    return test_accuracy, report


def train_model(args):
    """Main training pipeline."""
    
    # Set global verbose flag
    global VERBOSE
    VERBOSE = args.verbose
    
    if VERBOSE:
        print("="*50)
        print("VERBOSE MODE ENABLED")
        print("="*50)
        print(f"Configuration:")
        print(f"  Model type: {args.model_type}")
        print(f"  Embedding model: {args.embedding_model}")
        print(f"  Batch size: {args.batch_size}")
        print(f"  Max epochs: {args.epochs}")
        print(f"  Learning rate: {args.learning_rate}")
        print(f"  Dropout: {args.dropout}")
        print(f"  Early stopping patience: {args.patience}")
        print(f"  Random seed: {args.seed}")
        print(f"  Generate visualizations: {not args.no_viz}")
        print(f"  Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
        print("="*50 + "\n")
    
    # Set random seeds for reproducibility
    set_seed(args.seed)
    vprint(f"Random seed set to: {args.seed}\n")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Load data
    data_path = Path(args.data_path)
    (train_values, train_labels), (val_values, val_labels), (test_values, test_labels) = load_data(data_path, load_test=True)
    
    # Initialize embedder
    print("\n" + "="*50)
    print("INITIALIZING EMBEDDER")
    print("="*50)
    vprint(f"  Loading embedding model: {args.embedding_model}...")
    embedder = ColumnEmbedder(model_name=args.embedding_model)
    print(f"✓ Embedder ready: {args.embedding_model}")
    
    # Extract features
    print("\n" + "="*50)
    print("EXTRacting TRAIN FEATURES")
    print("="*50)
    X_train_emb, X_train_stats, y_train, label_encoder = extract_features(
        train_values, train_labels, embedder
    )
    
    print("\n" + "="*50)
    print("EXTRACTING VALIDATION FEATURES")
    print("="*50)
    X_val_emb, X_val_stats, y_val, _ = extract_features(
        val_values, val_labels, embedder, label_encoder=label_encoder
    )
    
    print("\n" + "="*50)
    print("EXTRACTING TEST FEATURES")
    print("="*50)
    X_test_emb, X_test_stats, y_test, _ = extract_features(
        test_values, test_labels, embedder, label_encoder=label_encoder
    )
    
    # Convert to PyTorch tensors
    vprint("\nConverting data to PyTorch tensors...")
    X_train_emb = torch.tensor(X_train_emb, dtype=torch.float32)
    X_train_stats = torch.tensor(X_train_stats, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    
    X_val_emb = torch.tensor(X_val_emb, dtype=torch.float32)
    X_val_stats = torch.tensor(X_val_stats, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.long)
    
    X_test_emb = torch.tensor(X_test_emb, dtype=torch.float32)
    X_test_stats = torch.tensor(X_test_stats, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)
    
    # Create DataLoaders
    if args.model_type == 'multi_input':
        train_dataset = TensorDataset(X_train_emb, X_train_stats, y_train)
        val_dataset = TensorDataset(X_val_emb, X_val_stats, y_val)
        test_dataset = TensorDataset(X_test_emb, X_test_stats, y_test)
    else:
        train_dataset = TensorDataset(X_train_emb, y_train)
        val_dataset = TensorDataset(X_val_emb, y_val)
        test_dataset = TensorDataset(X_test_emb, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Build model
    print("\n" + "="*50)
    print("BUILDING MODEL")
    print("="*50)
    vprint(f"\n  Model architecture: {args.model_type}")
    vprint(f"  Embedding dimension: {X_train_emb.shape[1]}")
    vprint(f"  Statistics dimension: {X_train_stats.shape[1]}")
    vprint(f"  Number of classes: {len(label_encoder.classes_)}")
    vprint(f"  Dropout rate: {args.dropout}\n")
    
    if args.model_type == 'multi_input':
        model = build_multi_input_model(
            embedding_dim=X_train_emb.shape[1],
            stats_dim=X_train_stats.shape[1],
            num_classes=len(label_encoder.classes_),
            dropout_rate=args.dropout
        )
    else:  # baseline
        model = build_baseline_model(
            input_dim=X_train_emb.shape[1],
            num_classes=len(label_encoder.classes_),
            dropout_rate=args.dropout
        )
    
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Model built:")
    print(f"    Total parameters: {total_params:,}")
    print(f"    Trainable parameters: {trainable_params:,}")
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Training loop
    print("\n" + "="*50)
    print("TRAINING MODEL")
    print("="*50)
    
    history = {
        'train_loss': [],
        'train_accuracy': [],
        'val_loss': [],
        'val_accuracy': []
    }
    
    best_val_acc = 0.0
    patience_counter = 0
    
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(args.epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for batch in train_loader:
            if args.model_type == 'multi_input':
                emb, stats, labels = batch
                emb, stats, labels = emb.to(device), stats.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(emb, stats)
            else:
                emb, labels = batch
                emb, labels = emb.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(emb)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_loss /= len(train_loader)
        train_acc = correct / total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                if args.model_type == 'multi_input':
                    emb, stats, labels = batch
                    emb, stats, labels = emb.to(device), stats.to(device), labels.to(device)
                    outputs = model(emb, stats)
                else:
                    emb, labels = batch
                    emb, labels = emb.to(device), labels.to(device)
                    outputs = model(emb)
                
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = correct / total
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_accuracy'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_acc)
        
        # Print progress
        print(f"Epoch [{epoch+1}/{args.epochs}] - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), output_path / 'best_model.pth')
            patience_counter = 0
            vprint(f"  ✓ Saved best model (val_acc: {val_acc:.4f})")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= args.patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
    
    # Save final model
    torch.save(model.state_dict(), output_path / 'final_model.pth')
    print(f"✓ Final model saved to {output_path / 'final_model.pth'}")
    
    # Save label encoder classes
    np.save(output_path / 'classes.npy', label_encoder.classes_)
    print(f"✓ Label encoder saved to {output_path / 'classes.npy'}")
    
    # Save training history
    history_df = pd.DataFrame(history)
    history_df.to_csv(output_path / 'training_history.csv', index=False)
    print(f"✓ Saved training history to {output_path / 'training_history.csv'}")
    
    # Save configuration
    config = {
        'model_type': args.model_type,
        'embedding_model': args.embedding_model,
        'embedding_dim': X_train_emb.shape[1],
        'stats_dim': X_train_stats.shape[1] if args.model_type == 'multi_input' else 0,
        'num_classes': len(label_encoder.classes_),
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'dropout': args.dropout,
        'total_epochs': len(history['train_loss']),
        'best_val_accuracy': float(best_val_acc),
        'seed': args.seed
    }
    
    # Generate training curve visualizations
    if not args.no_viz:
        print("\n" + "="*50)
        print("GENERATING TRAINING VISUALIZATIONS")
        print("="*50)
        
        try:
            plot_training_curves(history, output_path)
            if VERBOSE:
                print(f"\nTraining visualization created:")
                print(f"  - {output_path / 'training_curves.png'}")
        except Exception as e:
            print(f"\n⚠ Warning: Training visualization failed: {e}")
            print("  Model and history were saved successfully.")
    
    # Load best model for evaluation
    model.load_state_dict(torch.load(output_path / 'best_model.pth'))
    
    # Evaluate on test set
    test_accuracy, report = generate_evaluation_report(
        model, test_loader, label_encoder, device, output_path, args.model_type, generate_viz=not args.no_viz
    )
    
    # Update config with test accuracy
    config['test_accuracy'] = float(test_accuracy)
    with open(output_path / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    print(f"✓ Configuration saved to {output_path / 'config.json'}")
    
    print("\n" + "="*50)
    print("TRAINING COMPLETE!")
    print("="*50)
    print(f"\nBest validation accuracy: {best_val_acc:.4f}")
    print(f"Final validation accuracy: {history['val_accuracy'][-1]:.4f}")
    print(f"Final validation loss: {history['val_loss'][-1]:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    
    if not args.no_viz:
        print(f"\nView results:")
        print(f"  Model files: {output_path.absolute()}")
        print(f"  Visualizations: {output_path.absolute()}\\*.png")
    
    return model, history, test_accuracy


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Sherlock Modern with PyTorch')
    
    # Data args
    parser.add_argument('--data_path', type=str, default='data/prototype',
                        help='Path to preprocessed data')
    parser.add_argument('--output_path', type=str, default='outputs',
                        help='Path to save model and results')
    
    # Model args
    parser.add_argument('--model_type', type=str, default='multi_input',
                        choices=['multi_input', 'baseline'],
                        help='Model architecture to use')
    parser.add_argument('--embedding_model', type=str, default='all-MiniLM-L6-v2',
                        help='Sentence-transformer model name')
    
    # Training args
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Maximum number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout rate')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    # Output args
    parser.add_argument('--verbose', '-v',
                        action='store_true',
                        help='Show detailed progress information')
    parser.add_argument('--no-viz',
                        action='store_true',
                        help='Skip visualization generation (charts will not be created)')
    
    args = parser.parse_args()
    
    train_model(args)
