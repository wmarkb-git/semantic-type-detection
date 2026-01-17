"""
Evaluation script for Sherlock Modern (PyTorch).
Evaluates trained model on test set with verbose mode and comparison charts.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score, confusion_matrix
import torch
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import io

# Fix Windows console encoding for Unicode characters
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

sys.path.append(str(Path(__file__).parent))

from features import ColumnEmbedder, extract_stats_batch
from models import build_multi_input_model, build_baseline_model

# Global verbose flag
VERBOSE = False


def vprint(*args, **kwargs):
    """Print only if verbose mode is enabled."""
    if VERBOSE:
        print(*args, **kwargs)


def load_test_data(data_path):
    """Load test data."""
    print(f"Loading test data from {data_path}...")
    vprint(f"  Reading parquet files...")
    
    test_values = pd.read_parquet(data_path / 'test_values.parquet')
    test_labels = pd.read_parquet(data_path / 'test_labels.parquet')
    
    print(f"✓ Loaded {len(test_labels):,} test samples")
    vprint(f"  Test values shape: {test_values.shape}")
    vprint(f"  Test labels shape: {test_labels.shape}")
    
    return test_values, test_labels


def extract_test_features(values_df, embedder):
    """Extract features from test data."""
    print("\nExtracting test features...")
    vprint(f"  Number of columns to process: {len(values_df):,}")
    
    # Convert values column to list of lists
    vprint("  Converting values to list format...")
    columns_list = values_df['values'].apply(eval if isinstance(values_df['values'].iloc[0], str) else lambda x: x).tolist()
    
    # Extract embeddings
    print("  - Generating sentence embeddings...")
    embeddings = embedder.embed_columns_batch(columns_list, show_progress=True)
    vprint(f"    Generated embeddings: {embeddings.shape}")
    
    # Extract statistical features
    print("  - Computing statistical features...")
    stats = extract_stats_batch(columns_list, show_progress=True)
    vprint(f"    Computed statistics: {stats.shape}")
    
    print(f"✓ Features extracted:")
    print(f"    Embeddings shape: {embeddings.shape}")
    print(f"    Statistics shape: {stats.shape}")
    
    return embeddings, stats


def plot_training_vs_test_comparison(history_df, test_accuracy, test_f1, output_path):
    """Plot comparison of training/validation vs test performance."""
    vprint("\nGenerating training vs test comparison chart...")
    
    sns.set_style('whitegrid')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle('Training vs Test Performance Comparison', fontsize=16, fontweight='bold')
    
    # Get final training and validation accuracies
    final_train_acc = history_df['train_accuracy'].iloc[-1]
    final_val_acc = history_df['val_accuracy'].iloc[-1]
    
    # Accuracy comparison
    categories = ['Training\n(Final)', 'Validation\n(Final)', 'Test']
    accuracies = [final_train_acc, final_val_acc, test_accuracy]
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    
    bars = ax1.bar(categories, accuracies, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax1.set_title('Accuracy Comparison', fontsize=13, fontweight='bold')
    ax1.set_ylim(0, 1.0)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # Training history with test point
    epochs = range(1, len(history_df) + 1)
    ax2.plot(epochs, history_df['train_accuracy'], 'o-', label='Training', 
             color='#2E86AB', linewidth=2, markersize=4)
    ax2.plot(epochs, history_df['val_accuracy'], 's-', label='Validation', 
             color='#A23B72', linewidth=2, markersize=4)
    ax2.axhline(y=test_accuracy, color='#F18F01', linestyle='--', linewidth=2.5, 
                label=f'Test ({test_accuracy:.3f})', alpha=0.8)
    
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax2.set_title('Accuracy Over Training', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10, loc='lower right')
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    
    output_file = output_path / 'training_vs_test_comparison.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    vprint(f"  ✓ Saved: {output_file}")
    print(f"✓ Training vs test comparison saved: {output_file}")
    
    plt.close()


def plot_per_class_f1_comparison(class_report, classes, output_path):
    """Plot per-class F1 scores with support."""
    vprint("\nGenerating per-class F1 score chart...")
    
    sns.set_style('whitegrid')
    
    # Extract metrics for plotting
    f1_scores = []
    support = []
    for cls in classes:
        if cls in class_report:
            f1_scores.append(class_report[cls]['f1-score'])
            support.append(class_report[cls]['support'])
        else:
            f1_scores.append(0)
            support.append(0)
    
    # Create DataFrame and sort by F1
    metrics_df = pd.DataFrame({
        'Class': classes,
        'F1-Score': f1_scores,
        'Support': support
    }).sort_values('F1-Score', ascending=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Per-Class Performance on Test Set', fontsize=16, fontweight='bold')
    
    # F1 scores
    y_pos = np.arange(len(metrics_df))
    bars = ax1.barh(y_pos, metrics_df['F1-Score'], color='#2E86AB', alpha=0.7, edgecolor='black')
    
    # Color code: green for good (>0.9), yellow for ok (0.7-0.9), red for poor (<0.7)
    for i, (bar, f1) in enumerate(zip(bars, metrics_df['F1-Score'])):
        if f1 >= 0.9:
            bar.set_color('#6A994E')
        elif f1 >= 0.7:
            bar.set_color('#F18F01')
        else:
            bar.set_color('#D62828')
    
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(metrics_df['Class'], fontsize=10)
    ax1.set_xlabel('F1-Score', fontsize=12, fontweight='bold')
    ax1.set_title('F1-Score by Class', fontsize=13, fontweight='bold')
    ax1.set_xlim(0, 1.0)
    ax1.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(metrics_df['F1-Score']):
        ax1.text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=9, fontweight='bold')
    
    # Support
    ax2.barh(y_pos, metrics_df['Support'], color='#A23B72', alpha=0.7, edgecolor='black')
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(metrics_df['Class'], fontsize=10)
    ax2.set_xlabel('Number of Samples', fontsize=12, fontweight='bold')
    ax2.set_title('Test Set Support', fontsize=13, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(metrics_df['Support']):
        ax2.text(v + 1, i, str(int(v)), va='center', fontsize=9)
    
    plt.tight_layout()
    
    output_file = output_path / 'per_class_f1_scores.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    vprint(f"  ✓ Saved: {output_file}")
    print(f"✓ Per-class F1 scores saved: {output_file}")
    
    plt.close()


def evaluate_model(args):
    """Main evaluation pipeline."""
    
    # Set global verbose flag
    global VERBOSE
    VERBOSE = args.verbose
    
    if VERBOSE:
        print("="*50)
        print("VERBOSE MODE ENABLED")
        print("="*50)
        print(f"Model path: {args.model_path}")
        print(f"Data path: {args.data_path}")
        print(f"Save results: {args.save_results}")
        print("="*50 + "\n")
    
    output_path = Path(args.model_path)
    data_path = Path(args.data_path)
    
    # Load config
    print("="*50)
    print("LOADING MODEL CONFIG")
    print("="*50)
    
    with open(output_path / 'config.json', 'r') as f:
        config = json.load(f)
    
    print(f"Model type: {config['model_type']}")
    print(f"Number of classes: {config['num_classes']}")
    print(f"Embedding dim: {config['embedding_dim']}")
    if config['model_type'] == 'multi_input':
        print(f"Stats dim: {config['stats_dim']}")
    
    vprint(f"\nFull configuration:")
    vprint(json.dumps(config, indent=2))
    
    # Load classes
    classes = np.load(output_path / 'classes.npy', allow_pickle=True)
    print(f"Classes: {', '.join(classes)}")
    vprint(f"Number of classes: {len(classes)}")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Build model
    print("\n" + "="*50)
    print("BUILDING MODEL")
    print("="*50)
    
    if config['model_type'] == 'multi_input':
        model = build_multi_input_model(
            embedding_dim=config['embedding_dim'],
            stats_dim=config['stats_dim'],
            num_classes=config['num_classes'],
            dropout_rate=config.get('dropout', 0.3)
        )
    else:
        model = build_baseline_model(
            input_dim=config['embedding_dim'],
            num_classes=config['num_classes'],
            dropout_rate=config.get('dropout', 0.4)
        )
    
    # Load model weights
    print("\n" + "="*50)
    print("LOADING MODEL WEIGHTS")
    print("="*50)
    
    model_file = output_path / 'best_model.pth'
    if not model_file.exists():
        model_file = output_path / 'final_model.pth'
    
    model.load_state_dict(torch.load(model_file, map_location=device))
    model = model.to(device)
    model.eval()
    print(f"✓ Model loaded from {model_file}")
    
    # Load test data
    print("\n" + "="*50)
    print("LOADING TEST DATA")
    print("="*50)
    
    test_values, test_labels = load_test_data(data_path)
    
    # Initialize embedder
    embedder = ColumnEmbedder(model_name=config['embedding_model'])
    
    # Extract features
    print("\n" + "="*50)
    print("EXTRACTING FEATURES")
    print("="*50)
    
    X_test_emb, X_test_stats = extract_test_features(test_values, embedder)
    
    # Convert to tensors
    X_test_emb = torch.tensor(X_test_emb, dtype=torch.float32)
    X_test_stats = torch.tensor(X_test_stats, dtype=torch.float32)
    
    # Create DataLoader
    if config['model_type'] == 'multi_input':
        test_dataset = TensorDataset(X_test_emb, X_test_stats)
    else:
        test_dataset = TensorDataset(X_test_emb)
    
    batch_size = 64
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    vprint(f"\nCreated DataLoader with batch size: {batch_size}")
    vprint(f"Number of batches: {len(test_loader)}")
    
    # Make predictions
    print("\n" + "="*50)
    print("MAKING PREDICTIONS")
    print("="*50)
    
    y_pred_proba_list = []
    
    with torch.no_grad():
        for batch in test_loader:
            if config['model_type'] == 'multi_input':
                emb, stats = batch
                emb, stats = emb.to(device), stats.to(device)
                outputs = model(emb, stats)
            else:
                emb = batch[0]
                emb = emb.to(device)
                outputs = model(emb)
            
            # Apply softmax to get probabilities
            proba = torch.softmax(outputs, dim=1)
            y_pred_proba_list.append(proba.cpu().numpy())
    
    y_pred_proba = np.vstack(y_pred_proba_list)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_pred_labels = classes[y_pred]
    
    vprint(f"\nPrediction statistics:")
    vprint(f"  Prediction probabilities shape: {y_pred_proba.shape}")
    vprint(f"  Mean confidence: {y_pred_proba.max(axis=1).mean():.4f}")
    vprint(f"  Min confidence: {y_pred_proba.max(axis=1).min():.4f}")
    vprint(f"  Max confidence: {y_pred_proba.max(axis=1).max():.4f}")
    
    print(f"✓ Generated {len(y_pred):,} predictions")
    
    # Get true labels
    y_true_labels = test_labels['type'].values
    
    # Compute metrics
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    
    # Accuracy
    accuracy = np.mean(y_true_labels == y_pred_labels)
    print(f"\nAccuracy: {accuracy:.4f}")
    
    # Overall F1 score
    f1 = f1_score(y_true_labels, y_pred_labels, average='weighted')
    print(f"Weighted F1 Score: {f1:.4f}")
    
    # Classification report
    print("\n" + "-"*50)
    print("Classification Report:")
    print("-"*50)
    print(classification_report(y_true_labels, y_pred_labels, digits=4))
    
    # Per-class F1 scores
    class_report = classification_report(y_true_labels, y_pred_labels, output_dict=True)
    
    f1_scores = [(cls, metrics['f1-score']) for cls, metrics in class_report.items() 
                 if cls in classes]
    f1_scores_sorted = sorted(f1_scores, key=lambda x: x[1], reverse=True)
    
    print("\n" + "-"*50)
    print("Top 5 Best Performing Types:")
    print("-"*50)
    for cls, f1 in f1_scores_sorted[:5]:
        print(f"  {cls:20s}: {f1:.4f}")
    
    print("\n" + "-"*50)
    print("Top 5 Worst Performing Types:")
    print("-"*50)
    for cls, f1 in f1_scores_sorted[-5:][::-1]:  # Reverse to show worst first
        print(f"  {cls:20s}: {f1:.4f}")
    
    # Load training history for comparison
    history_df = None
    history_path = output_path / 'training_history.csv'
    if history_path.exists():
        vprint(f"\nLoading training history from {history_path}...")
        history_df = pd.read_csv(history_path)
        vprint(f"  Training ran for {len(history_df)} epochs")
    else:
        vprint(f"\n⚠ Training history not found at {history_path}")
    
    # Save results
    if args.save_results:
        results_path = output_path / 'evaluation_results'
        results_path.mkdir(exist_ok=True)
        vprint(f"\nResults will be saved to: {results_path}")
        
        print("\n" + "="*50)
        print("SAVING RESULTS")
        print("="*50)
        
        # Save classification report
        report_df = pd.DataFrame(class_report).transpose()
        report_df.to_csv(results_path / 'classification_report.csv')
        print(f"✓ Saved classification report to {results_path / 'classification_report.csv'}")
        
        # Save predictions
        predictions_df = pd.DataFrame({
            'true_label': y_true_labels,
            'predicted_label': y_pred_labels,
            'correct': y_true_labels == y_pred_labels
        })
        
        # Add probabilities for each class
        for i, cls in enumerate(classes):
            predictions_df[f'prob_{cls}'] = y_pred_proba[:, i]
        
        predictions_df.to_csv(results_path / 'predictions.csv', index=False)
        print(f"✓ Saved predictions to {results_path / 'predictions.csv'}")
        
        # Plot confusion matrix
        print("\nGenerating confusion matrix...")
        cm = confusion_matrix(y_true_labels, y_pred_labels, labels=classes)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=classes, yticklabels=classes,
                    linewidths=0.5, linecolor='gray')
        plt.title('Confusion Matrix', fontsize=14, fontweight='bold', pad=20)
        plt.ylabel('True Label', fontsize=12, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(results_path / 'confusion_matrix.png', dpi=150)
        print(f"✓ Saved confusion matrix to {results_path / 'confusion_matrix.png'}")
        plt.close()
        
        # Generate comparison charts
        if history_df is not None:
            print("\nGenerating comparison charts...")
            plot_training_vs_test_comparison(history_df, accuracy, f1, results_path)
            plot_per_class_f1_comparison(class_report, classes, results_path)
        else:
            vprint("\n⚠ Skipping comparison charts (no training history)")
        
        # Save summary
        summary = {
            'accuracy': float(accuracy),
            'weighted_f1': float(f1),
            'macro_avg_f1': float(class_report['macro avg']['f1-score']),
            'macro_avg_precision': float(class_report['macro avg']['precision']),
            'macro_avg_recall': float(class_report['macro avg']['recall']),
            'num_samples': len(y_true_labels),
            'num_correct': int((y_true_labels == y_pred_labels).sum()),
            'num_incorrect': int((y_true_labels != y_pred_labels).sum()),
            'model_type': config['model_type'],
            'embedding_model': config['embedding_model']
        }
        
        with open(results_path / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"✓ Saved summary to {results_path / 'summary.json'}")
    
    print("\n" + "="*50)
    print("EVALUATION COMPLETE!")
    print("="*50)
    print(f"\nFinal Test Accuracy: {accuracy:.4f}")
    print(f"Weighted F1 Score: {f1:.4f}")
    
    if args.save_results:
        print(f"\nResults saved to: {results_path.absolute()}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Sherlock Modern (PyTorch)')
    
    parser.add_argument('--model_path', type=str, default='outputs',
                        help='Path to trained model')
    parser.add_argument('--data_path', type=str, default='data/prototype',
                        help='Path to test data')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Show detailed progress information')
    parser.add_argument('--save_results', action='store_true', default=True,
                        help='Save evaluation results')
    
    args = parser.parse_args()
    
    evaluate_model(args)
