"""
Data Preparation Script for Sherlock Semantic Type Detection Prototype

=============================================================================
PURPOSE:
=============================================================================
This script prepares a focused subset of the original Sherlock dataset for
prototype development and testing. It filters the full dataset to a manageable
size while maintaining class balance and statistical validity.

The script transforms Sherlock's 78 semantic types (with ~780k training samples)
into a prototype dataset with 10 carefully selected types and balanced samples
per type. This allows for faster iteration during model development while still
testing core semantic type detection capabilities.

=============================================================================
PROCESS OVERVIEW:
=============================================================================
1. LOAD: Read original Sherlock parquet files (train/val/test splits)
2. FILTER: Keep only the 10 selected semantic types
3. SAMPLE: Extract balanced samples per type (1000 train, 150 val/test)
4. SAVE: Write processed data to data/prototype/ directory
5. VISUALIZE: Generate charts showing data distribution and statistics
6. REPORT: Print summary statistics and file locations

=============================================================================
INPUTS:
=============================================================================
Original Sherlock Dataset Location:
  - Path: data/data/raw/
  - Files: train_values.parquet, train_labels.parquet
           val_values.parquet, val_labels.parquet
           test_values.parquet, test_labels.parquet
  - Format: Parquet files with feature vectors and type labels

=============================================================================
OUTPUTS:
=============================================================================
Processed Prototype Dataset:
  - Path: data/prototype/
  - Data Files (6 total):
    • train_values.parquet, train_labels.parquet
    • val_values.parquet, val_labels.parquet
    • test_values.parquet, test_labels.parquet
  - Visualization Files:
    • class_distribution.png - Bar charts showing sample counts per type
    • dataset_sizes.png - Comparison of original vs filtered vs sampled sizes

=============================================================================
CONFIGURATION:
=============================================================================
Selected Semantic Types (10):
  'person', 'city', 'country', 'company', 'year',
  'address', 'category', 'ranking', 'isbn', 'jockey'

Sample Sizes:
  - Training set: 1,000 samples per type (10,000 total)
  - Validation set: 150 samples per type (1,500 total)
  - Test set: 150 samples per type (1,500 total)
  - TOTAL: 13,000 samples

Random Seed: 42 (for reproducibility)

=============================================================================
USAGE:
=============================================================================
Basic usage:
  python prepare_data.py

With detailed progress output:
  python prepare_data.py --verbose

Skip visualization generation:
  python prepare_data.py --no-viz

=============================================================================
AUTHOR: Sherlock Modernization Project
DATE: 2026-01-15
=============================================================================
"""
import sys
import io
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Fix Windows console encoding for Unicode characters
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

# Suppress matplotlib warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

# Configuration
SELECTED_TYPES = [
    'person', 'city', 'country', 'company', 'year',
    'address', 'category', 'ranking', 'isbn', 'jockey'
]
SAMPLES_PER_TYPE = 1000
RANDOM_SEED = 42

# Paths
ORIGINAL_DATA_PATH = Path('data/data/raw')
OUTPUT_PATH = Path('data/prototype')

# Global verbose flag
VERBOSE = False


def vprint(*args, **kwargs):
    """Print only if verbose mode is enabled."""
    if VERBOSE:
        print(*args, **kwargs)


def load_original_data():
    """Load original Sherlock data."""
    print("Loading original Sherlock data...")
    vprint(f"  Reading from: {ORIGINAL_DATA_PATH}")
    
    vprint("  Loading train_values.parquet...")
    train_values = pd.read_parquet(ORIGINAL_DATA_PATH / 'train_values.parquet')
    vprint(f"    ✓ Loaded {len(train_values):,} rows")
    
    vprint("  Loading train_labels.parquet...")
    train_labels = pd.read_parquet(ORIGINAL_DATA_PATH / 'train_labels.parquet')
    vprint(f"    ✓ Loaded {len(train_labels):,} labels")
    
    vprint("  Loading val_values.parquet...")
    val_values = pd.read_parquet(ORIGINAL_DATA_PATH / 'val_values.parquet')
    vprint(f"    ✓ Loaded {len(val_values):,} rows")
    
    vprint("  Loading val_labels.parquet...")
    val_labels = pd.read_parquet(ORIGINAL_DATA_PATH / 'val_labels.parquet')
    vprint(f"    ✓ Loaded {len(val_labels):,} labels")
    
    vprint("  Loading test_values.parquet...")
    test_values = pd.read_parquet(ORIGINAL_DATA_PATH / 'test_values.parquet')
    vprint(f"    ✓ Loaded {len(test_values):,} rows")
    
    vprint("  Loading test_labels.parquet...")
    test_labels = pd.read_parquet(ORIGINAL_DATA_PATH / 'test_labels.parquet')
    vprint(f"    ✓ Loaded {len(test_labels):,} labels")
    
    total_rows = len(train_values) + len(val_values) + len(test_values)
    vprint(f"\n  Total original data: {total_rows:,} samples")
    
    return (train_values, train_labels), (val_values, val_labels), (test_values, test_labels)


def filter_and_sample(values_df, labels_df, n_samples):
    """Filter to selected types and sample n_samples per type.
    
    Returns:
        tuple: (sampled_values, sampled_labels, statistics_dict)
    """
    print(f"\nFiltering to {len(SELECTED_TYPES)} types...")
    vprint(f"  Target types: {', '.join(SELECTED_TYPES)}")
    vprint(f"  Input data shape: {len(labels_df):,} rows")
    
    original_size = len(labels_df)
    
    # Convert labels to lowercase for consistency
    vprint("  Converting labels to lowercase...")
    labels_df['type'] = labels_df['type'].str.lower()
    
    # Filter to selected types
    vprint("  Applying filter mask...")
    mask = labels_df['type'].isin(SELECTED_TYPES)
    filtered_values = values_df[mask].reset_index(drop=True)
    filtered_labels = labels_df[mask].reset_index(drop=True)
    
    filtered_size = len(filtered_labels)
    print(f"Total samples after filtering: {filtered_size:,}")
    
    # Get distribution before sampling
    if VERBOSE:
        print("\n  Pre-sampling distribution:")
        dist = filtered_labels['type'].value_counts().sort_index()
        for typ, count in dist.items():
            print(f"    {typ:12s}: {count:,} samples")
    
    # Sample n_samples per type
    print(f"\nSampling {n_samples} per type...")
    sampled_indices = []
    
    for semantic_type in SELECTED_TYPES:
        type_mask = filtered_labels['type'] == semantic_type
        type_indices = filtered_labels[type_mask].index.tolist()
        
        # Sample with replacement if not enough samples
        n_available = len(type_indices)
        vprint(f"  Processing '{semantic_type}': {n_available:,} available")
        
        if n_available >= n_samples:
            sampled = np.random.choice(type_indices, size=n_samples, replace=False)
            vprint(f"    ✓ Sampled {n_samples} (no replacement)")
        else:
            print(f"  Warning: {semantic_type} has only {n_available} samples, using all")
            sampled = type_indices
            vprint(f"    ⚠ Using all {n_available} samples")
        
        sampled_indices.extend(sampled)
    
    vprint(f"\n  Total sampled indices: {len(sampled_indices):,}")
    
    # Extract sampled data
    vprint("  Extracting sampled data...")
    sampled_values = filtered_values.loc[sampled_indices].reset_index(drop=True)
    sampled_labels = filtered_labels.loc[sampled_indices].reset_index(drop=True)
    
    vprint(f"  ✓ Final shape: {len(sampled_values):,} rows")
    
    # Collect statistics
    stats = {
        'original_size': original_size,
        'filtered_size': filtered_size,
        'sampled_size': len(sampled_labels),
        'reduction_percent': ((original_size - len(sampled_labels)) / original_size * 100)
    }
    
    return sampled_values, sampled_labels, stats


def plot_class_distribution(train_labels, val_labels, test_labels, output_path):
    """Generate and save class distribution charts."""
    vprint("\nGenerating class distribution charts...")
    
    # Set style
    sns.set_style('whitegrid')
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Class Distribution Across Datasets', fontsize=16, fontweight='bold')
    
    datasets = [
        ('Training Set', train_labels, axes[0]),
        ('Validation Set', val_labels, axes[1]),
        ('Test Set', test_labels, axes[2])
    ]
    
    for title, labels, ax in datasets:
        counts = labels['type'].value_counts().sort_index()
        
        # Create bar plot
        bars = ax.bar(range(len(counts)), counts.values, color='steelblue', alpha=0.8)
        
        # Customize
        ax.set_title(f'{title}\n({len(labels):,} total samples)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Semantic Type', fontsize=10)
        ax.set_ylabel('Sample Count', fontsize=10)
        ax.set_xticks(range(len(counts)))
        ax.set_xticklabels(counts.index, rotation=45, ha='right', fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    # Save figure
    output_file = output_path / 'class_distribution.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    vprint(f"  ✓ Saved: {output_file}")
    
    print(f"✓ Class distribution chart saved: {output_file}")
    
    plt.close()


def plot_dataset_sizes(train_stats, val_stats, test_stats, output_path):
    """Generate and save dataset size comparison charts."""
    vprint("\nGenerating dataset size comparison charts...")
    
    # Set style
    sns.set_style('whitegrid')
    
    # Prepare data
    stages = ['Original', 'Filtered', 'Sampled']
    train_sizes = [train_stats['original_size'], train_stats['filtered_size'], train_stats['sampled_size']]
    val_sizes = [val_stats['original_size'], val_stats['filtered_size'], val_stats['sampled_size']]
    test_sizes = [test_stats['original_size'], test_stats['filtered_size'], test_stats['sampled_size']]
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Dataset Size Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Grouped bar chart
    x = np.arange(len(stages))
    width = 0.25
    
    bars1 = ax1.bar(x - width, train_sizes, width, label='Train', color='#2E86AB', alpha=0.8)
    bars2 = ax1.bar(x, val_sizes, width, label='Validation', color='#A23B72', alpha=0.8)
    bars3 = ax1.bar(x + width, test_sizes, width, label='Test', color='#F18F01', alpha=0.8)
    
    ax1.set_title('Sample Counts by Processing Stage', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Processing Stage', fontsize=11)
    ax1.set_ylabel('Number of Samples', fontsize=11)
    ax1.set_xticks(x)
    ax1.set_xticklabels(stages)
    ax1.legend(fontsize=10)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height):,}',
                    ha='center', va='bottom', fontsize=8, rotation=0)
    
    # Plot 2: Reduction percentages
    reductions = [
        train_stats['reduction_percent'],
        val_stats['reduction_percent'],
        test_stats['reduction_percent']
    ]
    datasets = ['Train', 'Validation', 'Test']
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    
    bars = ax2.bar(datasets, reductions, color=colors, alpha=0.8)
    ax2.set_title('Data Reduction by Dataset', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Dataset', fontsize=11)
    ax2.set_ylabel('Reduction (%)', fontsize=11)
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylim(0, 100)
    
    # Add percentage labels
    for bar, reduction in zip(bars, reductions):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{reduction:.1f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    # Save figure
    output_file = output_path / 'dataset_sizes.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    vprint(f"  ✓ Saved: {output_file}")
    
    print(f"✓ Dataset size comparison chart saved: {output_file}")
    
    plt.close()


def main():
    """Main data preparation pipeline."""
    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Prepare subset of Sherlock data for prototype training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python prepare_data.py                  # Run with default settings
  python prepare_data.py --verbose        # Show detailed progress
  python prepare_data.py -v               # Short form of verbose
        """
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show detailed progress information'
    )
    parser.add_argument(
        '--no-viz',
        action='store_true',
        help='Skip visualization generation (charts will not be created)'
    )
    args = parser.parse_args()
    
    # Set global verbose flag
    global VERBOSE
    VERBOSE = args.verbose
    
    if VERBOSE:
        print("="*50)
        print("VERBOSE MODE ENABLED")
        print("="*50)
        print(f"Configuration:")
        print(f"  Selected types: {len(SELECTED_TYPES)}")
        print(f"  Samples per type (train): {SAMPLES_PER_TYPE}")
        print(f"  Samples per type (val/test): 150")
        print(f"  Random seed: {RANDOM_SEED}")
        print(f"  Output path: {OUTPUT_PATH}")
        print("="*50 + "\n")
    
    np.random.seed(RANDOM_SEED)
    vprint(f"Random seed set to: {RANDOM_SEED}")
    
    # Create output directory
    vprint(f"\nCreating output directory: {OUTPUT_PATH}")
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    vprint(f"  ✓ Directory ready")
    
    # Load original data
    (train_vals, train_labs), (val_vals, val_labs), (test_vals, test_labs) = load_original_data()
    
    # Process train set (1000 per type)
    print("\n" + "="*50)
    print("Processing TRAIN set")
    print("="*50)
    train_values, train_labels, train_stats = filter_and_sample(train_vals, train_labs, SAMPLES_PER_TYPE)
    
    # Process validation set (150 per type)
    print("\n" + "="*50)
    print("Processing VALIDATION set")
    print("="*50)
    val_values, val_labels, val_stats = filter_and_sample(val_vals, val_labs, 150)
    
    # Process test set (150 per type)
    print("\n" + "="*50)
    print("Processing TEST set")
    print("="*50)
    test_values, test_labels, test_stats = filter_and_sample(test_vals, test_labs, 150)
    
    # Save processed data
    print("\n" + "="*50)
    print("Saving processed data...")
    print("="*50)
    
    vprint(f"\n  Writing train_values.parquet...")
    train_values.to_parquet(OUTPUT_PATH / 'train_values.parquet')
    vprint(f"    ✓ Written: {OUTPUT_PATH / 'train_values.parquet'}")
    
    vprint(f"  Writing train_labels.parquet...")
    train_labels.to_parquet(OUTPUT_PATH / 'train_labels.parquet')
    print(f"✓ Saved train set: {len(train_labels):,} samples")
    
    vprint(f"\n  Writing val_values.parquet...")
    val_values.to_parquet(OUTPUT_PATH / 'val_values.parquet')
    vprint(f"    ✓ Written: {OUTPUT_PATH / 'val_values.parquet'}")
    
    vprint(f"  Writing val_labels.parquet...")
    val_labels.to_parquet(OUTPUT_PATH / 'val_labels.parquet')
    print(f"✓ Saved validation set: {len(val_labels):,} samples")
    
    vprint(f"\n  Writing test_values.parquet...")
    test_values.to_parquet(OUTPUT_PATH / 'test_values.parquet')
    vprint(f"    ✓ Written: {OUTPUT_PATH / 'test_values.parquet'}")
    
    vprint(f"  Writing test_labels.parquet...")
    test_labels.to_parquet(OUTPUT_PATH / 'test_labels.parquet')
    print(f"✓ Saved test set: {len(test_labels):,} samples")
    
    # Print summary statistics
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print("\nClass distribution (train set):")
    print(train_labels['type'].value_counts().sort_index())
    
    if VERBOSE:
        print("\nClass distribution (validation set):")
        print(val_labels['type'].value_counts().sort_index())
        print("\nClass distribution (test set):")
        print(test_labels['type'].value_counts().sort_index())
    
    print(f"\nTotal samples:")
    print(f"  Train:      {len(train_labels):,}")
    print(f"  Validation: {len(val_labels):,}")
    print(f"  Test:       {len(test_labels):,}")
    total = len(train_labels) + len(val_labels) + len(test_labels)
    print(f"  TOTAL:      {total:,}")
    
    if VERBOSE:
        print(f"\nFiles saved to: {OUTPUT_PATH.absolute()}")
        print(f"Disk space used: ~{(total * 0.001):.1f} MB (estimated)")
    
    # Generate visualizations
    if not args.no_viz:
        print("\n" + "="*50)
        print("Generating visualizations...")
        print("="*50)
        
        try:
            plot_class_distribution(train_labels, val_labels, test_labels, OUTPUT_PATH)
            plot_dataset_sizes(train_stats, val_stats, test_stats, OUTPUT_PATH)
            
            if VERBOSE:
                print("\nVisualization files created:")
                print(f"  - {OUTPUT_PATH / 'class_distribution.png'}")
                print(f"  - {OUTPUT_PATH / 'dataset_sizes.png'}")
        except Exception as e:
            print(f"\n⚠ Warning: Visualization generation failed: {e}")
            print("  Data files were saved successfully.")
    else:
        vprint("\nSkipping visualization generation (--no-viz flag set)")
    
    print("\n" + "="*50)
    print("✓ Data preparation complete!")
    print("="*50)
    if not args.no_viz:
        print(f"\nView results:")
        print(f"  Data files: {OUTPUT_PATH.absolute()}")
        print(f"  Charts: {OUTPUT_PATH.absolute()}\\*.png")


if __name__ == '__main__':
    main()
