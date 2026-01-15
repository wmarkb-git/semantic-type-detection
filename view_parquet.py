"""
Quick utility to convert Parquet files to CSV for easy viewing.
"""
import pandas as pd
from pathlib import Path
import argparse


def parquet_to_csv(parquet_path, output_path=None, max_rows=None):
    """
    Convert a Parquet file to CSV.
    
    Args:
        parquet_path: Path to .parquet file
        output_path: Output CSV path (optional, auto-generates if not provided)
        max_rows: Limit number of rows (optional, for large files)
    """
    # Read parquet file
    print(f"Reading {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    
    # Limit rows if specified
    if max_rows:
        df = df.head(max_rows)
        print(f"Limited to first {max_rows} rows")
    
    # Auto-generate output path if not provided
    if output_path is None:
        output_path = Path(parquet_path).with_suffix('.csv')
    
    # Save to CSV
    print(f"Converting to CSV...")
    df.to_csv(output_path, index=False)
    
    print(f"✓ Saved to: {output_path}")
    print(f"  Rows: {len(df):,}")
    print(f"  Columns: {len(df.columns)}")
    print(f"\nFirst few rows:")
    print(df.head())


def preview_parquet(parquet_path, n_rows=10):
    """Just preview the file without converting."""
    print(f"Preview of {parquet_path}:")
    print("="*80)
    
    df = pd.read_parquet(parquet_path)
    
    print(f"\nShape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"\nColumns: {', '.join(df.columns)}")
    print(f"\nFirst {n_rows} rows:")
    print(df.head(n_rows))
    
    if 'type' in df.columns:
        print(f"\nUnique types: {df['type'].nunique()}")
        print(f"Type counts:")
        print(df['type'].value_counts())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='View or convert Parquet files')
    parser.add_argument('file', help='Path to .parquet file')
    parser.add_argument('--convert', action='store_true', 
                        help='Convert to CSV (default: just preview)')
    parser.add_argument('--output', '-o', help='Output CSV path (optional)')
    parser.add_argument('--limit', type=int, help='Limit rows (for large files)')
    parser.add_argument('--rows', type=int, default=10, 
                        help='Number of rows to preview (default: 10)')
    
    args = parser.parse_args()
    
    if args.convert:
        parquet_to_csv(args.file, args.output, args.limit)
    else:
        preview_parquet(args.file, args.rows)
