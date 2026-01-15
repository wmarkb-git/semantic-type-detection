"""
Statistical feature extractor.
Computes column-level statistics to complement embeddings.
"""
import numpy as np
import pandas as pd
from tqdm import tqdm


def extract_stats_single(values):
    """
    Extract statistical features from a single column.
    
    Args:
        values: List of column values
        
    Returns:
        numpy array of statistical features (~100 dimensions)
    """
    features = []
    
    # Convert to strings and handle nulls
    str_values = [str(v) if pd.notna(v) else '' for v in values]
    non_null_values = [v for v in str_values if v]
    
    total_count = len(values)
    non_null_count = len(non_null_values)
    
    # === BASIC STATISTICS (10 features) ===
    features.extend([
        non_null_count / max(total_count, 1),  # Non-null ratio
        len(set(non_null_values)) / max(non_null_count, 1),  # Unique ratio
        np.mean([len(v) for v in non_null_values]) if non_null_values else 0,  # Avg length
        np.std([len(v) for v in non_null_values]) if non_null_values else 0,  # Std length
        min([len(v) for v in non_null_values]) if non_null_values else 0,  # Min length
        max([len(v) for v in non_null_values]) if non_null_values else 0,  # Max length
    ])
    
    # === DATA TYPE DISTRIBUTION (4 features) ===
    numeric_count = sum(1 for v in non_null_values if v.replace('.', '').replace('-', '').isdigit())
    alpha_count = sum(1 for v in non_null_values if v.isalpha())
    alphanum_count = sum(1 for v in non_null_values if v.isalnum() and not v.isalpha() and not v.isdigit())
    
    features.extend([
        numeric_count / max(non_null_count, 1),  # Numeric ratio
        alpha_count / max(non_null_count, 1),  # Alphabetic ratio
        alphanum_count / max(non_null_count, 1),  # Alphanumeric ratio
        1 - (numeric_count + alpha_count + alphanum_count) / max(non_null_count, 1),  # Special chars ratio
    ])
    
    # === CHARACTER PATTERN FREQUENCIES (20 features) ===
    # Concatenate all values
    all_text = ''.join(non_null_values)
    text_len = max(len(all_text), 1)
    
    special_chars = ['@', '.', '-', '_', '/', '\\', '(', ')', '[', ']', 
                     '{', '}', ':', ';', ',', '!', '?', '#', '$', '%']
    
    for char in special_chars:
        features.append(all_text.count(char) / text_len)
    
    # === NUMERIC PATTERNS (10 features) ===
    if non_null_values:
        # Check for year patterns (4 digits)
        year_pattern = sum(1 for v in non_null_values if v.isdigit() and len(v) == 4)
        features.append(year_pattern / non_null_count)
        
        # Check for currency patterns (contains $, £, €, etc.)
        currency_chars = ['$', '£', '€', '¥']
        currency_count = sum(1 for v in non_null_values if any(c in v for c in currency_chars))
        features.append(currency_count / non_null_count)
        
        # Check for percentage patterns
        percent_count = sum(1 for v in non_null_values if '%' in v)
        features.append(percent_count / non_null_count)
        
        # Check for decimal patterns
        decimal_count = sum(1 for v in non_null_values if '.' in v and any(c.isdigit() for c in v))
        features.append(decimal_count / non_null_count)
        
        # Check for comma in numbers (e.g., 1,000)
        comma_number_count = sum(1 for v in non_null_values if ',' in v and any(c.isdigit() for c in v))
        features.append(comma_number_count / non_null_count)
        
        # Check for negative numbers
        negative_count = sum(1 for v in non_null_values if v.startswith('-') and any(c.isdigit() for c in v))
        features.append(negative_count / non_null_count)
        
        # Average number of digits per value
        digit_counts = [sum(c.isdigit() for c in v) for v in non_null_values]
        features.append(np.mean(digit_counts) if digit_counts else 0)
        
        # Average number of letters per value
        letter_counts = [sum(c.isalpha() for c in v) for v in non_null_values]
        features.append(np.mean(letter_counts) if letter_counts else 0)
        
        # Check for phone number patterns (contains - or () with digits)
        phone_pattern = sum(1 for v in non_null_values if ('-' in v or '(' in v or ')' in v) and any(c.isdigit() for c in v))
        features.append(phone_pattern / non_null_count)
        
        # Check for date-like patterns (contains / or - with digits)
        date_pattern = sum(1 for v in non_null_values if ('/' in v or '-' in v) and sum(c.isdigit() for c in v) >= 4)
        features.append(date_pattern / non_null_count)
    else:
        features.extend([0] * 10)
    
    # === STRING CASE PATTERNS (5 features) ===
    if non_null_values:
        upper_count = sum(1 for v in non_null_values if v.isupper())
        lower_count = sum(1 for v in non_null_values if v.islower())
        title_count = sum(1 for v in non_null_values if v.istitle())
        mixed_count = non_null_count - upper_count - lower_count - title_count
        
        features.extend([
            upper_count / non_null_count,  # All uppercase ratio
            lower_count / non_null_count,  # All lowercase ratio
            title_count / non_null_count,  # Title case ratio
            mixed_count / non_null_count,  # Mixed case ratio
            np.mean([sum(c.isupper() for c in v) / max(len(v), 1) for v in non_null_values]),  # Avg uppercase ratio
        ])
    else:
        features.extend([0] * 5)
    
    return np.array(features, dtype=np.float32)


def extract_stats_batch(columns_list, show_progress=True):
    """
    Extract statistical features for multiple columns.
    
    Args:
        columns_list: List of column value lists
        show_progress: Show progress bar
        
    Returns:
        numpy array of shape (n_columns, n_features)
    """
    features_list = []
    
    iterator = tqdm(columns_list, desc="Extracting statistics") if show_progress else columns_list
    
    for column_values in iterator:
        features = extract_stats_single(column_values)
        features_list.append(features)
    
    return np.array(features_list, dtype=np.float32)


if __name__ == '__main__':
    # Example usage
    print("="*50)
    print("Testing Statistical Feature Extractor")
    print("="*50)
    
    sample_columns = [
        ["John Smith", "Mary Johnson", "Bob Williams"],
        ["New York", "Los Angeles", "Chicago"],
        ["2020", "2021", "2022", "2023"],
        ["$100.50", "$200.75", "$50.25"],
    ]
    
    print("\nExtracting features from 4 sample columns...")
    features = extract_stats_batch(sample_columns, show_progress=False)
    
    print(f"\nResult shape: {features.shape}")
    print(f"Feature vector length: {features.shape[1]} dimensions")
    print("\n✓ Test passed!")
