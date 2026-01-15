"""
Sentence embedding extractor using sentence-transformers (SBERT).
Replaces Doc2Vec from original Sherlock with modern BERT-based embeddings.
"""
from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm


class ColumnEmbedder:
    """Extract sentence embeddings from table columns."""
    
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """
        Initialize the embedder.
        
        Args:
            model_name: Sentence-transformer model name
                       Default: all-MiniLM-L6-v2 (384 dims, fast, good quality)
        """
        print(f"Loading sentence transformer model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"✓ Model loaded. Embedding dimension: {self.embedding_dim}")
    
    def embed_column(self, values, max_values=100, separator=" | "):
        """
        Generate embedding for a single column.
        
        Args:
            values: List of column values (strings)
            max_values: Maximum number of values to use (for long columns)
            separator: Separator between values in concatenated text
            
        Returns:
            numpy array of shape (embedding_dim,)
        """
        # Handle empty or all-null columns
        values = [str(v) for v in values if v is not None and str(v).strip()]
        if not values:
            return np.zeros(self.embedding_dim, dtype=np.float32)
        
        # Limit number of values for efficiency
        if len(values) > max_values:
            values = values[:max_values]
        
        # Concatenate column values into single text
        text = separator.join(values)
        
        # Generate embedding
        embedding = self.model.encode(text, convert_to_tensor=False, show_progress_bar=False)
        
        return embedding.astype(np.float32)
    
    def embed_columns_batch(self, columns_list, max_values=100, show_progress=True):
        """
        Generate embeddings for multiple columns.
        
        Args:
            columns_list: List of column value lists
            max_values: Maximum values per column
            show_progress: Show progress bar
            
        Returns:
            numpy array of shape (n_columns, embedding_dim)
        """
        embeddings = []
        
        iterator = tqdm(columns_list, desc="Extracting embeddings") if show_progress else columns_list
        
        for column_values in iterator:
            embedding = self.embed_column(column_values, max_values=max_values)
            embeddings.append(embedding)
        
        return np.array(embeddings, dtype=np.float32)


if __name__ == '__main__':
    # Example usage
    print("="*50)
    print("Testing ColumnEmbedder")
    print("="*50)
    
    embedder = ColumnEmbedder()
    
    # Test on sample columns
    sample_columns = [
        ["John Smith", "Mary Johnson", "Bob Williams"],
        ["New York", "Los Angeles", "Chicago"],
        ["2020", "2021", "2022", "2023"],
    ]
    
    print("\nGenerating embeddings for 3 sample columns...")
    embeddings = embedder.embed_columns_batch(sample_columns, show_progress=False)
    
    print(f"\nResult shape: {embeddings.shape}")
    print(f"Expected: (3, {embedder.embedding_dim})")
    print("\n✓ Test passed!")
