# Sherlock Modern - Semantic Type Detection

A modernized, standalone implementation of the Sherlock semantic type detection system using current, supported libraries and PyTorch.

## Project Overview

This project uses:
- **Python 3.10+** (modern, supported version)
- **PyTorch** for neural networks
- **Sentence-BERT** for text embeddings
- **Focused scope**: 10 semantic types on a subset of data

## Project Structure

```
sherlock-modern-project/
├── data/                    # Data directory
│   └── data/
│       ├── raw/            # Original parquet files
│       └── processed/      # Processed outputs
├── features/               # Feature extraction modules
│   ├── embeddings.py      # SBERT embeddings
│   ├── statistics.py      # Statistical features
│   └── __init__.py
├── models/                 # Neural network models
│   ├── multi_input.py     # Multi-input PyTorch architecture
│   └── __init__.py
├── prepare_data.py         # Data preparation script
├── train.py               # Model training script
├── evaluate.py            # Model evaluation script
├── requirements.txt       # Python dependencies
└── venv/                  # Python 3.10 virtual environment
```

## Data Setup

This project requires the original Sherlock dataset. The data is not included in this repository due to its size (~440MB).

### Download the Dataset

1. **Clone the original Sherlock dataset:**
   ```bash
   git clone https://github.com/mitmedialab/sherlock-project.git
   ```

2. **Copy the data files to your project:**
   
   **Windows:**
   ```powershell
   # Create the data directory
   mkdir -p data/data/raw
   
   # Copy the parquet files from the cloned repository
   cp sherlock-project/data/*.parquet data/data/raw/
   ```
   
   **macOS/Linux:**
   ```bash
   # Create the data directory
   mkdir -p data/data/raw
   
   # Copy the parquet files from the cloned repository
   cp sherlock-project/data/*.parquet data/data/raw/
   ```

3. **Verify the data structure:**
   ```
   data/
   └── data/
       └── raw/
           ├── train_values.parquet
           ├── train_labels.parquet
           ├── val_values.parquet
           ├── val_labels.parquet
           ├── test_values.parquet
           └── test_labels.parquet
   ```

### Alternative: Direct Download

If you prefer not to clone the entire repository, you can download the parquet files directly from the GitHub repository and place them in `data/data/raw/`.

**Dataset Information:**
- **Source:** [MIT Media Lab Sherlock Project](https://github.com/mitmedialab/sherlock-project)
- **Size:** ~440MB (6 parquet files)
- **Contents:** 686,765 data columns with 78 semantic types

## Quick Start

### 1. Setup Environment

**Windows:**
```powershell
# Create Python 3.10 virtual environment
py -3.10 -m venv venv

# Activate it
.\venv\Scripts\Activate.ps1

# Verify Python version (should be 3.10.x)
python --version

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

**macOS/Linux:**
```bash
# Create Python 3.10 virtual environment
python3.10 -m venv venv

# Activate it
source venv/bin/activate

# Verify Python version (should be 3.10.x)
python --version

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Prepare Data

```bash
python prepare_data.py
```

This will:
- Load the original Sherlock dataset
- Filter to 10 semantic types
- Create train/val/test splits
- Save processed data to `data/prototype/`

### 3. Train Model

```bash
python train.py
```

This will:
- Extract SBERT embeddings + statistical features
- Train a multi-input PyTorch neural network
- Save the trained model to `outputs/best_model.pth`

### 4. Evaluate Model

```bash
python evaluate.py
```

This will:
- Load the trained model
- Evaluate on test set
- Display accuracy and classification report
- Generate visualizations and evaluation reports

## What's Different from Original Sherlock?

✅ **Python 3.10** instead of 3.7 (EOL)  
✅ **PyTorch** instead of TensorFlow/Keras  
✅ **Sentence-BERT** instead of Doc2Vec  
✅ **10 semantic types** instead of all 78  
✅ **No multiprocessing issues** - clean, modern code  
✅ **Cross-platform** - works on Windows, macOS, and Linux

## Requirements

- Python 3.10 or higher
- ~1GB disk space for dependencies
- ~440MB for data files

## Benefits of This Setup

1. **Modern stack** - PyTorch 2.9.1 with latest features
2. **Cross-platform** - Works seamlessly on Windows, macOS, and Linux
3. **Well-maintained libraries** - All dependencies actively supported
4. **Reproducible** - Clear setup process with locked versions
5. **Faster** - Modern PyTorch optimizations

## Next Steps

After running the scripts successfully, you can:
- Experiment with different model architectures
- Add more semantic types
- Try different embedding models
- Fine-tune hyperparameters
- Explore the generated visualizations in `outputs/`
