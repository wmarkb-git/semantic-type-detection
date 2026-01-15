# Sherlock Modern - Independent Project

This is a modernized, standalone implementation of the Sherlock semantic type detection system using current, supported libraries.

## Project Overview

This project uses:
- **Python 3.10+** (modern, supported version)
- **Sentence-BERT** for text embeddings
- **TensorFlow/Keras** for neural networks
- **Focused scope**: 10 semantic types on a subset of data

## Project Structure

```
sherlock-modern-project/
├── data/                    # Data directory (copied from original)
│   └── data/
│       ├── raw/            # Original parquet files
│       └── processed/      # Processed outputs
├── features/               # Feature extraction modules
│   ├── embeddings.py      # SBERT embeddings
│   ├── statistics.py      # Statistical features
│   └── __init__.py
├── models/                 # Neural network models
│   ├── multi_input.py     # Multi-input architecture
│   └── __init__.py
├── prepare_data.py         # Data preparation script
├── train.py               # Model training script
├── evaluate.py            # Model evaluation script
├── requirements.txt       # Python dependencies
└── venv/                  # Python 3.10 virtual environment
```

## Quick Start

### 1. Setup Environment

Run the automated setup (from the parent directory):
```powershell
cd C:\Users\getma\projects\msc\irp\sherlock\sherlock-project
.\setup_new_project.ps1
```

Or manually:
```powershell
cd C:\Users\getma\projects\msc\irp\sherlock-modern-project

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

### 2. Prepare Data

```powershell
python prepare_data.py
```

This will:
- Load the original Sherlock dataset
- Filter to 10 semantic types
- Create train/val/test splits
- Save processed data to `data/prototype/`

### 3. Train Model

```powershell
python train.py
```

This will:
- Extract SBERT embeddings + statistical features
- Train a multi-input neural network
- Save the trained model to `models/sherlock_modern.keras`

### 4. Evaluate Model

```powershell
python evaluate.py
```

This will:
- Load the trained model
- Evaluate on test set
- Display accuracy and classification report

## What's Different from Original Sherlock?

✅ **Python 3.10** instead of 3.7 (EOL)  
✅ **Sentence-BERT** instead of Doc2Vec  
✅ **TensorFlow 2.x** instead of old Keras  
✅ **10 semantic types** instead of all 78  
✅ **No multiprocessing issues** - clean, modern code  
✅ **Independent project** - no conflicts with old code

## Requirements

- Python 3.10 or higher
- ~500MB disk space for dependencies
- ~440MB for data files (already copied)

## Benefits of This Setup

1. **Clean environment** - No Python 3.7 conflicts
2. **Modern libraries** - All supported and maintained
3. **Independent** - Can delete old project when ready
4. **Reproducible** - Clear setup process
5. **Faster** - Modern libraries are more efficient

## Next Steps

After running the scripts successfully, you can:
- Experiment with different model architectures
- Add more semantic types
- Try different embedding models
- Compare results with original Sherlock

## Original Project

The original Sherlock project (Python 3.7) is still available at:
`C:\Users\getma\projects\msc\irp\sherlock\sherlock-project`

You can keep it for reference or delete it once you're confident this modern version meets your needs.
