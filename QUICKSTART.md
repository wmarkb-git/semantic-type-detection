# Sherlock Modern - Quick Start Guide

> **⚠️ IMPORTANT:** This project requires **Python 3.10+**. Check your version with `python --version`.

## Prerequisites

- Python 3.10 or higher ([Download here](https://www.python.org/downloads/))
- ~1GB disk space for dependencies
- ~440MB for data files

## Data Setup

> **⚠️ IMPORTANT:** The Sherlock dataset is NOT included in this repository. You must download it separately before running the scripts.

### Step 1: Download the Sherlock Dataset

The original Sherlock dataset is available on GitHub:

```bash
# Clone the original Sherlock repository
git clone https://github.com/mitmedialab/sherlock-project.git
```

### Step 2: Copy Data Files

**Windows:**
```powershell
# Create the directory structure
mkdir data\data\raw -Force

# Copy parquet files
cp sherlock-project\data\*.parquet data\data\raw\
```

**macOS/Linux:**
```bash
# Create the directory structure
mkdir -p data/data/raw

# Copy parquet files
cp sherlock-project/data/*.parquet data/data/raw/
```

### Step 3: Verify Data Structure

You should have the following files:
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

**Dataset Info:**
- **Size:** ~440MB (6 files)
- **Source:** [github.com/mitmedialab/sherlock-project](https://github.com/mitmedialab/sherlock-project)
- **Contents:** 686,765 data columns with 78 semantic types

---

## Platform-Specific Setup

### Windows

```powershell
# Navigate to project directory
cd path\to\sherlock-modern-project

# Create Python 3.10 virtual environment
py -3.10 -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# If you get an execution policy error, run:
Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process

# Verify Python version (should be 3.10.x)
python --version

# Upgrade pip
python -m pip install --upgrade pip

# Install dependencies (this may take 5-10 minutes)
pip install -r requirements.txt
```

### macOS/Linux

```bash
# Navigate to project directory
cd path/to/sherlock-modern-project

# Create Python 3.10 virtual environment
python3.10 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Verify Python version (should be 3.10.x)
python --version

# Upgrade pip
python -m pip install --upgrade pip

# Install dependencies (this may take 5-10 minutes)
pip install -r requirements.txt
```

## Step-by-Step Workflow

### 1. Prepare Data

Extract and process the Sherlock dataset for 10 semantic types:

```bash
python prepare_data.py
```

**What this does:**
- Loads original Sherlock dataset from parquet files
- Filters to 10 semantic types (person, place, description, etc.)
- Creates balanced train/val/test splits:
  - Train: 1,000 samples × 10 types = 10,000 total
  - Val: 150 samples × 10 types = 1,500 total
  - Test: 150 samples × 10 types = 1,500 total
- Saves processed data to `data/prototype/`

**Expected output:** Progress bars and class distribution summary

---

### 2. Train Model

Train the multi-input PyTorch model with embeddings and statistical features:

```bash
# Train with default settings (recommended)
python train.py

# Or customize training parameters
python train.py --epochs 50 --batch_size 32 --learning_rate 0.001
```

**Training time:**
- CPU: ~10-15 minutes
- GPU: ~2-3 minutes (if CUDA available)

**What this does:**
- Extracts SBERT embeddings from column values
- Computes statistical features (mean, std, entropy, etc.)
- Trains multi-input PyTorch neural network
- Saves best model to `outputs/best_model.pth`
- Saves training history to `outputs/training_history.csv`

**Expected output:**
- Training progress with epoch-by-epoch loss and accuracy
- Validation metrics after each epoch
- Model checkpoints saved automatically

---

### 3. Evaluate Model

Evaluate the trained model on the test set:

```bash
python evaluate.py
```

**What this does:**
- Loads best trained model from `outputs/best_model.pth`
- Evaluates on test set (1,500 samples)
- Generates classification report with F1 scores
- Creates confusion matrix visualization
- Saves detailed evaluation results

**Expected output:**
- Per-class F1 scores
- Overall weighted F1 (target: **0.82-0.87** for 10 types)
- Confusion matrix saved to `outputs/evaluation_results/confusion_matrix.png`
- Full report in `outputs/evaluation_results/`

---

## Expected Performance

| Metric | Expected Range |
|--------|---------------|
| Overall Accuracy | 82-87% |
| Weighted F1 Score | 0.82-0.87 |
| Training Time (CPU) | 10-15 min |
| Training Time (GPU) | 2-3 min |

## Troubleshooting

### Virtual Environment Not Activating

**Windows:**
```powershell
Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process
.\venv\Scripts\Activate.ps1
```

**macOS/Linux:**
```bash
source venv/bin/activate
```

### "Module not found" Errors

```bash
# Ensure virtual environment is activated (you should see (venv) in prompt)
# Then reinstall requirements
pip install -r requirements.txt
```

### CUDA Not Available

This is normal if you don't have a NVIDIA GPU. PyTorch will automatically use CPU (slower but works fine).

### Memory Errors

Reduce batch size:
```bash
python train.py --batch_size 16
```

## Advanced Usage

### Custom Hyperparameters

```bash
# Fine-tune dropout and learning rate
python train.py --dropout 0.4 --learning_rate 0.0005 --epochs 100

# Adjust model architecture
python train.py --embedding_dim 384 --hidden_dim 256
```

### Different Embedding Models

```bash
# Use a smaller/faster model
python train.py --embedding_model "paraphrase-MiniLM-L6-v2"

# Use a larger/more accurate model
python train.py --embedding_model "all-mpnet-base-v2"
```

### Verbose Output

```bash
# See detailed training progress
python train.py --verbose

# See detailed evaluation metrics
python evaluate.py --verbose
```

## Project Structure After Running

```
sherlock-modern-project/
├── venv/                           # Virtual environment
├── data/
│   └── prototype/                 # Processed data
│       ├── train_values.parquet
│       ├── train_labels.parquet
│       ├── val_values.parquet
│       ├── val_labels.parquet
│       ├── test_values.parquet
│       └── test_labels.parquet
├── outputs/                       # Training outputs
│   ├── best_model.pth            # Best model checkpoint
│   ├── final_model.pth           # Final model after training
│   ├── config.json               # Model configuration
│   ├── classes.npy               # Class label mapping
│   ├── training_history.csv      # Training metrics
│   ├── training_accuracy.png     # Accuracy plot
│   ├── training_loss.png         # Loss plot
│   └── evaluation_results/       # Evaluation outputs
│       ├── classification_report.csv
│       ├── predictions.csv
│       ├── confusion_matrix.png
│       └── summary.json
```

## Next Steps

1. **Visualize Results:** Check the plots in `outputs/` directory
2. **Experiment:** Try different hyperparameters and architectures
3. **Extend:** Add more semantic types from the full Sherlock dataset
4. **Deploy:** Use the trained model for inference on new data

## Key Technologies

- **PyTorch 2.9.1** - Neural network framework
- **Sentence-Transformers 5.2.0** - SBERT embeddings
- **scikit-learn 1.7.2** - ML utilities and metrics
- **pandas 2.3.3** - Data manipulation
- **matplotlib 3.10.8** - Visualizations
