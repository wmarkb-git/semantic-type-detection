# Sherlock Modern - Quick Start Guide

> **⚠️ IMPORTANT:** This project requires **Python 3.10.x**. Check your version with `python --version`. If you have a different version, download Python 3.10.11 from https://www.python.org/downloads/release/python-31011/

## Step 1: Create Virtual Environment

Open PowerShell in the `sherlock-modern` directory:

# CRITICAL: Use py -3.10 to avoid defaulting to Python 3.7
Remove-Item -Recurse -Force venv -ErrorAction SilentlyContinue
py -3.10 -m venv venv


# Allow scripts for the current session
Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process


# Activate it
.\venv\Scripts\activate

# VERIFY you're using Python 3.10 (not 3.7!)
python --version
# MUST show: Python 3.10.11
# If it shows 3.7.9, the venv creation failed - try again

# You should see (venv) at the start of your prompt
```

## Step 2: Install Dependencies

```powershell
# Upgrade pip
python -m pip install --upgrade pip

# Install requirements (this may take 5-10 minutes)
pip install -r requirements.txt
```

## Step 3: Prepare Data

```powershell
# This will extract 10 types from the original Sherlock data
# Train: 1000 samples per type = 10,000 total
# Val: 150 samples per type = 1,500 total
# Test: 150 samples per type = 1,500 total

python prepare_data.py
```

**Expected output:** You should see progress bars and a summary showing class distribution.

## Step 4: Train Model

```powershell
# Train multi-input model (recommended)
python train.py --model_type multi_input --epochs 50

# OR train baseline model (faster, simpler)
# python train.py --model_type baseline --epochs 30
```

**Training time:** 
- Multi-input: ~10-15 minutes on CPU, ~2-3 minutes on GPU
- Baseline: ~5-7 minutes on CPU, ~1-2 minutes on GPU

**Expected output:**
- Training progress with accuracy and loss
- Best model saved to `outputs/best_model.keras`
- Training history saved to `outputs/training_history.csv`

## Step 5: Evaluate Model

```powershell
python evaluate.py
```

**Expected output:**
- F1 scores per class
- Overall weighted F1 (target: 0.82-0.87 for 10 types)
- Confusion matrix saved as image
- Full report in `outputs/evaluation_results/`

## Expected Performance

| Model Type | Expected F1 (10 types) | Training Time (CPU) |
|-----------|----------------------|-------------------|
| Baseline (embeddings only) | 0.75-0.80 | 5-7 min |
| Multi-input (embeddings + stats) | 0.82-0.87 | 10-15 min |

## Troubleshooting

### "Module not found" errors
```powershell
# Make sure virtual environment is activated
.\venv\Scripts\activate

# Reinstall requirements
pip install -r requirements.txt
```

### "CUDA not available" warnings
This is normal if you don't have a GPU. The model will run on CPU (slower but still works).

### Memory errors
Reduce batch size in training:
```powershell
python train.py --batch_size 16
```

## Next Steps

1. **Experiment with hyperparameters:**
   ```powershell
   python train.py --dropout 0.4 --learning_rate 0.0005
   ```

2. **Try different embedding models:**
   ```powershell
   python train.py --embedding_model "paraphrase-MiniLM-L6-v2"
   ```

3. **Visualize results:**
   Check `outputs/evaluation_results/confusion_matrix.png`

4. **Use the demo notebook:**
   ```powershell
   jupyter notebook notebooks/01_prototype_demo.ipynb
   ```

## File Structure After Running

```
sherlock-modern/
├── venv/                    # Virtual environment
├── data/
│   └── prototype/          # Processed data (created by prepare_data.py)
│       ├── train_values.parquet
│       ├── train_labels.parquet
│       ├── val_values.parquet
│       ├── val_labels.parquet
│       ├── test_values.parquet
│       └── test_labels.parquet
├── outputs/                # Training outputs (created by train.py)
│   ├── best_model.keras
│   ├── final_model.keras
│   ├── config.json
│   ├── classes.npy
│   ├── training_history.csv
│   └── evaluation_results/  # Evaluation outputs (created by evaluate.py)
│       ├── classification_report.csv
│       ├── predictions.csv
│       ├── confusion_matrix.png
│       └── summary.json
```
