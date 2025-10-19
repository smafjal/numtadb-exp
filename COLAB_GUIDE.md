# Google Colab Training Guide

Train your NumtaDB model on Google Colab's free GPU in ~60 minutes.

---

## Prerequisites

**Before starting, you need:**
1. Kaggle API credentials (`kaggle.json`) - Download from https://www.kaggle.com/settings
2. Training package - Run `./notebooks/prepare_for_colab.sh` locally

---

## Step 1: Prepare Locally

```bash
cd /Users/afjal/Documents/Workspace/MLProject/numtadb-exp
./notebooks/prepare_for_colab.sh
```

This creates `trainer_package.zip` (63KB).

---

## Step 2: Upload Notebook to Colab

1. Go to https://colab.research.google.com/
2. **File → Upload notebook**
3. Select `notebooks/NumtaDB_Colab_Training.ipynb`
4. **Runtime → Change runtime type → GPU → Save**

---

## Step 3: Run the Notebook

Run each cell in order:

### Cell 1: Check GPU
```python
import torch
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
```

### Cell 2: Mount Google Drive (recommended)
```python
from google.colab import drive
import os
drive.mount('/content/drive')
os.makedirs('/content/drive/MyDrive/numtadb-project', exist_ok=True)
os.chdir('/content/drive/MyDrive/numtadb-project')
```

### Cell 3: Install Dependencies
```python
!pip install -q kaggle pandas matplotlib seaborn scikit-learn tqdm Pillow
```

### Cell 4: Upload Kaggle Credentials
Upload your `kaggle.json` file when prompted.

### Cell 5: Download Dataset
Downloads ~300MB NumtaDB dataset (2-3 minutes).

### Cell 6: Upload Trainer Package
Upload your `trainer_package.zip` file when prompted.

### Cell 7: Configure Training
```python
import sys
sys.path.insert(0, os.getcwd())
from trainer.config import Config

Config.BATCH_SIZE = 64          # Reduce to 32 or 16 if OOM error
Config.NUM_EPOCHS = 30          # Increase for better accuracy
Config.NUM_WORKERS = 2
Config.MODEL_NAME = 'mobilenetv2'
Config.create_dirs()
```

### Cell 8: Load Dataset
Creates train, validation, and test data loaders.

### Cell 9: Train Model ⭐
Main training step - takes 30-60 minutes.

### Cell 10: Evaluate
Tests model on unseen data.

### Cell 11: Visualize Results
Generates training plots.

### Cell 12: Download Model
Downloads your trained model (~14MB).

---

## Configuration Options

Modify in Cell 7:

| Parameter | Default | Options | Purpose |
|-----------|---------|---------|---------|
| `BATCH_SIZE` | 64 | 16, 32, 64, 128 | Larger = faster, more memory |
| `NUM_EPOCHS` | 30 | 10-100 | More = better accuracy, longer training |
| `LEARNING_RATE` | 0.001 | 0.0001-0.01 | Training speed |
| `MODEL_NAME` | mobilenetv2 | alexnet, mobilenetv2 | Architecture |

---

## Expected Results

- **Training Time:** 30-60 minutes (30 epochs)
- **Test Accuracy:** 96-98%
- **Model Size:** ~14 MB
- **GPU:** Tesla T4 (16GB) on free tier

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| No GPU | Runtime → Change runtime type → GPU → Save |
| Out of Memory | Set `Config.BATCH_SIZE = 16` |
| Kaggle 403 error | Re-download kaggle.json, accept dataset terms on Kaggle |
| Module not found | Re-upload trainer_package.zip |
| Session timeout | Files saved in Google Drive, can resume |

---

## Quick Commands

```bash
# Prepare package
./notebooks/prepare_for_colab.sh

# Check package
ls -lh trainer_package.zip
unzip -l trainer_package.zip
```

---

## Files Generated

After training:
- `checkpoints/best_model.pth` - Best model (highest validation accuracy)
- `logs/training_metrics.csv` - Training history
- `logs/training_plots.png` - Visualization plots

All saved in Google Drive: `/content/drive/MyDrive/numtadb-project/`

---

**Ready to start?** Upload `notebooks/NumtaDB_Colab_Training.ipynb` to https://colab.research.google.com/

