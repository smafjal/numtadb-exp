# Google Colab Training - Quick Start

A simple 3-step guide to train your model on Google Colab.

## 🚀 3 Steps to Train on Colab

### Step 1: Prepare Files (On Your Computer)

```bash
# Navigate to your project
cd /Users/afjal/Documents/Workspace/MLProject/numtadb-exp

# Create trainer.zip
./prepare_for_colab.sh
```

This creates `trainer.zip` which you'll upload to Colab.

### Step 2: Open Colab with GPU

1. Go to https://colab.research.google.com/
2. File > New notebook
3. Runtime > Change runtime type > **GPU** > Save

### Step 3: Copy-Paste and Run

Copy the code blocks from `COLAB_TRAINING_GUIDE.md` into cells and run them in order.

**That's it!** Your model will train in 30-60 minutes.

---

## 📦 What You Need to Upload

1. **kaggle.json** - Your Kaggle API credentials
   - Get from: https://www.kaggle.com/settings (API section)
   
2. **trainer.zip** - Your trainer folder
   - Created by running `./prepare_for_colab.sh`

---

## 🎯 Quick Cell Sequence

Here's the essential code to run (in separate cells):

```python
# 1. GPU Check
import torch
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

# 2. Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# 3. Setup
import os
os.chdir('/content/drive/MyDrive/numtadb-project')
!pip install -q kaggle torch torchvision pandas matplotlib seaborn scikit-learn tqdm Pillow

# 4. Upload Kaggle credentials (kaggle.json)
from google.colab import files
os.makedirs('/root/.kaggle', exist_ok=True)
uploaded = files.upload()  # Upload kaggle.json
!cp kaggle.json /root/.kaggle/kaggle.json && chmod 600 /root/.kaggle/kaggle.json

# 5. Download dataset
os.makedirs('data/raw', exist_ok=True)
os.chdir('data/raw')
!kaggle datasets download -d BengaliAI/numta && unzip -q numta.zip && rm numta.zip
os.chdir('../..')

# 6. Upload trainer folder (trainer.zip)
import zipfile
uploaded = files.upload()  # Upload trainer.zip
with zipfile.ZipFile('trainer.zip', 'r') as z:
    z.extractall('.')

# 7. Import and configure
import sys
sys.path.insert(0, os.getcwd())
from trainer.config import Config
from trainer.train import Trainer
from trainer.dataset import create_dataloaders

Config.BATCH_SIZE = 64
Config.NUM_EPOCHS = 30

# 8. Load data
train_loader, val_loader, test_loader = create_dataloaders(str(Config.DATA_DIR), Config)

# 9. Train
trainer = Trainer(Config)
metrics = trainer.train(train_loader, val_loader)

# 10. Evaluate
test_loss, test_acc = trainer.validate(test_loader)
print(f"Test Accuracy: {test_acc:.2f}%")

# 11. Download model
files.download('checkpoints/best_model.pth')
```

---

## ⚡ Super Quick Version (Single Cell)

If you're experienced, here's a single cell that does everything (you still need to upload files when prompted):

```python
# Setup
from google.colab import drive, files
import os, sys, zipfile, torch

# Mount and setup
drive.mount('/content/drive')
os.makedirs('/content/drive/MyDrive/numtadb-project', exist_ok=True)
os.chdir('/content/drive/MyDrive/numtadb-project')

# Install
!pip install -q kaggle pandas matplotlib seaborn scikit-learn tqdm

# Kaggle setup
os.makedirs('/root/.kaggle', exist_ok=True)
if not os.path.exists('/root/.kaggle/kaggle.json'):
    print("Upload kaggle.json:")
    uploaded = files.upload()
    !cp kaggle.json /root/.kaggle/kaggle.json && chmod 600 /root/.kaggle/kaggle.json

# Download dataset
if not os.path.exists('data/raw/training-a.csv'):
    os.makedirs('data/raw', exist_ok=True)
    os.chdir('data/raw')
    !kaggle datasets download -d BengaliAI/numta && unzip -q numta.zip && rm numta.zip
    os.chdir('../..')

# Upload trainer
if not os.path.exists('trainer'):
    print("Upload trainer.zip:")
    uploaded = files.upload()
    with zipfile.ZipFile('trainer.zip', 'r') as z:
        z.extractall('.')

# Import modules
sys.path.insert(0, os.getcwd())
from trainer.config import Config
from trainer.train import Trainer
from trainer.dataset import create_dataloaders

# Configure
Config.BATCH_SIZE = 64
Config.NUM_EPOCHS = 30
Config.NUM_WORKERS = 2

# Train
print("Loading data...")
train_loader, val_loader, test_loader = create_dataloaders(str(Config.DATA_DIR), Config)

print("Training...")
trainer = Trainer(Config)
metrics = trainer.train(train_loader, val_loader)

# Evaluate
test_loss, test_acc = trainer.validate(test_loader)
print(f"\n🎉 Test Accuracy: {test_acc:.2f}%")

# Download
print("\nDownloading model...")
files.download('checkpoints/best_model.pth')
```

---

## 💡 Pro Tips

- **First Time**: Use the step-by-step guide in `COLAB_TRAINING_GUIDE.md`
- **Repeat Training**: Files in Google Drive persist across sessions
- **GPU Memory**: If OOM error, reduce `Config.BATCH_SIZE` to 32
- **Monitoring**: Watch the training progress in real-time
- **Time Limit**: Colab sessions timeout after 12 hours (plenty of time)

## 📊 Expected Results

- **Training Time**: 30-60 minutes (depends on dataset size and GPU)
- **Test Accuracy**: 95-98% (with full training dataset)
- **Model Size**: ~14 MB
- **GPU**: Usually gets T4 (16GB) on free tier

## ❓ Troubleshooting

| Problem | Solution |
|---------|----------|
| No GPU | Runtime > Change runtime type > GPU |
| Out of Memory | Reduce `Config.BATCH_SIZE` to 32 or 16 |
| Kaggle error | Re-upload kaggle.json, check permissions |
| Module not found | Re-extract trainer.zip, check `sys.path` |
| Session timeout | Use Google Drive to save progress |

---

## 📁 Files You'll Get

After training completes:

1. ✅ `best_model.pth` - Best model (highest validation accuracy)
2. ✅ `mobilenetv2_final.pth` - Final model after all epochs
3. ✅ `training_metrics.csv` - Full training history
4. ✅ `training_results.png` - Visualization plots

All files are saved in Google Drive and can be downloaded anytime.

---

## 🎓 What's Next?

After training:
- Test on new images
- Deploy as web app
- Convert to ONNX for production
- Fine-tune with more data

See the main README.md for more options!

---

**Ready to train? Run `./prepare_for_colab.sh` and let's go! 🚀**

