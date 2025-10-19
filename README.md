# NumtaDB Bengali Digit Recognition

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

Train deep learning models (AlexNet, MobileNetV2) on NumtaDB dataset for Bengali handwritten digit classification.

**🌐 Community**: [BengaliAI](https://www.bengali.ai/) | **📊 Dataset**: [Kaggle NumtaDB](https://www.kaggle.com/datasets/BengaliAI/numta) | **📄 Paper**: [arXiv:1806.02452](https://arxiv.org/abs/1806.02452)

---

## Quick Start

```bash
make help              # Show all available commands
make install           # Install dependencies
make setup-kaggle      # Configure Kaggle API
make download-data     # Download dataset
make train             # Start training
```

### 🔗 **Live Demo:** https://smafjal.github.io/NumtaDB/
### 🔗 **Demo Repo:** https://github.com/smafjal/NumtaDB


## Make Commands

Run `make help` to see all commands. Key commands:

| Command | Description |
|---------|-------------|
| **Setup** | |
| `make install` | Install Python dependencies |
| `make setup-kaggle` | Setup Kaggle API credentials |
| **Data** | |
| `make download-data` | Download NumtaDB dataset |
| `make verify-data` | Verify dataset integrity |
| **Training** | |
| `make train` | Train MobileNetV2 (default) |
| `make train-alexnet` | Train AlexNet model |
| `make train-resume CHECKPOINT=path` | Resume from checkpoint |
| **Evaluation** | |
| `make evaluate` | Evaluate trained model |
| `make visualize` | Generate training plots |
| `make show-metrics` | Display metrics summary |
| **Inference** | |
| `make predict IMAGE_PATH=path` | Predict single image |
| `make inference` | Detailed inference with probabilities |
| **Deployment** | |
| `make convert-onnx` | Convert model to ONNX |
| `make verify-onnx` | Verify ONNX model |
| **Utilities** | |
| `make info` | Show project information |
| `make clean` | Clean temporary files |
| `make clean-logs` | Clean log files |

### Makefile Variables

Override defaults by passing variables:
```bash
make evaluate CHECKPOINT=checkpoints/mobilenetv2/best_model_99_52.pth
make convert-onnx CHECKPOINT=path/to/checkpoint.pth MODEL_NAME=mobilenetv2
make predict IMAGE_PATH=my_digit.png
```

**Available Variables:**
- `CHECKPOINT` - Model checkpoint path (default: `checkpoints/best_model.pth`)
- `MODEL_NAME` - Architecture: `alexnet` or `mobilenetv2` (default: `mobilenetv2`)
- `ONNX_OUTPUT` - ONNX output path (default: `models/best_model.onnx`)
- `IMAGE_PATH` - Image for inference (default: `data/raw/testing-a/a00000.png`)

---

## Python Commands

Use Python directly for more control:

```bash
# Training
python train_model.py --model mobilenetv2 --epochs 50 --batch-size 64
python train_model.py --resume checkpoints/epoch_10.pth

# Evaluation
python evaluate_model.py --checkpoint checkpoints/best_model.pth --model mobilenetv2

# Inference
python -m trainer.inference --image path/to/image.png --detailed

# ONNX Conversion
python convert_to_onnx.py --checkpoint checkpoints/best_model.pth --model mobilenetv2
```

**Common Options:**
- Training: `--model`, `--epochs`, `--batch-size`, `--lr`, `--resume`, `--freeze-backbone`
- Evaluation: `--checkpoint`, `--model`, `--model-name`
- Inference: `--image`, `--checkpoint`, `--model-name`, `--detailed`
- ONNX: `--checkpoint`, `--output`, `--model`

---

## Workflows

### Complete Setup (First Time)
```bash
make install && make setup-kaggle && make download-data && make train
```

### Training Workflow
```bash
make train                  # Train model
make evaluate               # Evaluate results
make visualize              # View training plots
```

### Custom Checkpoint Workflow
```bash
make evaluate CHECKPOINT=checkpoints/mobilenetv2/best_model_99_52.pth
make convert-onnx CHECKPOINT=checkpoints/mobilenetv2/best_model_99_52.pth MODEL_NAME=mobilenetv2
make verify-onnx ONNX_OUTPUT=models/best_model.onnx
```

---

## Project Structure

```
numtadb-exp/
├── trainer/              # Training modules
│   ├── config.py        # Configuration
│   ├── model.py         # Model architectures
│   ├── dataset.py       # Data loading
│   ├── train.py         # Training logic
│   ├── evaluate.py      # Evaluation logic
│   └── inference.py     # Inference logic
├── notebooks/           # Jupyter notebooks
│   ├── NumtaDB_Colab_Training.ipynb  # Google Colab training
│   └── prepare_for_colab.sh          # Colab setup script
├── data/raw/            # Dataset directory
├── checkpoints/         # Saved model checkpoints
├── models/              # Converted ONNX models
├── logs/                # Training logs and metrics
├── Makefile             # Task automation
├── train_model.py       # Training entry point
├── evaluate_model.py    # Evaluation entry point
└── convert_to_onnx.py   # ONNX conversion script
```

---

## Configuration

Edit `trainer/config.py` for custom settings:

```python
# Model
MODEL_NAME = 'mobilenetv2'  # Options: 'alexnet', 'mobilenetv2'
NUM_CLASSES = 10
IMAGE_SIZE = 224

# Training
BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 0.001
OPTIMIZER = 'adam'          # Options: 'adam', 'adamw', 'sgd'

# Logging
LOG_LEVEL = logging.INFO
CONSOLE_LOGGING = True
FILE_LOGGING = True
```

---

## Kaggle Setup

**Option 1: Using script**
```bash
bash setup_kaggle.sh
```

**Option 2: Manual setup**
1. Download `kaggle.json` from https://www.kaggle.com/settings (API section)
2. Run:
```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| **Checkpoint not found** | Pass correct path: `make evaluate CHECKPOINT=path/to/checkpoint.pth` |
| **Make not found** | macOS: `xcode-select --install`<br>Ubuntu: `sudo apt-get install build-essential` |
| **Kaggle auth error** | Fix permissions: `chmod 600 ~/.kaggle/kaggle.json` |
| **Python version** | Edit Makefile: change `PYTHON := python3` to `PYTHON := python` |
| **CUDA out of memory** | Reduce batch size: `python train_model.py --batch-size 16` |

---

## Training on Google Colab

```bash
./notebooks/prepare_for_colab.sh
```

Then upload `notebooks/NumtaDB_Colab_Training.ipynb` to [Google Colab](https://colab.research.google.com/).

See [COLAB_GUIDE.md](COLAB_GUIDE.md) for complete instructions.

---

## Tips

💡 **Quick tips:**
- View all commands: `make help`
- Check project status: `make info`
- See training metrics: `make show-metrics`
- Chain commands: `make train evaluate visualize`
- Run in background: `make train &`

---

Made with ❤️ for Bengali language technology
