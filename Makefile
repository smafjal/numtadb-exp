.PHONY: help install setup-kaggle download-data verify-data train train-alexnet train-mobilenet train-resume evaluate evaluate-checkpoint visualize show-metrics inference predict convert-onnx convert-all verify-onnx clean clean-logs info

# Variables
PYTHON := python3
PIP := pip3
MODEL_NAME := mobilenetv2
CHECKPOINT := checkpoints/best_model.pth
ONNX_OUTPUT := models/best_model.onnx
IMAGE_PATH := data/raw/testing-a/a00000.png

# Colors for terminal output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[0;33m
BLUE := \033[0;34m
NC := \033[0m # No Color

##@ Help

help: ## Display this help message
	@echo "$(BLUE)NumtaDB Training Pipeline - Makefile Commands$(NC)"
	@echo ""
	@awk 'BEGIN {FS = ":.*##"; printf "Usage:\n  make $(CYAN)<target>$(NC)\n"} /^[a-zA-Z_0-9-]+:.*?##/ { printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2 } /^##@/ { printf "\n$(YELLOW)%s$(NC)\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

##@ Setup & Installation

install: ## Install Python dependencies
	@echo "$(BLUE)Installing dependencies...$(NC)"
	$(PIP) install -r requirements.txt
	@echo "$(GREEN)✓ Dependencies installed$(NC)"

setup-kaggle: ## Setup Kaggle API credentials
	@echo "$(BLUE)Setting up Kaggle API...$(NC)"
	@bash setup_kaggle.sh
	@echo "$(GREEN)✓ Kaggle setup complete$(NC)"

##@ Data Management

download-data: ## Download NumtaDB dataset from Kaggle
	@echo "$(BLUE)Downloading dataset...$(NC)"
	$(PYTHON) download_dataset.py
	@echo "$(GREEN)✓ Dataset downloaded$(NC)"

verify-data: ## Verify dataset integrity
	@echo "$(BLUE)Verifying dataset...$(NC)"
	@if [ -d "data/raw/training-a" ]; then \
		echo "$(GREEN)✓ Training data found$(NC)"; \
	else \
		echo "$(RED)✗ Training data not found$(NC)"; \
		exit 1; \
	fi
	@echo "$(GREEN)✓ Dataset verified$(NC)"

##@ Training

train: ## Train the model with default settings
	@echo "$(BLUE)Starting training...$(NC)"
	$(PYTHON) train_model.py
	@echo "$(GREEN)✓ Training complete$(NC)"

train-alexnet: ## Train AlexNet model
	@echo "$(BLUE)Training AlexNet model...$(NC)"
	$(PYTHON) train_model.py --model alexnet
	@echo "$(GREEN)✓ AlexNet training complete$(NC)"

train-mobilenet: ## Train MobileNetV2 model
	@echo "$(BLUE)Training MobileNetV2 model...$(NC)"
	$(PYTHON) train_model.py --model mobilenetv2
	@echo "$(GREEN)✓ MobileNetV2 training complete$(NC)"

train-resume: ## Resume training from checkpoint
	@echo "$(BLUE)Resuming training...$(NC)"
	$(PYTHON) train_model.py --resume $(CHECKPOINT)
	@echo "$(GREEN)✓ Training resumed$(NC)"

##@ Evaluation

evaluate: ## Evaluate the trained model
	@echo "$(BLUE)Evaluating model...$(NC)"
	@if [ "$(CHECKPOINT)" != "checkpoints/best_model.pth" ]; then \
		$(PYTHON) evaluate_model.py --checkpoint $(CHECKPOINT); \
	else \
		$(PYTHON) evaluate_model.py; \
	fi
	@echo "$(GREEN)✓ Evaluation complete$(NC)"

evaluate-checkpoint: ## Evaluate specific checkpoint
	@echo "$(BLUE)Evaluating checkpoint: $(CHECKPOINT)$(NC)"
	$(PYTHON) evaluate_model.py --checkpoint $(CHECKPOINT)
	@echo "$(GREEN)✓ Evaluation complete$(NC)"

##@ Visualization

visualize: ## Visualize training metrics
	@echo "$(BLUE)Generating visualizations...$(NC)"
	$(PYTHON) -m trainer.visualize
	@echo "$(GREEN)✓ Visualizations saved to logs/$(NC)"

show-metrics: ## Display training metrics summary
	@echo "$(BLUE)Training Metrics Summary:$(NC)"
	@if [ -f "logs/training_metrics.csv" ]; then \
		$(PYTHON) -c "import pandas as pd; df = pd.read_csv('logs/training_metrics.csv'); print(df.tail(10))"; \
	else \
		echo "$(RED)✗ No training metrics found$(NC)"; \
	fi

##@ Inference

inference: ## Run inference on a single image
	@echo "$(BLUE)Running inference on $(IMAGE_PATH)...$(NC)"
	$(PYTHON) -m trainer.inference --image $(IMAGE_PATH) --detailed
	@echo "$(GREEN)✓ Inference complete$(NC)"

predict: ## Quick prediction (specify IMAGE_PATH=path/to/image.png)
	@echo "$(BLUE)Predicting digit in $(IMAGE_PATH)...$(NC)"
	$(PYTHON) -m trainer.inference --image $(IMAGE_PATH)

##@ Model Conversion

convert-onnx: ## Convert PyTorch model to ONNX format
	@echo "$(BLUE)Converting model to ONNX...$(NC)"
	$(PYTHON) convert_to_onnx.py \
		--checkpoint $(CHECKPOINT) \
		--output $(ONNX_OUTPUT) \
		--model $(MODEL_NAME)
	@echo "$(GREEN)✓ Model converted to $(ONNX_OUTPUT)$(NC)"

convert-all: ## Convert all available checkpoints to ONNX
	@echo "$(BLUE)Converting all checkpoints to ONNX...$(NC)"
	@for checkpoint in checkpoints/*.pth; do \
		output="models/$$(basename $$checkpoint .pth).onnx"; \
		echo "Converting $$checkpoint to $$output..."; \
		$(PYTHON) convert_to_onnx.py --checkpoint $$checkpoint --output $$output; \
	done
	@echo "$(GREEN)✓ All models converted$(NC)"

verify-onnx: ## Verify ONNX model
	@echo "$(BLUE)Verifying ONNX model...$(NC)"
	@$(PYTHON) -c "import onnx; model = onnx.load('$(ONNX_OUTPUT)'); onnx.checker.check_model(model); print('✓ ONNX model is valid')"
	@echo "$(GREEN)✓ ONNX model verified$(NC)"

##@ Cleaning

clean: ## Clean temporary files and cache
	@echo "$(BLUE)Cleaning temporary files...$(NC)"
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name ".DS_Store" -delete
	@echo "$(GREEN)✓ Cleaned temporary files$(NC)"

clean-logs: ## Clean log files
	@echo "$(YELLOW)Cleaning log files...$(NC)"
	rm -rf logs/*.log
	@echo "$(GREEN)✓ Log files cleaned$(NC)"


##@ Information

info: ## Display project information
	@echo "$(BLUE)═══════════════════════════════════════════════$(NC)"
	@echo "$(BLUE)  NumtaDB Bengali Digit Classification Project$(NC)"
	@echo "$(BLUE)═══════════════════════════════════════════════$(NC)"
	@echo ""
	@echo "$(YELLOW)Python Version:$(NC)"
	@$(PYTHON) --version
	@echo ""
	@echo "$(YELLOW)Project Structure:$(NC)"
	@echo "  📁 data/          - Training and testing data"
	@echo "  📁 trainer/       - Training modules"
	@echo "  📁 checkpoints/   - Saved model checkpoints"
	@echo "  📁 models/        - Converted ONNX models"
	@echo "  📁 logs/          - Training logs and metrics"
	@echo "  📁 frontend/      - Web interface"
	@echo ""
	@echo "$(YELLOW)Available Models:$(NC)"
	@if [ -d "checkpoints" ]; then ls -lh checkpoints/*.pth 2>/dev/null || echo "  No checkpoints found"; fi
	@echo ""
	@echo "$(YELLOW)Available ONNX Models:$(NC)"
	@if [ -d "models" ]; then ls -lh models/*.onnx 2>/dev/null || echo "  No ONNX models found"; fi
	@echo ""
	@echo "$(YELLOW)Dataset Status:$(NC)"
	@if [ -d "data/raw/training-a" ]; then \
		echo "  $(GREEN)✓$(NC) Training data present"; \
	else \
		echo "  $(RED)✗$(NC) Training data missing"; \
	fi
	@if [ -d "data/raw/testing-a" ]; then \
		echo "  $(GREEN)✓$(NC) Testing data present"; \
	else \
		echo "  $(RED)✗$(NC) Testing data missing"; \
	fi

