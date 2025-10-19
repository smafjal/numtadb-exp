#!/bin/bash

# Setup script for Kaggle API

echo "=========================================="
echo "Kaggle API Setup Script"
echo "=========================================="

# Check if kaggle directory exists
if [ ! -d "$HOME/.kaggle" ]; then
    echo "Creating ~/.kaggle directory..."
    mkdir -p ~/.kaggle
fi

# Check if kaggle.json exists
if [ ! -f "$HOME/.kaggle/kaggle.json" ]; then
    echo ""
    echo "⚠️  Kaggle API credentials not found!"
    echo ""
    echo "Please follow these steps:"
    echo "1. Go to https://www.kaggle.com/settings"
    echo "2. Scroll down to 'API' section"
    echo "3. Click 'Create New API Token'"
    echo "4. This will download 'kaggle.json'"
    echo "5. Move it to ~/.kaggle/kaggle.json"
    echo ""
    echo "Then run this command:"
    echo "  mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json"
    echo "  chmod 600 ~/.kaggle/kaggle.json"
    echo ""
    exit 1
else
    echo "✓ Found kaggle.json"
    
    # Set correct permissions
    chmod 600 ~/.kaggle/kaggle.json
    echo "✓ Set correct permissions (600)"
    
    echo ""
    echo "✓ Kaggle API is configured!"
    echo ""
    echo "You can now run: python download_dataset.py"
fi

