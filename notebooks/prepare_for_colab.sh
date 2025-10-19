#!/bin/bash
# Prepare project files for Google Colab training
# This script creates a zip file with all necessary code

set -e  # Exit on error

echo "🚀 Preparing files for Google Colab..."
echo ""

# Check if we're in the right directory
if [ ! -d "trainer" ]; then
    echo "❌ Error: trainer directory not found!"
    echo "Please run this script from the project root directory."
    exit 1
fi

# Create a temporary directory
TEMP_DIR="colab_package"
rm -rf "$TEMP_DIR"
mkdir -p "$TEMP_DIR"

echo "📦 Packaging trainer module..."
# Copy trainer folder
cp -r trainer "$TEMP_DIR/"

echo "📦 Packaging training scripts..."
# Copy training scripts
cp train_model.py "$TEMP_DIR/"
cp evaluate_model.py "$TEMP_DIR/"
cp convert_to_onnx.py "$TEMP_DIR/" 2>/dev/null || true

echo "📦 Packaging requirements..."
# Copy requirements
cp requirements.txt "$TEMP_DIR/"

echo "📦 Creating README for Colab..."
# Create a simple README for the package
cat > "$TEMP_DIR/README.txt" << 'EOF'
NumtaDB Training Package for Google Colab
==========================================

This package contains all the code needed to train your model on Google Colab.

Files included:
- trainer/       : Core training modules
- train_model.py : Training script
- evaluate_model.py : Evaluation script
- requirements.txt : Python dependencies

Usage:
1. Upload this zip to Google Colab
2. Extract: !unzip trainer_package.zip
3. Follow the steps in the Colab notebook

For detailed instructions, see:
- NumtaDB_Colab_Training.ipynb
- COLAB_QUICK_START.md
EOF

# Create the zip file
echo "🗜️  Creating zip file..."
cd "$TEMP_DIR"
zip -r ../trainer_package.zip . > /dev/null
cd ..

# Clean up temp directory
rm -rf "$TEMP_DIR"

# Get file size
FILE_SIZE=$(du -h trainer_package.zip | cut -f1)

echo ""
echo "✅ Success! Package created: trainer_package.zip ($FILE_SIZE)"
echo ""
echo "📤 Next steps:"
echo "   1. Open Google Colab: https://colab.research.google.com/"
echo "   2. Upload 'trainer_package.zip' when prompted"
echo "   3. Also upload your 'kaggle.json' credentials"
echo "   4. Follow the notebook or guide to start training"
echo ""
echo "💡 Tip: Upload notebooks/NumtaDB_Colab_Training.ipynb to Colab"
echo "   for a complete guided experience!"
echo ""

