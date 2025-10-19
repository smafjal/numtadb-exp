"""
Script to download NumtaDB dataset from Kaggle
Author: Kaggle Dataset Downloader
Dataset: BengaliAI/numta
"""

import os
import zipfile
from pathlib import Path

def setup_directories():
    """Create necessary directories for the project"""
    directories = [
        'data/raw',
        'data/processed',
        'notebooks',
        'models',
        'src'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {directory}")

def download_kaggle_dataset():
    """Download the NumtaDB dataset from Kaggle using Kaggle API"""
    try:
        import kaggle
        
        print("\n📥 Downloading NumtaDB dataset from Kaggle...")
        print("Dataset: BengaliAI/numta")
        
        # Download the dataset
        kaggle.api.dataset_download_files(
            'BengaliAI/numta',
            path='data/raw',
            unzip=True
        )
        
        print("✓ Dataset downloaded successfully to 'data/raw' directory")
        
        # List downloaded files
        print("\n📂 Downloaded files:")
        for file in Path('data/raw').iterdir():
            if file.is_file():
                size_mb = file.stat().st_size / (1024 * 1024)
                print(f"  - {file.name} ({size_mb:.2f} MB)")
        
        return True
        
    except ImportError:
        print("❌ Error: 'kaggle' package not found.")
        print("\n📦 Please install it using:")
        print("   pip install kaggle")
        return False
        
    except Exception as e:
        print(f"❌ Error downloading dataset: {str(e)}")
        print("\n⚠️  Make sure you have:")
        print("   1. Kaggle API credentials configured")
        print("   2. Created ~/.kaggle/kaggle.json with your API token")
        print("   3. Set correct permissions: chmod 600 ~/.kaggle/kaggle.json")
        print("\n📖 Get your API token from: https://www.kaggle.com/settings")
        return False

def main():
    print("=" * 60)
    print("NumtaDB Dataset Downloader")
    print("=" * 60)
    
    # Setup directories
    print("\n1. Setting up project directories...")
    setup_directories()
    
    # Download dataset
    print("\n2. Downloading dataset from Kaggle...")
    success = download_kaggle_dataset()
    
    if success:
        print("\n" + "=" * 60)
        print("✓ Setup complete! Dataset is ready in 'data/raw' directory")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("⚠️  Please resolve the errors above and run again")
        print("=" * 60)

if __name__ == "__main__":
    main()

