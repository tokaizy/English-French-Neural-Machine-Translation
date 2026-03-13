"""
Utility Module
Handles dataset downloading and other utility functions
"""

import os
import urllib.request
import zipfile
import tarfile
from typing import Optional
from config import Config


def download_file(url: str, filepath: str):
    """
    Download file from URL
    
    Args:
        url: URL to download from
        filepath: Path to save file
    """
    print(f"Downloading {url}...")
    urllib.request.urlretrieve(url, filepath)
    print(f"Downloaded to {filepath}")


def extract_archive(filepath: str, extract_to: str):
    """
    Extract archive file
    
    Args:
        filepath: Path to archive file
        extract_to: Directory to extract to
    """
    print(f"Extracting {filepath}...")
    
    if filepath.endswith('.zip'):
        with zipfile.ZipFile(filepath, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
    elif filepath.endswith('.tar.gz') or filepath.endswith('.tgz'):
        with tarfile.open(filepath, 'r:gz') as tar_ref:
            tar_ref.extractall(extract_to)
    elif filepath.endswith('.tar'):
        with tarfile.open(filepath, 'r') as tar_ref:
            tar_ref.extractall(extract_to)
    
    print(f"Extracted to {extract_to}")


def download_multi30k():
    """
    Download Multi30K dataset
    
    Note: The exact URLs may change. These are example URLs.
    You should replace them with the actual Multi30K download links.
    """
    base_url = "https://github.com/multi30k/dataset/raw/master/data/task1/raw/"
    
    files = {
        'train.en': 'train.en.gz',
        'train.fr': 'train.fr.gz',
        'val.en': 'val.en.gz',
        'val.fr': 'val.fr.gz',
        'test.en': 'test_2016_flickr.en.gz',
        'test.fr': 'test_2016_flickr.fr.gz'
    }
    
    print("Downloading Multi30K dataset...")
    os.makedirs(Config.DATA_DIR, exist_ok=True)
    
    for local_name, remote_name in files.items():
        local_path = os.path.join(Config.DATA_DIR, local_name)
        
        if os.path.exists(local_path):
            print(f"{local_name} already exists, skipping...")
            continue
        
        # Download compressed file
        compressed_path = local_path + '.gz'
        if not os.path.exists(compressed_path):
            try:
                url = base_url + remote_name
                download_file(url, compressed_path)
            except Exception as e:
                print(f"Error downloading {remote_name}: {e}")
                print("Please download the Multi30K dataset manually")
                return False
        
        # Decompress
        import gzip
        with gzip.open(compressed_path, 'rt', encoding='utf-8') as f_in:
            with open(local_path, 'w', encoding='utf-8') as f_out:
                f_out.write(f_in.read())
        
        print(f"Decompressed {local_name}")
    
    print("Multi30K dataset downloaded successfully!")
    return True


def prepare_sample_data():
    """
    Create sample data files for testing
    This creates small sample files with the same format as Multi30K
    """
    print("Creating sample data files for testing...")
    
    os.makedirs(Config.DATA_DIR, exist_ok=True)
    
    # Sample English sentences
    en_sentences = [
        "Two men are looking at something in the garden",
        "The men are working on the cable car",
        "A girl in a pink dress climbs into a stall",
        "A man on a ladder cleans a window",
        "A man at the stove fills another man's plate",
        "A makeup artist works on a guitarist's costume",
        "A young man holds a large plush toy",
        "A woman in a blue shirt talks on the phone while roller skating",
        "Two women and a man walk on the sidewalk",
        "Two shirtless men jumping over a railing"
    ]
    
    # Sample French sentences (translations)
    fr_sentences = [
        "Deux hommes regardent quelque chose dans le jardin",
        "Les hommes travaillent sur le téléphérique",
        "Une fille en robe rose grimpe dans une stalle",
        "Un homme sur une échelle nettoie une fenêtre",
        "Un homme au fourneau remplit l'assiette d'un autre homme",
        "Un maquilleur travaille sur le costume d'un guitariste",
        "Un jeune homme tient une grande peluche",
        "Une femme en chemise bleue parle au téléphone en faisant du roller",
        "Deux femmes et un homme marchent sur le trottoir",
        "Deux hommes torse nu sautant par-dessus une balustrade"
    ]
    
    # Create training data (duplicate to have more samples)
    train_en = en_sentences * 100  # 1000 samples
    train_fr = fr_sentences * 100
    
    # Create validation data
    val_en = en_sentences * 10  # 100 samples
    val_fr = fr_sentences * 10
    
    # Create test data
    test_en = en_sentences * 10  # 100 samples
    test_fr = fr_sentences * 10
    
    # Write files
    files_to_write = [
        (Config.TRAIN_EN, train_en),
        (Config.TRAIN_FR, train_fr),
        (Config.VAL_EN, val_en),
        (Config.VAL_FR, val_fr),
        (Config.TEST_EN, test_en),
        (Config.TEST_FR, test_fr)
    ]
    
    for filepath, sentences in files_to_write:
        with open(filepath, 'w', encoding='utf-8') as f:
            for sentence in sentences:
                f.write(sentence + '\n')
        print(f"Created {filepath}")
    
    print("Sample data files created successfully!")


def install_requirements():
    """
    Install required packages
    """
    requirements = [
        "torch",
        "numpy",
        "nltk",
        "spacy",
        "tqdm",
        "matplotlib",
        "tensorboard"
    ]
    
    print("Installing required packages...")
    import subprocess
    
    for package in requirements:
        try:
            subprocess.check_call(["pip", "install", package])
        except:
            print(f"Failed to install {package}, please install manually")
    
    # Download spaCy models
    print("Downloading spaCy language models...")
    try:
        subprocess.check_call(["python", "-m", "spacy", "download", "en_core_web_sm"])
        subprocess.check_call(["python", "-m", "spacy", "download", "fr_core_news_sm"])
    except:
        print("Failed to download spaCy models, please run:")
        print("  python -m spacy download en_core_web_sm")
        print("  python -m spacy download fr_core_news_sm")
    
    # Download NLTK data for BLEU score
    print("Downloading NLTK data...")
    try:
        import nltk
        nltk.download('punkt')
    except:
        print("Failed to download NLTK data")
    
    print("Requirements installed successfully!")


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility
    
    Args:
        seed: Random seed
    """
    import random
    import numpy as np
    import torch
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def print_system_info():
    """Print system and PyTorch information"""
    import torch
    import platform
    
    print("=" * 60)
    print("SYSTEM INFORMATION")
    print("=" * 60)
    print(f"Python version: {platform.python_version()}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"    Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
    
    print(f"Device: {Config.DEVICE}")
    print("=" * 60)
    print()


if __name__ == "__main__":
    # Create directories
    Config.create_dirs()
    
    # Install requirements
    # install_requirements()
    
    # Create sample data for testing
    prepare_sample_data()
    
    print("\nUtility setup complete!")
