"""
Configuration file for English-French Translation Model
Contains all hyperparameters and settings
"""

import torch
import os
import platform
class Config:
    """Main configuration class for the translation model"""
    
    # ============= Paths =============
    DATA_DIR = "./data"
    CHECKPOINT_DIR = "./checkpoints"
    LOG_DIR = "./logs"
    
    # Multi30K Dataset paths
    TRAIN_EN = os.path.join(DATA_DIR, "eng", "train.1.en")
    TRAIN_FR = os.path.join(DATA_DIR, "fr", "train.1.fr")
    VAL_EN = os.path.join(DATA_DIR, "eng", "val.1.en")
    VAL_FR = os.path.join(DATA_DIR, "fr", "val.1.fr")
    TEST_EN = os.path.join(DATA_DIR, "eng", "test_2016.1.en")
    TEST_FR = os.path.join(DATA_DIR, "fr", "test_2016.1.fr")

    # Extended dataset (optional)
    WMT14_EN = os.path.join(DATA_DIR, "wmt14_en.txt")
    WMT14_FR = os.path.join(DATA_DIR, "wmt14_fr.txt")
    
    # ============= Model Architecture =============
    EMBEDDING_DIM = 256      # Can be 256 or 512
    HIDDEN_SIZE = 512        # Hidden size for LSTM
    NUM_LAYERS = 2           # Number of LSTM layers
    DROPOUT = 0.3           # Dropout rate (0.3-0.5)
    MAX_VOCAB_SIZE = 10000  # Maximum vocabulary size per language
    
    # ============= Special Tokens =============
    PAD_TOKEN = '<pad>'
    SOS_TOKEN = '<sos>'
    EOS_TOKEN = '<eos>'
    UNK_TOKEN = '<unk>'
    
    PAD_IDX = 0
    SOS_IDX = 1
    EOS_IDX = 2
    UNK_IDX = 3
    
    # ============= Training Settings =============
    BATCH_SIZE = 64         # Batch size (32-128)
    NUM_EPOCHS = 20         # Number of training epochs
    LEARNING_RATE = 0.001   # Learning rate for Adam optimizer
    TEACHER_FORCING_RATIO = 0.5  # Teacher forcing ratio
    CLIP_GRAD_NORM = 1.0    # Gradient clipping value
    
    # ============= Early Stopping =============
    PATIENCE = 5            # Early stopping patience
    MIN_DELTA = 0.001       # Minimum change in validation loss
    
    # ============= Learning Rate Scheduler =============
    USE_SCHEDULER = True
    SCHEDULER_FACTOR = 0.5  # Factor by which LR will be reduced
    SCHEDULER_PATIENCE = 3  # Number of epochs with no improvement
    
    # ============= Data Processing =============
    MAX_LENGTH = 50         # Maximum sentence length
    MIN_LENGTH = 3          # Minimum sentence length
    
    # Set NUM_WORKERS based on platform
    if platform.system() == "Windows":
        NUM_WORKERS = 0  # Windows has issues with multiprocessing and spaCy
    else:
        NUM_WORKERS = 0  # Set to 0 for safety, or use 2-4 for Linux/Mac if needed
    
    # ============= Device Settings =============
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # ============= Logging & Checkpointing =============
    SAVE_BEST_ONLY = True   # Save only the best model
    CHECKPOINT_FREQ = 5     # Save checkpoint every N epochs
    LOG_FREQ = 100          # Log training stats every N batches
    
    # ============= Evaluation =============
    BEAM_SIZE = 3           # Beam size for beam search (optional)
    USE_BEAM_SEARCH = False # Whether to use beam search
    
    # ============= Reproducibility =============
    SEED = 42
    
    @classmethod
    def create_dirs(cls):
        """Create necessary directories if they don't exist"""
        os.makedirs(cls.DATA_DIR, exist_ok=True)
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.LOG_DIR, exist_ok=True)
    
    @classmethod
    def get_config_dict(cls):
        """Return configuration as dictionary for logging"""
        return {
            'embedding_dim': cls.EMBEDDING_DIM,
            'hidden_size': cls.HIDDEN_SIZE,
            'num_layers': cls.NUM_LAYERS,
            'dropout': cls.DROPOUT,
            'batch_size': cls.BATCH_SIZE,
            'learning_rate': cls.LEARNING_RATE,
            'teacher_forcing_ratio': cls.TEACHER_FORCING_RATIO,
            'max_length': cls.MAX_LENGTH,
            'max_vocab_size': cls.MAX_VOCAB_SIZE
        }