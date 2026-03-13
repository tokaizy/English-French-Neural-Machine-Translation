"""
Data Preparation Module
Handles tokenization, vocabulary building, and data loading
"""

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import spacy
from typing import List, Tuple, Dict
import os
from config import Config

class Tokenizer:
    """Tokenizer class for both English and French"""
    
    def __init__(self, lang: str):
        """
        Initialize tokenizer with spaCy model
        
        Args:
            lang: Language code ('en' or 'fr')
        """
        self.lang = lang
        self.nlp = None
        self._load_model()
    
    def _load_model(self):
        """Load spaCy model"""
        if self.nlp is None:
            try:
                if self.lang == 'en':
                    self.nlp = spacy.load('en_core_web_sm')
                elif self.lang == 'fr':
                    self.nlp = spacy.load('fr_core_news_sm')
                else:
                    raise ValueError(f"Unsupported language: {self.lang}")
            except:
                print(f"Please install spacy model for {self.lang}")
                print(f"Run: python -m spacy download {'en_core_web_sm' if self.lang == 'en' else 'fr_core_news_sm'}")
                raise
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words
        
        Args:
            text: Input text string
            
        Returns:
            List of tokens
        """
        if self.nlp is None:
            self._load_model()
        return [token.text.lower() for token in self.nlp(text.strip())]
    
    def __getstate__(self):
        """
        Custom pickle support - exclude nlp object
        This is needed for Windows multiprocessing compatibility
        """
        state = self.__dict__.copy()
        # Don't pickle the spaCy nlp object
        state['nlp'] = None
        return state
    
    def __setstate__(self, state):
        """
        Custom unpickle support - reload nlp object
        """
        self.__dict__.update(state)
        # Don't reload immediately - will be loaded on first use
        self.nlp = None


class Vocabulary:
    """Vocabulary class for mapping tokens to indices"""
    
    def __init__(self, max_size: int = 10000):
        """
        Initialize vocabulary
        
        Args:
            max_size: Maximum vocabulary size
        """
        self.max_size = max_size
        self.token2idx = {
            Config.PAD_TOKEN: Config.PAD_IDX,
            Config.SOS_TOKEN: Config.SOS_IDX,
            Config.EOS_TOKEN: Config.EOS_IDX,
            Config.UNK_TOKEN: Config.UNK_IDX
        }
        self.idx2token = {idx: token for token, idx in self.token2idx.items()}
        self.token_freq = Counter()
        
    def build_vocab_from_iterator(self, token_lists: List[List[str]]):
        """
        Build vocabulary from iterator of token lists
        
        Args:
            token_lists: List of tokenized sentences
        """
        # Count token frequencies
        for tokens in token_lists:
            self.token_freq.update(tokens)
        
        # Get most common tokens
        most_common = self.token_freq.most_common(self.max_size - 4)  # Reserve 4 for special tokens
        
        # Add to vocabulary
        for token, freq in most_common:
            idx = len(self.token2idx)
            self.token2idx[token] = idx
            self.idx2token[idx] = token
    
    def __len__(self):
        return len(self.token2idx)
    
    def token_to_idx(self, token: str) -> int:
        """Convert token to index"""
        return self.token2idx.get(token, Config.UNK_IDX)
    
    def idx_to_token(self, idx: int) -> str:
        """Convert index to token"""
        return self.idx2token.get(idx, Config.UNK_TOKEN)
    
    def encode(self, tokens: List[str]) -> List[int]:
        """Encode list of tokens to indices"""
        return [self.token_to_idx(token) for token in tokens]
    
    def decode(self, indices: List[int]) -> List[str]:
        """Decode list of indices to tokens"""
        return [self.idx_to_token(idx) for idx in indices]


class TranslationDataset(Dataset):
    """Custom Dataset for translation task"""
    
    def __init__(self, src_file: str, tgt_file: str, 
                 src_tokenizer: Tokenizer, tgt_tokenizer: Tokenizer,
                 src_vocab: Vocabulary = None, tgt_vocab: Vocabulary = None,
                 max_length: int = 50):
        """
        Initialize dataset
        
        Args:
            src_file: Path to source language file
            tgt_file: Path to target language file
            src_tokenizer: Source language tokenizer
            tgt_tokenizer: Target language tokenizer
            src_vocab: Source vocabulary (will be built if None)
            tgt_vocab: Target vocabulary (will be built if None)
            max_length: Maximum sentence length
        """
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.max_length = max_length
        
        # Read files
        self.src_sentences = self._read_file(src_file)
        self.tgt_sentences = self._read_file(tgt_file)
        
        assert len(self.src_sentences) == len(self.tgt_sentences), \
            "Source and target files must have the same number of lines"
        
        # Tokenize all sentences
        print(f"Tokenizing {len(self.src_sentences)} sentence pairs...")
        self.src_tokens = [src_tokenizer.tokenize(sent) for sent in self.src_sentences]
        self.tgt_tokens = [tgt_tokenizer.tokenize(sent) for sent in self.tgt_sentences]
        
        # Filter by length
        self._filter_by_length()
        
        # Build or use provided vocabularies
        if src_vocab is None:
            print("Building source vocabulary...")
            self.src_vocab = Vocabulary(Config.MAX_VOCAB_SIZE)
            self.src_vocab.build_vocab_from_iterator(self.src_tokens)
        else:
            self.src_vocab = src_vocab
            
        if tgt_vocab is None:
            print("Building target vocabulary...")
            self.tgt_vocab = Vocabulary(Config.MAX_VOCAB_SIZE)
            self.tgt_vocab.build_vocab_from_iterator(self.tgt_tokens)
        else:
            self.tgt_vocab = tgt_vocab
        
        print(f"Source vocab size: {len(self.src_vocab)}")
        print(f"Target vocab size: {len(self.tgt_vocab)}")
        print(f"Dataset size: {len(self)}")
    
    def _read_file(self, filepath: str) -> List[str]:
        """Read sentences from file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]
    
    def _filter_by_length(self):
        """Filter sentences by length"""
        filtered_src = []
        filtered_tgt = []
        
        for src, tgt in zip(self.src_tokens, self.tgt_tokens):
            if (Config.MIN_LENGTH <= len(src) <= self.max_length and 
                Config.MIN_LENGTH <= len(tgt) <= self.max_length):
                filtered_src.append(src)
                filtered_tgt.append(tgt)
        
        self.src_tokens = filtered_src
        self.tgt_tokens = filtered_tgt
        print(f"Filtered dataset size: {len(self.src_tokens)}")
    
    def __len__(self):
        return len(self.src_tokens)
    
    def __getitem__(self, idx):
        """Get a single sample"""
        src_tokens = self.src_tokens[idx]
        tgt_tokens = self.tgt_tokens[idx]
        
        # Add SOS and EOS tokens
        src_indices = self.src_vocab.encode(src_tokens)
        tgt_indices = [Config.SOS_IDX] + self.tgt_vocab.encode(tgt_tokens) + [Config.EOS_IDX]
        
        return {
            'src': torch.tensor(src_indices, dtype=torch.long),
            'tgt': torch.tensor(tgt_indices, dtype=torch.long),
            'src_len': len(src_indices),
            'tgt_len': len(tgt_indices)
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for padding and packing
    
    Args:
        batch: List of samples from dataset
        
    Returns:
        Dictionary with padded tensors
    """
    # Sort batch by source length (descending) for pack_padded_sequence
    batch.sort(key=lambda x: x['src_len'], reverse=True)
    
    # Extract sequences
    src_seqs = [item['src'] for item in batch]
    tgt_seqs = [item['tgt'] for item in batch]
    src_lens = torch.tensor([item['src_len'] for item in batch])
    tgt_lens = torch.tensor([item['tgt_len'] for item in batch])
    
    # Pad sequences
    src_padded = pad_sequence(src_seqs, batch_first=True, padding_value=Config.PAD_IDX)
    tgt_padded = pad_sequence(tgt_seqs, batch_first=True, padding_value=Config.PAD_IDX)
    
    return {
        'src': src_padded,
        'tgt': tgt_padded,
        'src_lens': src_lens,
        'tgt_lens': tgt_lens
    }


def create_data_loaders(train_dataset: TranslationDataset,
                        val_dataset: TranslationDataset,
                        test_dataset: TranslationDataset,
                        batch_size: int = 64) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create DataLoaders for train, validation, and test sets
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset
        batch_size: Batch size
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Force num_workers=0 on Windows to avoid pickle errors with spaCy
    import platform
    num_workers = 0 if platform.system() == "Windows" else Config.NUM_WORKERS
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True if Config.DEVICE.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True if Config.DEVICE.type == 'cuda' else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True if Config.DEVICE.type == 'cuda' else False
    )
    
    return train_loader, val_loader, test_loader


def prepare_data():
    """
    Main function to prepare all data
    
    Returns:
        Dictionary containing tokenizers, vocabularies, and data loaders
    """
    print("Initializing tokenizers...")
    en_tokenizer = Tokenizer('en')
    fr_tokenizer = Tokenizer('fr')
    
    print("Creating training dataset...")
    train_dataset = TranslationDataset(
        Config.TRAIN_EN, Config.TRAIN_FR,
        en_tokenizer, fr_tokenizer,
        max_length=Config.MAX_LENGTH
    )
    
    # Use vocabularies from training set for validation and test
    print("Creating validation dataset...")
    val_dataset = TranslationDataset(
        Config.VAL_EN, Config.VAL_FR,
        en_tokenizer, fr_tokenizer,
        src_vocab=train_dataset.src_vocab,
        tgt_vocab=train_dataset.tgt_vocab,
        max_length=Config.MAX_LENGTH
    )
    
    print("Creating test dataset...")
    test_dataset = TranslationDataset(
        Config.TEST_EN, Config.TEST_FR,
        en_tokenizer, fr_tokenizer,
        src_vocab=train_dataset.src_vocab,
        tgt_vocab=train_dataset.tgt_vocab,
        max_length=Config.MAX_LENGTH
    )
    
    print("Creating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(
        train_dataset, val_dataset, test_dataset,
        batch_size=Config.BATCH_SIZE
    )
    
    return {
        'en_tokenizer': en_tokenizer,
        'fr_tokenizer': fr_tokenizer,
        'src_vocab': train_dataset.src_vocab,
        'tgt_vocab': train_dataset.tgt_vocab,
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'train_dataset': train_dataset,
        'val_dataset': val_dataset,
        'test_dataset': test_dataset
    }


if __name__ == "__main__":
    # Test data preparation
    Config.create_dirs()
    data = prepare_data()
    print(f"\nData preparation complete!")
    print(f"Train batches: {len(data['train_loader'])}")
    print(f"Val batches: {len(data['val_loader'])}")
    print(f"Test batches: {len(data['test_loader'])}")
