"""
Training Module
Handles model training, validation, and checkpointing
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import os
import time
from typing import Dict, Optional
from config import Config
from model_attention import EncoderDecoderLSTM_Attn


class Trainer:
    """Trainer class for managing the training process"""

    def __init__(self, model: EncoderDecoderLSTM_Attn, 
                 train_loader, val_loader,
                 src_vocab, tgt_vocab,
                 device: torch.device = None):
        """
        Initialize trainer
        
        Args:
            model: EncoderDecoderLSTM_Attn model
            train_loader: Training data loader
            val_loader: Validation data loader
            src_vocab: Source vocabulary
            tgt_vocab: Target vocabulary
            device: Device to use for training
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.device = device or Config.DEVICE
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Loss function (ignore padding index)
        self.criterion = nn.CrossEntropyLoss(ignore_index=Config.PAD_IDX)
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=Config.LEARNING_RATE
        )
        
        # Learning rate scheduler
        if Config.USE_SCHEDULER:
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=Config.SCHEDULER_FACTOR,
                patience=Config.SCHEDULER_PATIENCE,
                verbose=True
            )
        else:
            self.scheduler = None
        
        # Initialize tracking variables
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.epoch = 0
        
        # Tensorboard writer
        self.writer = SummaryWriter(Config.LOG_DIR)
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rates': []
        }
    
    def train_epoch(self, teacher_forcing_ratio: float = 0.5) -> float:
        """
        Train for one epoch
        
        Args:
            teacher_forcing_ratio: Teacher forcing ratio
            
        Returns:
            Average training loss
        """
        self.model.train()
        epoch_loss = 0
        
        with tqdm(self.train_loader, desc=f"Epoch {self.epoch+1} Training") as pbar:
            for batch_idx, batch in enumerate(pbar):
                # Move batch to device
                src = batch['src'].to(self.device)
                tgt = batch['tgt'].to(self.device)
                src_lens = batch['src_lens'].to(self.device)
                
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                output = self.model(src, src_lens, tgt, teacher_forcing_ratio)
                
                # Reshape for loss calculation
                # output: [batch_size, seq_len-1, vocab_size]
                # tgt: [batch_size, seq_len]
                output_dim = output.shape[-1]
                output = output.reshape(-1, output_dim)
                tgt = tgt[:, 1:].reshape(-1)  # Skip SOS token
                
                # Calculate loss
                loss = self.criterion(output, tgt)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    Config.CLIP_GRAD_NORM
                )
                
                # Update weights
                self.optimizer.step()
                
                # Track loss
                epoch_loss += loss.item()
                
                # Update progress bar
                pbar.set_postfix({'loss': loss.item()})
                
                # Log to tensorboard
                if batch_idx % Config.LOG_FREQ == 0:
                    global_step = self.epoch * len(self.train_loader) + batch_idx
                    self.writer.add_scalar('Train/BatchLoss', loss.item(), global_step)
        
        avg_loss = epoch_loss / len(self.train_loader)
        return avg_loss
    
    def validate(self) -> float:
        """
        Validate model
        
        Returns:
            Average validation loss
        """
        self.model.eval()
        epoch_loss = 0
        
        with torch.no_grad():
            with tqdm(self.val_loader, desc=f"Epoch {self.epoch+1} Validation") as pbar:
                for batch in pbar:
                    # Move batch to device
                    src = batch['src'].to(self.device)
                    tgt = batch['tgt'].to(self.device)
                    src_lens = batch['src_lens'].to(self.device)
                    
                    # Forward pass (no teacher forcing during validation)
                    output = self.model(src, src_lens, tgt, teacher_forcing_ratio=0)
                    
                    # Reshape for loss calculation
                    output_dim = output.shape[-1]
                    output = output.reshape(-1, output_dim)
                    tgt = tgt[:, 1:].reshape(-1)
                    
                    # Calculate loss
                    loss = self.criterion(output, tgt)
                    
                    # Track loss
                    epoch_loss += loss.item()
                    
                    # Update progress bar
                    pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = epoch_loss / len(self.val_loader)
        return avg_loss
    
    def save_checkpoint(self, filepath: str = None, is_best: bool = False):
        """
        Save model checkpoint
        
        Args:
            filepath: Path to save checkpoint
            is_best: Whether this is the best model so far
        """
        if filepath is None:
            filepath = os.path.join(Config.CHECKPOINT_DIR, f'checkpoint_epoch_{self.epoch}.pth')
        
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'history': self.history,
            'src_vocab': self.src_vocab,
            'tgt_vocab': self.tgt_vocab,
            'config': Config.get_config_dict()
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, filepath)
        
        if is_best:
            best_path = os.path.join(Config.CHECKPOINT_DIR, 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f"Saved best model to {best_path}")
    
    def load_checkpoint(self, filepath: str):
        """
        Load model checkpoint
        
        Args:
            filepath: Path to checkpoint file
        """
        checkpoint = torch.load(filepath, map_location=self.device,
                               weights_only=False)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.history = checkpoint['history']
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"Loaded checkpoint from epoch {self.epoch}")
    
    def train(self, num_epochs: int = None, resume_from: str = None):
        """
        Main training loop
        
        Args:
            num_epochs: Number of epochs to train
            resume_from: Path to checkpoint to resume from
        """
        num_epochs = num_epochs or Config.NUM_EPOCHS
        
        # Resume from checkpoint if specified
        if resume_from and os.path.exists(resume_from):
            self.load_checkpoint(resume_from)
            print(f"Resuming training from epoch {self.epoch + 1}")
        
        print(f"Training on device: {self.device}")
        print(f"Starting training for {num_epochs} epochs...")
        
        for epoch in range(self.epoch, num_epochs):
            self.epoch = epoch
            start_time = time.time()
            
            # Calculate teacher forcing ratio (decay over time)
            if epoch < num_epochs // 2:
                teacher_forcing_ratio = Config.TEACHER_FORCING_RATIO
            else:
                # Gradually reduce teacher forcing
                decay_factor = (epoch - num_epochs // 2) / (num_epochs // 2)
                teacher_forcing_ratio = Config.TEACHER_FORCING_RATIO * (1 - decay_factor * 0.5)
            
            # Train
            train_loss = self.train_epoch(teacher_forcing_ratio)
            
            # Validate
            val_loss = self.validate()
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['learning_rates'].append(current_lr)
            
            # Log to tensorboard
            self.writer.add_scalar('Train/EpochLoss', train_loss, epoch)
            self.writer.add_scalar('Val/EpochLoss', val_loss, epoch)
            self.writer.add_scalar('Train/LearningRate', current_lr, epoch)
            self.writer.add_scalar('Train/TeacherForcingRatio', teacher_forcing_ratio, epoch)
            
            # Print epoch summary
            epoch_time = time.time() - start_time
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Learning Rate: {current_lr:.6f}")
            print(f"  Teacher Forcing: {teacher_forcing_ratio:.2f}")
            print(f"  Time: {epoch_time:.2f}s")
            
            # Check if validation loss improved
            is_best = val_loss < self.best_val_loss
            
            if is_best:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                print(f"  ✓ New best validation loss!")
            else:
                self.patience_counter += 1
                print(f"  Patience: {self.patience_counter}/{Config.PATIENCE}")
            
            # Save checkpoint
            if (epoch + 1) % Config.CHECKPOINT_FREQ == 0 or is_best:
                self.save_checkpoint(is_best=is_best)
            
            # Learning rate scheduling
            if self.scheduler is not None:
                self.scheduler.step(val_loss)
            
            # Early stopping
            if self.patience_counter >= Config.PATIENCE:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
            
            print("-" * 50)
        
        # Save final model
        self.save_checkpoint(
            filepath=os.path.join(Config.CHECKPOINT_DIR, 'final_model.pth')
        )
        
        self.writer.close()
        print("\nTraining completed!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
    
    def plot_history(self):
        """Plot training history"""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot losses
        axes[0].plot(self.history['train_loss'], label='Train Loss')
        axes[0].plot(self.history['val_loss'], label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Plot learning rate
        axes[1].plot(self.history['learning_rates'])
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Learning Rate')
        axes[1].set_title('Learning Rate Schedule')
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(Config.LOG_DIR, 'training_history.png'))
        plt.show()


def train_model(data_dict: Dict):
    """
    Main training function
    
    Args:
        data_dict: Dictionary containing data loaders and vocabularies
    """
    # Create model
    model = EncoderDecoderLSTM_Attn(
        src_vocab_size=len(data_dict['src_vocab']),
        tgt_vocab_size=len(data_dict['tgt_vocab']),
        embedding_dim=Config.EMBEDDING_DIM,
        hidden_size=Config.HIDDEN_SIZE,
        num_layers=Config.NUM_LAYERS,
        dropout=Config.DROPOUT
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=data_dict['train_loader'],
        val_loader=data_dict['val_loader'],
        src_vocab=data_dict['src_vocab'],
        tgt_vocab=data_dict['tgt_vocab']
    )
    
    # Train model
    trainer.train()
    
    # Plot history
    trainer.plot_history()
    
    return trainer


if __name__ == "__main__":
    print("Training module loaded. Use main.py to run training.")
