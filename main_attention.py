"""
Main Script for English-French Translation Model
Implements the complete pipeline from data preparation to evaluation
"""

import os
import argparse
import torch
from config import Config
from utils import set_seed, print_system_info, prepare_sample_data
from data_preparation import prepare_data
from model_attention import EncoderDecoderLSTM_Attn as EncoderDecoderLSTM, count_parameters
from training import train_model
from evaluation import evaluate_model



def main(args):
    """
    Main function to run the complete pipeline
    
    Args:
        args: Command line arguments
    """
    # Set random seed for reproducibility
    set_seed(Config.SEED)
    
    # Create necessary directories
    Config.create_dirs()
    
    # Print system information
    if args.verbose:
        print_system_info()
    
    # Step 1: Prepare data
    if args.prepare_data or args.mode == 'all':
        print("\n" + "=" * 60)
        print("STEP 1: DATA PREPARATION")
        print("=" * 60)
        
        # Check if data files exist
        data_files = [
            Config.TRAIN_EN, Config.TRAIN_FR,
            Config.VAL_EN, Config.VAL_FR,
            Config.TEST_EN, Config.TEST_FR
        ]
        
        if not all(os.path.exists(f) for f in data_files):
            print("Data files not found. Creating sample data...")
            prepare_sample_data()
        
        data_dict = prepare_data()
        
        # Save data dictionary for later use
        torch.save(data_dict, os.path.join(Config.DATA_DIR, 'data_dict.pth'))
        print("Data preparation complete!")
    
    # Step 2: Train model
    if args.train or args.mode == 'all':
        print("\n" + "=" * 60)
        print("STEP 2: MODEL TRAINING")
        print("=" * 60)
        
        # Load data dictionary
        if not args.prepare_data and args.mode != 'all':
            data_dict = torch.load(os.path.join(Config.DATA_DIR, 'data_dict.pth'), 
                                   weights_only=False)
        
        # Print model information
        model = EncoderDecoderLSTM(
            src_vocab_size=len(data_dict['src_vocab']),
            tgt_vocab_size=len(data_dict['tgt_vocab']),
            embedding_dim=Config.EMBEDDING_DIM,
            hidden_size=Config.HIDDEN_SIZE,
            num_layers=Config.NUM_LAYERS,
            dropout=Config.DROPOUT
        )
        
        print(f"\nModel Architecture:")
        print(f"  Total parameters: {count_parameters(model):,}")
        print(f"  Encoder parameters: {count_parameters(model.encoder):,}")
        print(f"  Decoder parameters: {count_parameters(model.decoder):,}")
        print()
        
        # Train model
        trainer = train_model(data_dict)
        print("Training complete!")
    
    # Step 3: Evaluate model
    if args.evaluate or args.mode == 'all':
        print("\n" + "=" * 60)
        print("STEP 3: MODEL EVALUATION")
        print("=" * 60)
        
        # Load data dictionary
        if not (args.prepare_data or args.train) and args.mode != 'all':
            data_dict = torch.load(os.path.join(Config.DATA_DIR, 'data_dict.pth'),
                                   weights_only=False)
        
        # Determine model path
        if args.model_path:
            model_path = args.model_path
        else:
            # Use best model if available, otherwise use final model
            best_model_path = os.path.join(Config.CHECKPOINT_DIR, 'best_model.pth')
            final_model_path = os.path.join(Config.CHECKPOINT_DIR, 'final_model.pth')
            
            if os.path.exists(best_model_path):
                model_path = best_model_path
            elif os.path.exists(final_model_path):
                model_path = final_model_path
            else:
                print("No trained model found. Please train a model first.")
                return
        
        print(f"Evaluating model: {model_path}")
        evaluator = evaluate_model(model_path, data_dict)
        print("Evaluation complete!")
    
    # Step 4: Interactive translation
    if args.interactive:
        print("\n" + "=" * 60)
        print("INTERACTIVE TRANSLATION MODE")
        print("=" * 60)
        print("Enter English sentences to translate (type 'quit' to exit)")
        print()
        
        # Load model and data
        if not (args.evaluate or args.mode == 'all'):
            data_dict = torch.load(os.path.join(Config.DATA_DIR, 'data_dict.pth'),
                                   weights_only=False)
            
            # Load model
            best_model_path = os.path.join(Config.CHECKPOINT_DIR, 'best_model.pth')
            if os.path.exists(best_model_path):
                checkpoint = torch.load(best_model_path, map_location=Config.DEVICE,
                                       weights_only=False)
                
                model = EncoderDecoderLSTM(
                    src_vocab_size=len(data_dict['src_vocab']),
                    tgt_vocab_size=len(data_dict['tgt_vocab']),
                    embedding_dim=Config.EMBEDDING_DIM,
                    hidden_size=Config.HIDDEN_SIZE,
                    num_layers=Config.NUM_LAYERS,
                    dropout=Config.DROPOUT
                )
                
                model.load_state_dict(checkpoint['model_state_dict'])
                model = model.to(Config.DEVICE)
                model.eval()
                
                from evaluation import Evaluator
                evaluator = Evaluator(
                    model=model,
                    test_loader=None,
                    src_vocab=data_dict['src_vocab'],
                    tgt_vocab=data_dict['tgt_vocab']
                )
            else:
                print("No trained model found. Please train a model first.")
                return
        
        # Interactive loop
        while True:
            try:
                sentence = input("\nEnglish: ").strip()
                
                if sentence.lower() == 'quit':
                    break
                
                if not sentence:
                    continue
                
                # Tokenize
                tokens = data_dict['en_tokenizer'].tokenize(sentence)
                
                # Translate
                translation_tokens = evaluator.translate_sentence(
                    tokens, 
                    use_beam_search=args.beam_search
                )
                
                translation = ' '.join(translation_tokens)
                print(f"French: {translation}")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")
        
        print("\nGoodbye!")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="English-French Translation Model with Encoder-Decoder LSTM"
    )
    
    # Mode selection
    parser.add_argument(
        '--mode',
        type=str,
        choices=['all', 'custom'],
        default='all',
        help='Run mode: all (complete pipeline) or custom (specific steps)'
    )
    
    # Individual step flags
    parser.add_argument(
        '--prepare-data',
        action='store_true',
        help='Run data preparation step'
    )
    
    parser.add_argument(
        '--train',
        action='store_true',
        help='Run training step'
    )
    
    parser.add_argument(
        '--evaluate',
        action='store_true',
        help='Run evaluation step'
    )
    
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Run interactive translation mode'
    )
    
    # Configuration overrides
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Override batch size'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Override number of epochs'
    )
    
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=None,
        help='Override learning rate'
    )
    
    parser.add_argument(
        '--hidden-size',
        type=int,
        default=None,
        help='Override hidden size'
    )
    
    parser.add_argument(
        '--embedding-dim',
        type=int,
        default=None,
        help='Override embedding dimension'
    )
    
    # Other options
    parser.add_argument(
        '--model-path',
        type=str,
        default=None,
        help='Path to model checkpoint for evaluation'
    )
    
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume training from'
    )
    
    parser.add_argument(
        '--beam-search',
        action='store_true',
        help='Use beam search for translation'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print verbose output'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    # Parse arguments
    args = parse_arguments()
    
    # Update configuration with command line arguments
    if args.batch_size:
        Config.BATCH_SIZE = args.batch_size
    if args.epochs:
        Config.NUM_EPOCHS = args.epochs
    if args.learning_rate:
        Config.LEARNING_RATE = args.learning_rate
    if args.hidden_size:
        Config.HIDDEN_SIZE = args.hidden_size
    if args.embedding_dim:
        Config.EMBEDDING_DIM = args.embedding_dim
    if args.beam_search:
        Config.USE_BEAM_SEARCH = True
    
    Config.SEED = args.seed
    
    # Run main function
    try:
        main(args)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        raise
