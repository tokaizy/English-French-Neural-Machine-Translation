"""
Evaluation Module
Handles model evaluation, BLEU score calculation, and translation examples
"""

import torch
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from typing import List, Dict, Tuple
from tqdm import tqdm
import random
from config import Config
from model_attention import EncoderDecoderLSTM_Attn


class Evaluator:
    """Evaluator class for model evaluation"""

    def __init__(self, model: EncoderDecoderLSTM_Attn, 
                 test_loader, src_vocab, tgt_vocab,
                 device: torch.device = None):
        """
        Initialize evaluator
        
        Args:
            model: Trained model
            test_loader: Test data loader
            src_vocab: Source vocabulary
            tgt_vocab: Target vocabulary
            device: Device to use
        """
        self.model = model
        self.test_loader = test_loader
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.device = device or Config.DEVICE
        
        # Move model to device
        self.model = self.model.to(self.device)
        self.model.eval()
    
    def translate_sentence(self, src_tokens: List[str], 
                          use_beam_search: bool = False,
                          beam_size: int = 3) -> List[str]:
        """
        Translate a sentence from source to target language
        
        Args:
            src_tokens: List of source tokens
            use_beam_search: Whether to use beam search
            beam_size: Size of beam for beam search
            
        Returns:
            List of target tokens
        """
        # Convert tokens to indices
        src_indices = self.src_vocab.encode(src_tokens)
        src_tensor = torch.tensor(src_indices, dtype=torch.long).to(self.device)
        src_len = torch.tensor([len(src_indices)], dtype=torch.long).to(self.device)
        
        # Translate
        if use_beam_search and Config.USE_BEAM_SEARCH:
            tgt_indices = self.model.translate_beam_search(
                src_tensor, src_len, beam_size=beam_size
            )
        else:
            tgt_indices = self.model.translate(src_tensor, src_len)
        
        # Convert indices to tokens
        tgt_tokens = self.tgt_vocab.decode(tgt_indices)
        
        # Remove special tokens
        tgt_tokens = [token for token in tgt_tokens 
                     if token not in [Config.SOS_TOKEN, Config.EOS_TOKEN, Config.PAD_TOKEN]]
        
        return tgt_tokens
    
    def calculate_bleu_score(self, num_samples: int = None) -> Dict[str, float]:
        """
        Calculate BLEU scores on test set
        
        Args:
            num_samples: Number of samples to evaluate (None for all)
            
        Returns:
            Dictionary with BLEU scores
        """
        print("Calculating BLEU scores...")
        
        references = []
        hypotheses = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.test_loader, desc="Evaluating")):
                # Move batch to device
                src = batch['src'].to(self.device)
                tgt = batch['tgt'].to(self.device)
                src_lens = batch['src_lens'].to(self.device)
                
                batch_size = src.size(0)
                
                for i in range(batch_size):
                    # Get source and reference
                    src_seq = src[i]
                    tgt_seq = tgt[i]
                    src_len = src_lens[i].unsqueeze(0)
                    
                    # Remove padding from source
                    src_seq = src_seq[:src_len.item()]
                    
                    # Translate
                    pred_indices = self.model.translate(
                        src_seq.unsqueeze(0), src_len,
                        max_length=Config.MAX_LENGTH
                    )
                    
                    # Convert to tokens
                    pred_tokens = self.tgt_vocab.decode(pred_indices)
                    
                    # Get reference tokens (excluding special tokens)
                    ref_indices = tgt_seq.cpu().numpy().tolist()
                    ref_tokens = self.tgt_vocab.decode(ref_indices)
                    
                    # Clean tokens (remove special tokens)
                    pred_tokens = [t for t in pred_tokens 
                                 if t not in [Config.SOS_TOKEN, Config.EOS_TOKEN, Config.PAD_TOKEN]]
                    ref_tokens = [t for t in ref_tokens 
                                if t not in [Config.SOS_TOKEN, Config.EOS_TOKEN, Config.PAD_TOKEN]]
                    
                    if len(pred_tokens) > 0 and len(ref_tokens) > 0:
                        references.append([ref_tokens])  # Note: reference should be a list of references
                        hypotheses.append(pred_tokens)
                    
                    # Limit number of samples if specified
                    if num_samples and len(references) >= num_samples:
                        break
                
                if num_samples and len(references) >= num_samples:
                    break
        
        # Calculate BLEU scores
        bleu_1 = corpus_bleu(references, hypotheses, weights=(1, 0, 0, 0))
        bleu_2 = corpus_bleu(references, hypotheses, weights=(0.5, 0.5, 0, 0))
        bleu_3 = corpus_bleu(references, hypotheses, weights=(0.33, 0.33, 0.34, 0))
        bleu_4 = corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25))
        
        scores = {
            'BLEU-1': bleu_1 * 100,
            'BLEU-2': bleu_2 * 100,
            'BLEU-3': bleu_3 * 100,
            'BLEU-4': bleu_4 * 100,
            'num_samples': len(references)
        }
        
        return scores
    
    def get_translation_examples(self, num_examples: int = 5, 
                                random_sample: bool = True) -> List[Dict]:
        """
        Get translation examples
        
        Args:
            num_examples: Number of examples to generate
            random_sample: Whether to randomly sample examples
            
        Returns:
            List of translation examples
        """
        examples = []
        dataset = self.test_loader.dataset
        
        # Select indices
        if random_sample:
            indices = random.sample(range(len(dataset)), min(num_examples, len(dataset)))
        else:
            indices = list(range(min(num_examples, len(dataset))))
        
        for idx in indices:
            # Get sample
            sample = dataset[idx]
            src_indices = sample['src'].cpu().numpy().tolist()
            tgt_indices = sample['tgt'].cpu().numpy().tolist()
            
            # Convert to tokens
            src_tokens = self.src_vocab.decode(src_indices)
            ref_tokens = self.tgt_vocab.decode(tgt_indices)
            
            # Clean reference tokens
            ref_tokens = [t for t in ref_tokens 
                        if t not in [Config.SOS_TOKEN, Config.EOS_TOKEN, Config.PAD_TOKEN]]
            
            # Translate
            pred_tokens = self.translate_sentence(src_tokens)
            
            # Calculate sentence BLEU
            if len(pred_tokens) > 0 and len(ref_tokens) > 0:
                sentence_bleu_score = sentence_bleu(
                    [ref_tokens], pred_tokens,
                    weights=(0.25, 0.25, 0.25, 0.25)
                ) * 100
            else:
                sentence_bleu_score = 0.0
            
            example = {
                'source': ' '.join(src_tokens),
                'reference': ' '.join(ref_tokens),
                'prediction': ' '.join(pred_tokens),
                'bleu_score': sentence_bleu_score
            }
            
            examples.append(example)
        
        return examples
    
    def analyze_errors(self, num_samples: int = 100) -> Dict:
        """
        Analyze common translation errors
        
        Args:
            num_samples: Number of samples to analyze
            
        Returns:
            Dictionary with error analysis
        """
        print("Analyzing translation errors...")
        
        error_types = {
            'empty_translation': 0,
            'too_short': 0,  # Less than 50% of reference length
            'too_long': 0,   # More than 150% of reference length
            'repeated_words': 0,
            'unk_tokens': 0,
            'perfect_match': 0,
            'total': 0
        }
        
        length_ratios = []
        bleu_scores = []
        
        dataset = self.test_loader.dataset
        indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
        
        for idx in tqdm(indices, desc="Analyzing errors"):
            sample = dataset[idx]
            src_indices = sample['src'].cpu().numpy().tolist()
            tgt_indices = sample['tgt'].cpu().numpy().tolist()
            
            # Convert to tokens
            src_tokens = self.src_vocab.decode(src_indices)
            ref_tokens = self.tgt_vocab.decode(tgt_indices)
            
            # Clean reference tokens
            ref_tokens = [t for t in ref_tokens 
                        if t not in [Config.SOS_TOKEN, Config.EOS_TOKEN, Config.PAD_TOKEN]]
            
            # Translate
            pred_tokens = self.translate_sentence(src_tokens)
            
            error_types['total'] += 1
            
            # Check for errors
            if len(pred_tokens) == 0:
                error_types['empty_translation'] += 1
                continue
            
            # Length ratio
            length_ratio = len(pred_tokens) / max(len(ref_tokens), 1)
            length_ratios.append(length_ratio)
            
            if length_ratio < 0.5:
                error_types['too_short'] += 1
            elif length_ratio > 1.5:
                error_types['too_long'] += 1
            
            # Check for repeated words
            if len(pred_tokens) > 3:
                for i in range(len(pred_tokens) - 2):
                    if pred_tokens[i] == pred_tokens[i+1] == pred_tokens[i+2]:
                        error_types['repeated_words'] += 1
                        break
            
            # Check for UNK tokens
            if Config.UNK_TOKEN in pred_tokens:
                error_types['unk_tokens'] += 1
            
            # Check for perfect match
            if pred_tokens == ref_tokens:
                error_types['perfect_match'] += 1
            
            # Calculate BLEU
            if len(pred_tokens) > 0 and len(ref_tokens) > 0:
                bleu = sentence_bleu(
                    [ref_tokens], pred_tokens,
                    weights=(0.25, 0.25, 0.25, 0.25)
                ) * 100
                bleu_scores.append(bleu)
        
        # Calculate statistics
        analysis = {
            'error_counts': error_types,
            'error_percentages': {
                k: (v / error_types['total']) * 100 
                for k, v in error_types.items() if k != 'total'
            },
            'avg_length_ratio': np.mean(length_ratios) if length_ratios else 0,
            'std_length_ratio': np.std(length_ratios) if length_ratios else 0,
            'avg_bleu': np.mean(bleu_scores) if bleu_scores else 0,
            'std_bleu': np.std(bleu_scores) if bleu_scores else 0
        }
        
        return analysis
    
    def generate_report(self, output_file: str = None) -> str:
        """
        Generate comprehensive evaluation report
        
        Args:
            output_file: Path to save report (optional)
            
        Returns:
            Report as string
        """
        report = []
        report.append("=" * 60)
        report.append("TRANSLATION MODEL EVALUATION REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Model configuration
        report.append("Model Configuration:")
        report.append("-" * 30)
        for key, value in Config.get_config_dict().items():
            report.append(f"  {key}: {value}")
        report.append("")
        
        # BLEU scores
        report.append("BLEU Scores (on test set):")
        report.append("-" * 30)
        bleu_scores = self.calculate_bleu_score()
        for metric, score in bleu_scores.items():
            if metric != 'num_samples':
                report.append(f"  {metric}: {score:.2f}")
        report.append(f"  Evaluated on {bleu_scores['num_samples']} samples")
        report.append("")
        
        # Translation examples
        report.append("Translation Examples:")
        report.append("-" * 30)
        examples = self.get_translation_examples(num_examples=5)
        for i, example in enumerate(examples, 1):
            report.append(f"\nExample {i}:")
            report.append(f"  Source: {example['source']}")
            report.append(f"  Reference: {example['reference']}")
            report.append(f"  Prediction: {example['prediction']}")
            report.append(f"  BLEU Score: {example['bleu_score']:.2f}")
        report.append("")
        
        # Error analysis
        report.append("Error Analysis:")
        report.append("-" * 30)
        errors = self.analyze_errors(num_samples=200)
        
        report.append("Error Type Distribution:")
        for error_type, percentage in errors['error_percentages'].items():
            report.append(f"  {error_type}: {percentage:.1f}%")
        
        report.append(f"\nTranslation Length Statistics:")
        report.append(f"  Average length ratio: {errors['avg_length_ratio']:.2f}")
        report.append(f"  Std dev length ratio: {errors['std_length_ratio']:.2f}")
        
        report.append(f"\nBLEU Score Statistics:")
        report.append(f"  Average BLEU: {errors['avg_bleu']:.2f}")
        report.append(f"  Std dev BLEU: {errors['std_bleu']:.2f}")
        
        report.append("")
        report.append("=" * 60)
        
        # Join report
        report_str = '\n'.join(report)
        
        # Save to file if specified
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_str)
            print(f"Report saved to {output_file}")
        
        return report_str


def evaluate_model(model_path: str, data_dict: Dict):
    """
    Main evaluation function
    
    Args:
        model_path: Path to model checkpoint
        data_dict: Dictionary containing data loaders and vocabularies
    """
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=Config.DEVICE, 
                           weights_only=False)
    
    # Create model
    model = EncoderDecoderLSTM_Attn(
        src_vocab_size=len(data_dict['src_vocab']),
        tgt_vocab_size=len(data_dict['tgt_vocab']),
        embedding_dim=Config.EMBEDDING_DIM,
        hidden_size=Config.HIDDEN_SIZE,
        num_layers=Config.NUM_LAYERS,
        dropout=Config.DROPOUT
    )
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create evaluator
    evaluator = Evaluator(
        model=model,
        test_loader=data_dict['test_loader'],
        src_vocab=data_dict['src_vocab'],
        tgt_vocab=data_dict['tgt_vocab']
    )
    
    # Generate report
    report = evaluator.generate_report(
        output_file='evaluation_report.txt'
    )
    
    print(report)
    
    return evaluator


if __name__ == "__main__":
    print("Evaluation module loaded. Use main.py to run evaluation.")
