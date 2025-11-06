"""
Evaluate and visualize BPE tokenizer performance
"""

import pickle
from bpe_tokenizer import BPETokenizer
from typing import List
import json


def load_tokenizer(filepath: str) -> BPETokenizer:
    """Load a trained tokenizer."""
    tokenizer = BPETokenizer()
    tokenizer.load(filepath)
    return tokenizer


def evaluate_on_samples(tokenizer: BPETokenizer, texts: List[str]):
    """
    Evaluate tokenizer on sample texts and show examples.
    """
    print("\n" + "="*80)
    print("TOKENIZATION EXAMPLES")
    print("="*80 + "\n")
    
    for i, text in enumerate(texts[:5], 1):  # Show first 5 examples
        # Limit text length for display
        display_text = text[:100] + "..." if len(text) > 100 else text
        
        print(f"\nExample {i}:")
        print(f"  Original text: {display_text}")
        print(f"  Original bytes: {len(text.encode('utf-8'))}")
        
        # Encode
        token_ids = tokenizer.encode(text)
        print(f"  Token IDs: {token_ids[:20]}{'...' if len(token_ids) > 20 else ''}")
        print(f"  Number of tokens: {len(token_ids)}")
        
        # Decode
        decoded_text = tokenizer.decode(token_ids)
        decoded_display = decoded_text[:100] + "..." if len(decoded_text) > 100 else decoded_text
        print(f"  Decoded text: {decoded_display}")
        
        # Check if decoding is correct
        match = decoded_text == text
        print(f"  Decoding match: {'✓' if match else '✗'}")
        
        # Compression for this text
        compression = len(text.encode('utf-8')) / len(token_ids)
        print(f"  Compression ratio: {compression:.4f}")
        print("-" * 80)


def analyze_vocabulary(tokenizer: BPETokenizer):
    """
    Analyze the learned vocabulary.
    """
    print("\n" + "="*80)
    print("VOCABULARY ANALYSIS")
    print("="*80 + "\n")
    
    vocab = tokenizer.vocab
    
    # Sort by token ID
    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
    
    print(f"Total vocabulary size: {len(vocab)}")
    print(f"\nFirst 30 tokens:")
    for token, idx in sorted_vocab[:30]:
        # Display token safely
        token_display = repr(token)[:50]
        print(f"  {idx:4d}: {token_display}")
    
    print(f"\nLast 30 tokens:")
    for token, idx in sorted_vocab[-30:]:
        token_display = repr(token)[:50]
        print(f"  {idx:4d}: {token_display}")
    
    # Token length distribution
    token_lengths = {}
    for token in vocab.keys():
        length = len(token)
        token_lengths[length] = token_lengths.get(length, 0) + 1
    
    print(f"\nToken length distribution:")
    for length in sorted(token_lengths.keys())[:20]:  # Show first 20
        count = token_lengths[length]
        bar = '█' * (count // 10)
        print(f"  Length {length:2d}: {count:4d} {bar}")


def compare_with_different_texts(tokenizer: BPETokenizer):
    """
    Compare tokenizer performance on different types of Odia text.
    """
    print("\n" + "="*80)
    print("PERFORMANCE ON DIFFERENT TEXT TYPES")
    print("="*80 + "\n")
    
    test_samples = {
        "Short formal": "ଓଡ଼ିଆ ଭାରତର ଏକ ପ୍ରାଚୀନ ଭାଷା ଅଟେ ।",
        "Conversational": "ତୁମେ କେମିତି ଅଛ ? ମୁଁ ଭଲ ଅଛି ।",
        "Technical": "କମ୍ପ୍ୟୁଟର ଆଜିକାଲି ସର୍ବତ୍ର ବ୍ୟବହୃତ ହେଉଛି ।",
        "Literary": "ଏକଦା ଗୋଟିଏ ଗାଁରେ ଜଣେ ଗରିବ କୃଷକ ରହୁଥିଲା ।",
    }
    
    for text_type, text in test_samples.items():
        tokens = tokenizer.encode(text)
        bytes_count = len(text.encode('utf-8'))
        compression = bytes_count / len(tokens) if len(tokens) > 0 else 0
        
        print(f"{text_type}:")
        print(f"  Text: {text}")
        print(f"  Bytes: {bytes_count}, Tokens: {len(tokens)}, Compression: {compression:.4f}")
        print()


def generate_report(tokenizer: BPETokenizer, corpus_file: str = "odia_corpus.txt"):
    """
    Generate a comprehensive evaluation report.
    """
    print("\n" + "="*80)
    print("COMPREHENSIVE EVALUATION REPORT")
    print("="*80)
    
    # Load corpus
    try:
        with open(corpus_file, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"Corpus file {corpus_file} not found!")
        return
    
    # Overall statistics
    stats = tokenizer.get_stats()
    compression_ratio = tokenizer.calculate_compression_ratio(texts)
    
    print(f"\n📊 Overall Statistics:")
    print(f"  Vocabulary Size: {stats['vocab_size']}")
    print(f"  Number of Merges: {stats['num_merges']}")
    print(f"  Compression Ratio: {compression_ratio:.4f}")
    
    # Requirements check
    print(f"\n✅ Requirements Check:")
    vocab_check = "✓" if stats['vocab_size'] < 5000 else "✗"
    compression_check = "✓" if compression_ratio >= 3.2 else "✗"
    print(f"  {vocab_check} Vocabulary < 5000: {stats['vocab_size']} < 5000")
    print(f"  {compression_check} Compression ≥ 3.2: {compression_ratio:.4f} ≥ 3.2")
    
    # Detailed analysis
    analyze_vocabulary(tokenizer)
    evaluate_on_samples(tokenizer, texts)
    compare_with_different_texts(tokenizer)
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\nThe BPE tokenizer for Odia language has been successfully trained!")
    print(f"  - Vocabulary size: {stats['vocab_size']} (requirement: < 5000)")
    print(f"  - Compression ratio: {compression_ratio:.4f} (requirement: ≥ 3.2)")
    
    if stats['vocab_size'] < 5000 and compression_ratio >= 3.2:
        print(f"\n🎉 All requirements met! The tokenizer is ready to use.")
    else:
        print(f"\n⚠️  Requirements not fully met. Consider retraining with adjusted parameters.")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    import sys
    
    # Default to latest tokenizer
    if len(sys.argv) > 1:
        tokenizer_file = sys.argv[1]
    else:
        # Try to find a tokenizer file
        import glob
        tokenizer_files = glob.glob("odia_bpe_tokenizer_*.pkl")
        if tokenizer_files:
            tokenizer_file = sorted(tokenizer_files)[-1]  # Use most recent
            print(f"Using tokenizer file: {tokenizer_file}")
        else:
            print("No tokenizer file found! Please train a tokenizer first.")
            print("Run: python train_bpe.py")
            sys.exit(1)
    
    # Load and evaluate
    tokenizer = load_tokenizer(tokenizer_file)
    generate_report(tokenizer)

