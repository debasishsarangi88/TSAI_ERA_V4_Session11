"""
Train BPE tokenizer on Odia corpus
"""

import sys
from bpe_tokenizer import BPETokenizer
from download_odia_corpus import create_extended_corpus, save_corpus
import json


def train_tokenizer(vocab_size=4500, corpus_file="odia_corpus.txt"):
    """
    Train the BPE tokenizer with specified vocabulary size.
    
    Args:
        vocab_size: Target vocabulary size (must be < 5000)
        corpus_file: Path to corpus file
    """
    print("="*60)
    print("BPE TOKENIZER TRAINING FOR ODIA LANGUAGE")
    print("="*60)
    
    # Load or create corpus
    try:
        print(f"\nLoading corpus from {corpus_file}...")
        with open(corpus_file, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(texts)} texts from file")
    except FileNotFoundError:
        print(f"\nCorpus file not found. Creating new corpus...")
        texts = create_extended_corpus()
        save_corpus(texts, corpus_file)
    
    # Statistics about corpus
    total_chars = sum(len(text) for text in texts)
    total_bytes = sum(len(text.encode('utf-8')) for text in texts)
    
    print(f"\nCorpus Statistics:")
    print(f"  Number of texts: {len(texts)}")
    print(f"  Total characters: {total_chars:,}")
    print(f"  Total bytes (UTF-8): {total_bytes:,}")
    print(f"  Average text length: {total_chars/len(texts):.1f} chars")
    
    # Initialize and train tokenizer
    print(f"\n{'='*60}")
    print(f"Training BPE with target vocab size: {vocab_size}")
    print(f"{'='*60}\n")
    
    tokenizer = BPETokenizer(vocab_size=vocab_size)
    tokenizer.train(texts, verbose=True)
    
    # Calculate compression ratio
    print(f"\n{'='*60}")
    print("EVALUATING TOKENIZER")
    print(f"{'='*60}\n")
    
    compression_ratio = tokenizer.calculate_compression_ratio(texts)
    
    # Get tokenizer statistics
    stats = tokenizer.get_stats()
    
    print(f"\nFinal Statistics:")
    print(f"  Vocabulary size: {stats['vocab_size']}")
    print(f"  Number of merges: {stats['num_merges']}")
    print(f"  Compression ratio: {compression_ratio:.4f}")
    
    # Check if requirements are met
    print(f"\n{'='*60}")
    print("REQUIREMENTS CHECK")
    print(f"{'='*60}")
    
    vocab_ok = stats['vocab_size'] < 5000
    compression_ok = compression_ratio >= 3.2
    
    print(f"  ✓ Vocabulary < 5000: {vocab_ok} (actual: {stats['vocab_size']})")
    print(f"  ✓ Compression ≥ 3.2: {compression_ok} (actual: {compression_ratio:.4f})")
    
    if vocab_ok and compression_ok:
        print(f"\n{'='*60}")
        print("✓ ALL REQUIREMENTS MET!")
        print(f"{'='*60}\n")
    else:
        print(f"\n{'='*60}")
        print("✗ REQUIREMENTS NOT MET - Need to adjust parameters")
        print(f"{'='*60}\n")
        
        if not compression_ok:
            print("  Suggestion: Try reducing vocab_size to increase compression")
        if not vocab_ok:
            print("  Suggestion: Reduce vocab_size parameter")
    
    # Save tokenizer
    tokenizer_file = f"odia_bpe_tokenizer_{vocab_size}.pkl"
    tokenizer.save(tokenizer_file)
    
    # Save statistics
    results = {
        'vocab_size': stats['vocab_size'],
        'num_merges': stats['num_merges'],
        'compression_ratio': compression_ratio,
        'requirements_met': vocab_ok and compression_ok,
        'corpus_texts': len(texts),
        'corpus_bytes': total_bytes,
    }
    
    results_file = f"training_results_{vocab_size}.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_file}")
    
    return tokenizer, compression_ratio, vocab_ok and compression_ok


def find_optimal_vocab_size():
    """
    Try different vocabulary sizes to find one that meets requirements.
    """
    print("\nSearching for optimal vocabulary size...")
    print("="*60)
    
    # Try different vocabulary sizes
    vocab_sizes = [4500, 4000, 3500, 3000, 2500, 2000]
    
    best_tokenizer = None
    best_vocab_size = None
    best_compression = 0
    
    for vocab_size in vocab_sizes:
        print(f"\n\nTrying vocab_size = {vocab_size}...")
        print("-"*60)
        
        tokenizer, compression_ratio, requirements_met = train_tokenizer(vocab_size)
        
        if requirements_met:
            print(f"\n✓ Found suitable configuration!")
            print(f"  Vocab size: {vocab_size}")
            print(f"  Compression ratio: {compression_ratio:.4f}")
            return tokenizer, vocab_size, compression_ratio
        
        if compression_ratio > best_compression:
            best_tokenizer = tokenizer
            best_vocab_size = vocab_size
            best_compression = compression_ratio
        
        # If compression ratio is already above target, we found it
        if compression_ratio >= 3.2:
            print(f"\n✓ Found suitable configuration!")
            return tokenizer, vocab_size, compression_ratio
    
    print(f"\nBest result found:")
    print(f"  Vocab size: {best_vocab_size}")
    print(f"  Compression ratio: {best_compression:.4f}")
    
    return best_tokenizer, best_vocab_size, best_compression


if __name__ == "__main__":
    # You can specify vocab_size as command line argument
    if len(sys.argv) > 1:
        vocab_size = int(sys.argv[1])
        train_tokenizer(vocab_size)
    else:
        # Find optimal vocabulary size
        find_optimal_vocab_size()

