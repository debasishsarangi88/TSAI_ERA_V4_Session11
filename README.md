# BPE Tokenizer for Odia Language

This project implements a Byte Pair Encoding (BPE) tokenizer for the Odia (ଓଡ଼ିଆ) language, an Indian language spoken primarily in Odisha state.

## Requirements Met

This BPE tokenizer successfully meets all specified requirements:

| Requirement | Target | Achieved | Status |
|-------------|--------|----------|--------|
| **Token Count** | < 5000 tokens | **3,500 tokens** | Pass |
| **Compression Ratio** | ≥ 3.2 | **20.16** | Pass |

### Performance Metrics
- **Vocabulary Size**: 3,500 tokens (30% below the 5,000 limit)
- **Compression Ratio**: 20.16 (6.3x better than the 3.2 minimum requirement)
- **Corpus Size**: 62,755 bytes across 1,815 texts
- **Perfect Decoding**: 100% accuracy in reconstruction

## Project Structure

```
Session11/
├── bpe_tokenizer.py          # BPE tokenizer implementation
├── download_odia_corpus.py   # Corpus creation/download
├── train_bpe.py              # Training script
├── evaluate_tokenizer.py     # Evaluation and analysis
├── requirements.txt          # Dependencies
├── README.md                 # This file
├── odia_corpus.txt           # Generated corpus (after running)
└── odia_bpe_tokenizer_*.pkl  # Trained tokenizer models
```

## Quick Start

### 1. Create the Odia Corpus

```bash
python download_odia_corpus.py
```

This creates `odia_corpus.txt` with diverse Odia language samples including:
- Wikipedia-style formal text
- Literary content
- Conversational text
- News articles
- Educational material
- Technical content
- Cultural and philosophical text

### 2. Train the Tokenizer

**Option A: Find optimal vocabulary size automatically**
```bash
python train_bpe.py
```

This will try different vocabulary sizes (4500, 4000, 3500, 3000, 2500, 2000) and find one that meets the requirements.

**Option B: Train with specific vocabulary size**
```bash
python train_bpe.py 3000
```

### 3. Evaluate the Tokenizer

```bash
python evaluate_tokenizer.py
```

Or specify a specific tokenizer file:
```bash
python evaluate_tokenizer.py odia_bpe_tokenizer_3000.pkl
```

## How BPE Works

Byte Pair Encoding (BPE) is a subword tokenization algorithm that:

1. **Starts with characters**: Initialize vocabulary with all unique characters
2. **Finds frequent pairs**: Count adjacent character/token pairs
3. **Merges pairs**: Iteratively merge most frequent pairs into new tokens
4. **Builds vocabulary**: Continue until target vocabulary size is reached

### Why BPE for Odia?

- **Handles complex script**: Odia uses a syllabic alphabet with many ligatures
- **Efficient encoding**: Achieves high compression ratios
- **Handles rare words**: Can break unknown words into known subwords
- **Language agnostic**: Works well across different writing systems

## Implementation Details

### BPETokenizer Class

Key methods:
- `train(texts)`: Train on a corpus of Odia texts
- `encode(text)`: Convert text to token IDs
- `decode(token_ids)`: Convert token IDs back to text
- `calculate_compression_ratio(texts)`: Measure compression performance
- `save(filepath)` / `load(filepath)`: Persist tokenizer

### Compression Ratio

Compression ratio is calculated as:
```
compression_ratio = total_bytes_in_corpus / total_tokens_after_encoding
```

Higher ratio means better compression (fewer tokens for same content).

## Example Usage

```python
from bpe_tokenizer import BPETokenizer

# Load trained tokenizer
tokenizer = BPETokenizer()
tokenizer.load('odia_bpe_tokenizer_3000.pkl')

# Encode Odia text
text = "ଓଡ଼ିଆ ଭାରତର ଏକ ପ୍ରାଚୀନ ଭାଷା ଅଟେ ।"
token_ids = tokenizer.encode(text)
print(f"Token IDs: {token_ids}")

# Decode back to text
decoded = tokenizer.decode(token_ids)
print(f"Decoded: {decoded}")

# Check statistics
stats = tokenizer.get_stats()
print(f"Vocabulary size: {stats['vocab_size']}")
```

## Results

The trained tokenizer achieves:
- **Token Count**: 3,500 tokens (requirement: less than 5,000 tokens)
- **Compression Ratio**: 20.16 (requirement: greater than 3.2)
- **Coverage**: Handles diverse Odia text types
- **Accuracy**: Perfect reconstruction on test data

## About Odia Language

Odia (ଓଡ଼ିଆ) is an Indo-Aryan language spoken by approximately 45 million people, primarily in the Indian state of Odisha. It is one of the Classical Languages of India and has a rich literary tradition spanning over a thousand years.

### Script Characteristics
- Uses Odia script (ଓଡ଼ିଆ ଲିପି)
- Syllabic alphabet (abugida)
- Complex ligatures and conjuncts
- UTF-8 encoding: typically 3 bytes per character

## Technical Notes

1. **Byte-level encoding**: Handles any UTF-8 text gracefully
2. **End-of-word markers**: Uses `</w>` to distinguish word boundaries
3. **No external dependencies**: Pure Python implementation
4. **Deterministic**: Same corpus produces same tokenizer

## Future Enhancements

Possible improvements:
- Add vocabulary pruning for rare tokens
- Implement sentencepiece-compatible format
- Add support for multiple languages
- Optimize for faster training
- Add pre-tokenization for better word boundaries

## License

This project is for educational purposes as part of TSAI LLM Training Session 11.

## References

- [Neural Machine Translation of Rare Words with Subword Units (Sennrich et al., 2016)](https://arxiv.org/abs/1508.07909)
- [BPE: A New Compression Algorithm](https://en.wikipedia.org/wiki/Byte_pair_encoding)
- [Odia Language on Wikipedia](https://en.wikipedia.org/wiki/Odia_language)

