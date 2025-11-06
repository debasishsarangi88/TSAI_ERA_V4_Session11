"""
Byte Pair Encoding (BPE) Tokenizer for Odia Language
"""

import re
import json
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Set
import pickle


class BPETokenizer:
    """
    A Byte Pair Encoding tokenizer implementation for Odia language.
    """
    
    def __init__(self, vocab_size: int = 5000):
        """
        Initialize the BPE tokenizer.
        
        Args:
            vocab_size: Maximum vocabulary size (must be < 5000 as per requirements)
        """
        self.vocab_size = vocab_size
        self.vocab = {}  # token -> id mapping
        self.merges = []  # list of merge operations (pair, new_token)
        self.byte_encoder = self._bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        
    def _bytes_to_unicode(self) -> Dict[int, str]:
        """
        Create a mapping from bytes to unicode characters.
        This helps handle any byte gracefully.
        """
        bs = list(range(ord("!"), ord("~")+1)) + list(range(ord("¡"), ord("¬")+1)) + list(range(ord("®"), ord("ÿ")+1))
        cs = bs[:]
        n = 0
        for b in range(2**8):
            if b not in bs:
                bs.append(b)
                cs.append(2**8+n)
                n += 1
        cs = [chr(n) for n in cs]
        return dict(zip(bs, cs))
    
    def _get_stats(self, word_freqs: Dict[Tuple[str, ...], int]) -> Counter:
        """
        Count the frequency of adjacent pairs in the vocabulary.
        
        Args:
            word_freqs: Dictionary mapping word tuples to their frequencies
            
        Returns:
            Counter of pair frequencies
        """
        pairs = Counter()
        for word, freq in word_freqs.items():
            for i in range(len(word) - 1):
                pairs[(word[i], word[i+1])] += freq
        return pairs
    
    def _merge_vocab(self, pair: Tuple[str, str], word_freqs: Dict[Tuple[str, ...], int]) -> Dict[Tuple[str, ...], int]:
        """
        Merge all occurrences of the most frequent pair.
        
        Args:
            pair: The pair to merge
            word_freqs: Current word frequencies
            
        Returns:
            Updated word frequencies after merge
        """
        new_word_freqs = {}
        bigram = ' '.join(pair)
        replacement = ''.join(pair)
        
        for word, freq in word_freqs.items():
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == pair[0] and word[i+1] == pair[1]:
                    new_word.append(replacement)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word_freqs[tuple(new_word)] = freq
        
        return new_word_freqs
    
    def train(self, texts: List[str], verbose: bool = True):
        """
        Train the BPE tokenizer on a corpus of texts.
        
        Args:
            texts: List of text strings to train on
            verbose: Whether to print progress information
        """
        if verbose:
            print(f"Training BPE tokenizer with vocab_size={self.vocab_size}")
            print(f"Corpus size: {len(texts)} texts")
        
        # Preprocess: convert text to bytes and then to unicode characters
        word_freqs = Counter()
        for text in texts:
            # Convert text to bytes then to our unicode representation
            text_bytes = text.encode('utf-8')
            text_unicode = ''.join(self.byte_encoder[b] for b in text_bytes)
            
            # Split into words (keeping whitespace as part of words)
            words = re.findall(r'\S+|\s+', text_unicode)
            for word in words:
                # Split word into characters with end-of-word marker
                word_tuple = tuple(list(word) + ['</w>'])
                word_freqs[word_tuple] += 1
        
        if verbose:
            print(f"Initial vocabulary size: {sum(len(set(word)) for word in word_freqs.keys())}")
            total_chars = sum(len(word) * freq for word, freq in word_freqs.items())
            print(f"Total characters: {total_chars}")
        
        # Build initial vocabulary from all unique characters
        vocab = set()
        for word in word_freqs.keys():
            vocab.update(word)
        
        # Add base vocabulary
        self.vocab = {token: idx for idx, token in enumerate(sorted(vocab))}
        
        if verbose:
            print(f"Base vocabulary size: {len(self.vocab)}")
        
        # Perform BPE merges
        num_merges = self.vocab_size - len(self.vocab)
        
        for i in range(num_merges):
            pairs = self._get_stats(word_freqs)
            
            if not pairs:
                if verbose:
                    print(f"No more pairs to merge at iteration {i}")
                break
            
            best_pair = max(pairs, key=pairs.get)
            
            if verbose and i % 100 == 0:
                print(f"Merge {i}/{num_merges}: {best_pair} (freq: {pairs[best_pair]})")
            
            # Merge the best pair
            word_freqs = self._merge_vocab(best_pair, word_freqs)
            
            # Add to vocabulary and merges
            new_token = ''.join(best_pair)
            self.merges.append(best_pair)
            if new_token not in self.vocab:
                self.vocab[new_token] = len(self.vocab)
        
        if verbose:
            print(f"\nFinal vocabulary size: {len(self.vocab)}")
            print(f"Number of merges: {len(self.merges)}")
    
    def _tokenize_word(self, word: str) -> List[str]:
        """
        Tokenize a single word using learned merges.
        
        Args:
            word: Word to tokenize (in byte-encoded unicode)
            
        Returns:
            List of tokens
        """
        # Start with character-level tokenization
        word = tuple(list(word) + ['</w>'])
        
        # Apply merges
        while len(word) > 1:
            pairs = [(word[i], word[i+1]) for i in range(len(word)-1)]
            
            # Find the earliest merge that applies
            min_merge_idx = float('inf')
            best_pair = None
            
            for pair in pairs:
                if pair in self.merges:
                    merge_idx = self.merges.index(pair)
                    if merge_idx < min_merge_idx:
                        min_merge_idx = merge_idx
                        best_pair = pair
            
            if best_pair is None:
                break
            
            # Apply the merge
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == best_pair[0] and word[i+1] == best_pair[1]:
                    new_word.append(''.join(best_pair))
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = tuple(new_word)
        
        return list(word)
    
    def encode(self, text: str) -> List[int]:
        """
        Encode text into token IDs.
        
        Args:
            text: Input text
            
        Returns:
            List of token IDs
        """
        # Convert text to bytes then to unicode
        text_bytes = text.encode('utf-8')
        text_unicode = ''.join(self.byte_encoder[b] for b in text_bytes)
        
        # Split into words
        words = re.findall(r'\S+|\s+', text_unicode)
        
        # Tokenize each word
        token_ids = []
        for word in words:
            tokens = self._tokenize_word(word)
            for token in tokens:
                if token in self.vocab:
                    token_ids.append(self.vocab[token])
                else:
                    # Fallback: tokenize character by character
                    for char in token:
                        if char in self.vocab:
                            token_ids.append(self.vocab[char])
        
        return token_ids
    
    def decode(self, token_ids: List[int]) -> str:
        """
        Decode token IDs back into text.
        
        Args:
            token_ids: List of token IDs
            
        Returns:
            Decoded text string
        """
        # Create reverse vocabulary
        id_to_token = {idx: token for token, idx in self.vocab.items()}
        
        # Get tokens
        tokens = [id_to_token[tid] for tid in token_ids if tid in id_to_token]
        
        # Join tokens and remove end-of-word markers
        text_unicode = ''.join(tokens).replace('</w>', '')
        
        # Convert back to bytes then to UTF-8 string
        try:
            text_bytes = bytes([self.byte_decoder[c] for c in text_unicode])
            text = text_bytes.decode('utf-8', errors='replace')
        except:
            text = text_unicode  # Fallback
        
        return text
    
    def calculate_compression_ratio(self, texts: List[str]) -> float:
        """
        Calculate the compression ratio on a set of texts.
        Compression ratio = original_bytes / encoded_tokens
        
        Args:
            texts: List of texts to evaluate
            
        Returns:
            Compression ratio
        """
        total_bytes = sum(len(text.encode('utf-8')) for text in texts)
        total_tokens = sum(len(self.encode(text)) for text in texts)
        
        if total_tokens == 0:
            return 0.0
        
        compression_ratio = total_bytes / total_tokens
        return compression_ratio
    
    def save(self, filepath: str):
        """Save the tokenizer to a file."""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'vocab_size': self.vocab_size,
                'vocab': self.vocab,
                'merges': self.merges,
                'byte_encoder': self.byte_encoder,
            }, f)
        print(f"Tokenizer saved to {filepath}")
    
    def load(self, filepath: str):
        """Load the tokenizer from a file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.vocab_size = data['vocab_size']
            self.vocab = data['vocab']
            self.merges = data['merges']
            self.byte_encoder = data['byte_encoder']
            self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        print(f"Tokenizer loaded from {filepath}")
    
    def get_stats(self) -> Dict[str, any]:
        """Get statistics about the tokenizer."""
        return {
            'vocab_size': len(self.vocab),
            'num_merges': len(self.merges),
            'target_vocab_size': self.vocab_size,
        }

