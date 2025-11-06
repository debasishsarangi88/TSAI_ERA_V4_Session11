"""
Gradio App for Odia BPE Tokenizer
Deploy this on Hugging Face Spaces
"""

import gradio as gr
import pickle
from typing import List, Dict
import json


class BPETokenizer:
    """
    A Byte Pair Encoding tokenizer implementation for Odia language.
    """
    
    def __init__(self, vocab_size: int = 5000):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.merges = []
        self.byte_encoder = self._bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        
    def _bytes_to_unicode(self) -> Dict[int, str]:
        """Create a mapping from bytes to unicode characters."""
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
    
    def _tokenize_word(self, word: str) -> List[str]:
        """Tokenize a single word using learned merges."""
        import re
        word = tuple(list(word) + ['</w>'])
        
        while len(word) > 1:
            pairs = [(word[i], word[i+1]) for i in range(len(word)-1)]
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
        """Encode text into token IDs."""
        import re
        text_bytes = text.encode('utf-8')
        text_unicode = ''.join(self.byte_encoder[b] for b in text_bytes)
        words = re.findall(r'\S+|\s+', text_unicode)
        
        token_ids = []
        for word in words:
            tokens = self._tokenize_word(word)
            for token in tokens:
                if token in self.vocab:
                    token_ids.append(self.vocab[token])
                else:
                    for char in token:
                        if char in self.vocab:
                            token_ids.append(self.vocab[char])
        
        return token_ids
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs back into text."""
        id_to_token = {idx: token for token, idx in self.vocab.items()}
        tokens = [id_to_token[tid] for tid in token_ids if tid in id_to_token]
        text_unicode = ''.join(tokens).replace('</w>', '')
        
        try:
            text_bytes = bytes([self.byte_decoder[c] for c in text_unicode])
            text = text_bytes.decode('utf-8', errors='replace')
        except:
            text = text_unicode
        
        return text
    
    def load(self, filepath: str):
        """Load the tokenizer from a file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.vocab_size = data['vocab_size']
            self.vocab = data['vocab']
            self.merges = data['merges']
            self.byte_encoder = data['byte_encoder']
            self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}


# Load the trained tokenizer
tokenizer = BPETokenizer()
tokenizer.load('odia_bpe_tokenizer_4500.pkl')

# Load training results
with open('training_results_4500.json', 'r') as f:
    training_results = json.load(f)


def tokenize_text(text: str):
    """
    Tokenize input text and return analysis.
    """
    if not text.strip():
        return "⚠️ Please enter some text to tokenize.", "", "", "", ""
    
    try:
        # Encode
        token_ids = tokenizer.encode(text)
        
        # Decode
        decoded = tokenizer.decode(token_ids)
        
        # Statistics
        original_bytes = len(text.encode('utf-8'))
        num_tokens = len(token_ids)
        compression_ratio = original_bytes / num_tokens if num_tokens > 0 else 0
        match = "✅ Perfect match!" if decoded == text else "⚠️ Decoding mismatch"
        
        # Format output
        token_ids_str = str(token_ids[:50]) + ("..." if len(token_ids) > 50 else "")
        
        stats = f"""
📊 **Tokenization Statistics:**
- Original text length: {len(text)} characters
- Original size: {original_bytes} bytes
- Number of tokens: {num_tokens}
- Compression ratio: {compression_ratio:.2f}
- Decoding accuracy: {match}
"""
        
        return decoded, token_ids_str, str(num_tokens), f"{compression_ratio:.2f}", stats
        
    except Exception as e:
        return f"❌ Error: {str(e)}", "", "", "", ""


def show_tokenizer_info():
    """Display tokenizer information."""
    info = f"""
# 🎯 Odia BPE Tokenizer

## ✅ Requirements Met

| Requirement | Target | Achieved | Status |
|-------------|--------|----------|--------|
| **Token Count** | < 5,000 | **{training_results['vocab_size']:,}** | ✅ |
| **Compression Ratio** | ≥ 3.2 | **{training_results['compression_ratio']:.2f}** | ✅ |

## 📈 Performance Metrics

- **Vocabulary Size**: {training_results['vocab_size']:,} tokens
- **Compression Ratio**: {training_results['compression_ratio']:.2f}
- **Corpus Size**: {training_results['corpus_bytes']:,} bytes
- **Training Texts**: {training_results['corpus_texts']} samples

## 🔤 About Odia Language

Odia (ଓଡ଼ିଆ) is an Indo-Aryan language spoken by ~45 million people in India. 
It is one of the Classical Languages of India with a rich literary tradition.

## 🛠️ How to Use

1. Enter Odia text in the input box
2. Click "Tokenize" to see the results
3. View token IDs, compression ratio, and decoded output
"""
    return info


# Create Gradio Interface
with gr.Blocks(title="Odia BPE Tokenizer", theme=gr.themes.Soft()) as demo:
    
    gr.Markdown("# 🔤 Odia Language BPE Tokenizer")
    gr.Markdown("### Byte Pair Encoding tokenizer for Odia (ଓଡ଼ିଆ) language")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown(show_tokenizer_info())
    
    gr.Markdown("---")
    gr.Markdown("## 🧪 Try the Tokenizer")
    
    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(
                label="Input Odia Text",
                placeholder="ଓଡ଼ିଆ ଭାଷାରେ କିଛି ଲେଖନ୍ତୁ...",
                lines=5
            )
            
            with gr.Row():
                tokenize_btn = gr.Button("🔄 Tokenize", variant="primary")
                clear_btn = gr.ClearButton([input_text], value="🗑️ Clear")
    
    with gr.Row():
        with gr.Column():
            decoded_output = gr.Textbox(
                label="Decoded Output",
                lines=3
            )
        with gr.Column():
            token_ids_output = gr.Textbox(
                label="Token IDs (first 50)",
                lines=3
            )
    
    with gr.Row():
        with gr.Column():
            num_tokens_output = gr.Textbox(
                label="Number of Tokens"
            )
        with gr.Column():
            compression_output = gr.Textbox(
                label="Compression Ratio"
            )
    
    stats_output = gr.Markdown(label="Statistics")
    
    # Examples
    gr.Markdown("## 📝 Example Texts")
    gr.Examples(
        examples=[
            ["ଓଡ଼ିଆ ଭାରତର ଏକ ପ୍ରାଚୀନ ଭାଷା ଅଟେ ।"],
            ["ନମସ୍କାର ! ଆପଣ କେମିତି ଅଛନ୍ତି ?"],
            ["ଓଡ଼ିଶା ରାଜ୍ୟର ରାଜଧାନୀ ଭୁବନେଶ୍ୱର ।"],
            ["କମ୍ପ୍ୟୁଟର ବିଜ୍ଞାନ ଆଧୁନିକ ଯୁଗର ମୂଳଦୁଆ ।"],
            ["ଶିକ୍ଷା ହିଁ ଜୀବନର ପ୍ରକୃତ ସମ୍ପତ୍ତି ।"],
        ],
        inputs=input_text,
    )
    
    # Event handlers
    tokenize_btn.click(
        fn=tokenize_text,
        inputs=[input_text],
        outputs=[decoded_output, token_ids_output, num_tokens_output, compression_output, stats_output]
    )
    
    gr.Markdown("---")
    gr.Markdown("""
    ### 📚 About This Project
    
    This BPE tokenizer was trained on a diverse Odia language corpus and successfully meets all requirements:
    - ✅ Vocabulary size < 5,000 tokens
    - ✅ Compression ratio ≥ 3.2
    
    **Created for**: TSAI ERA V4 Session 11
    
    **GitHub**: [View Source Code](https://github.com/yourusername/odia-bpe-tokenizer)
    """)

# Launch the app
if __name__ == "__main__":
    demo.launch()
