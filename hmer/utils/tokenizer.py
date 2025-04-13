"""
Tokenizer for LaTeX expressions.
"""

import re
import os
import json
from typing import List, Dict, Optional, Union, Tuple


class LaTeXTokenizer:
    """
    Tokenizer for LaTeX expressions.
    """
    
    def __init__(self, 
                vocab_file: Optional[str] = None,
                special_tokens: Optional[Dict[str, int]] = None,
                max_vocab_size: int = 1000):
        """
        Initialize the tokenizer.
        
        Args:
            vocab_file: Optional file path to load vocabulary from
            special_tokens: Dictionary mapping special token names to token IDs
            max_vocab_size: Maximum vocabulary size for building from data
        """
        self.max_vocab_size = max_vocab_size
        
        # Define special tokens
        self.special_tokens = special_tokens or {
            '<pad>': 0,
            '<sos>': 1,  # Start of sequence
            '<eos>': 2,  # End of sequence
            '<unk>': 3,  # Unknown token
        }
        
        # Initialize vocabulary with special tokens
        self.token_to_id = {token: idx for token, idx in self.special_tokens.items()}
        self.id_to_token = {idx: token for token, idx in self.special_tokens.items()}
        
        # Load vocabulary if provided
        if vocab_file and os.path.exists(vocab_file):
            self.load_vocab(vocab_file)
            
        # Regex pattern for tokenizing LaTeX
        self.pattern = re.compile(r'(\\[a-zA-Z]+|[^a-zA-Z0-9\s]|\s+|[a-zA-Z0-9]+)')
        
        # Keep track of token frequencies for vocabulary building
        self.token_freqs = {}
        self.vocab_size = len(self.token_to_id)
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize a LaTeX expression.
        
        Args:
            text: LaTeX expression
            
        Returns:
            List of tokens
        """
        tokens = []
        
        # Extract all tokens
        matches = self.pattern.findall(text)
        for match in matches:
            if match.strip():  # Skip empty strings
                tokens.append(match)
                
                # Update token frequency if building vocabulary
                if match not in self.token_to_id:
                    self.token_freqs[match] = self.token_freqs.get(match, 0) + 1
        
        return tokens
    
    def encode(self, text: str, add_special_tokens: bool = True, 
              max_length: Optional[int] = None) -> List[int]:
        """
        Encode a LaTeX expression to token IDs.
        
        Args:
            text: LaTeX expression
            add_special_tokens: Whether to add start and end tokens
            max_length: Maximum sequence length (truncates if longer)
            
        Returns:
            List of token IDs
        """
        tokens = self.tokenize(text)
        
        # Convert tokens to IDs
        token_ids = []
        if add_special_tokens:
            token_ids.append(self.special_tokens['<sos>'])
            
        for token in tokens:
            if token in self.token_to_id:
                token_ids.append(self.token_to_id[token])
            else:
                # Add to vocabulary if building it
                if len(self.token_to_id) < self.max_vocab_size:
                    idx = len(self.token_to_id)
                    self.token_to_id[token] = idx
                    self.id_to_token[idx] = token
                    token_ids.append(idx)
                else:
                    # Use unknown token
                    token_ids.append(self.special_tokens['<unk>'])
        
        if add_special_tokens:
            token_ids.append(self.special_tokens['<eos>'])
            
        # Truncate if needed
        if max_length and len(token_ids) > max_length:
            token_ids = token_ids[:max_length-1] + [self.special_tokens['<eos>']]
        
        return token_ids
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs back to a LaTeX expression.
        
        Args:
            token_ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens
            
        Returns:
            Decoded LaTeX expression
        """
        tokens = []
        
        for idx in token_ids:
            if idx in self.id_to_token:
                token = self.id_to_token[idx]
                if skip_special_tokens and token in self.special_tokens.keys():
                    continue
                tokens.append(token)
            else:
                # Unknown ID
                if not skip_special_tokens:
                    tokens.append(self.id_to_token.get(self.special_tokens['<unk>'], '<unk>'))
        
        # Join tokens to form the expression
        # LaTeX commands don't need spaces, but other tokens might
        result = ''
        for token in tokens:
            # Add space between tokens that need it
            if result and not (result.endswith('\\') or token.startswith('\\')):
                result += ' '
            result += token
                
        return result
    
    def save_vocab(self, vocab_file: str) -> None:
        """
        Save the vocabulary to a file.
        
        Args:
            vocab_file: File path to save vocabulary to
        """
        with open(vocab_file, 'w') as f:
            json.dump({
                'token_to_id': self.token_to_id,
                'special_tokens': self.special_tokens
            }, f, indent=2)
    
    def load_vocab(self, vocab_file: str) -> None:
        """
        Load vocabulary from a file.
        
        Args:
            vocab_file: File path to load vocabulary from
        """
        with open(vocab_file, 'r') as f:
            data = json.load(f)
            
        self.token_to_id = data['token_to_id']
        self.id_to_token = {int(idx): token for token, idx in self.token_to_id.items()}
        self.special_tokens = data.get('special_tokens', self.special_tokens)
        self.vocab_size = len(self.token_to_id)
    
    def build_vocab_from_data(self, latex_expressions: List[str], 
                             min_freq: int = 2) -> None:
        """
        Build vocabulary from a list of LaTeX expressions.
        
        Args:
            latex_expressions: List of LaTeX expressions
            min_freq: Minimum frequency for a token to be included
        """
        # Reset token frequencies
        self.token_freqs = {}
        
        # Tokenize all expressions to count frequencies
        for expr in latex_expressions:
            self.tokenize(expr)
        
        # Sort tokens by frequency
        sorted_tokens = sorted(
            self.token_freqs.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Reset vocabulary to special tokens
        self.token_to_id = {token: idx for token, idx in self.special_tokens.items()}
        self.id_to_token = {idx: token for token, idx in self.special_tokens.items()}
        
        # Add tokens with frequency >= min_freq, up to max_vocab_size
        idx = len(self.token_to_id)
        for token, freq in sorted_tokens:
            if freq < min_freq:
                break
            if idx >= self.max_vocab_size:
                break
            if token not in self.token_to_id:
                self.token_to_id[token] = idx
                self.id_to_token[idx] = token
                idx += 1
        
        self.vocab_size = len(self.token_to_id)
        print(f"Built vocabulary with {self.vocab_size} tokens")
    
    def __len__(self) -> int:
        """Get the vocabulary size."""
        return len(self.token_to_id)