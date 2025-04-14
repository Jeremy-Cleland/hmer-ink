"""
Test script to validate the attention mask fix in the transformer decoder.
"""

import os
import sys
import torch
# Unused imports
# import torch.nn as nn
# import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from hmer.models.decoder import TransformerDecoder


def test_attention_mask():
    """
    Test the attention mask fix in the transformer decoder.
    """
    # Create a simple decoder instance
    vocab_size = 1000
    embedding_dim = 256
    num_heads = 8
    batch_size = 16
    tgt_len = 45
    src_len = 1024

    # Initialize decoder
    decoder = TransformerDecoder(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        num_layers=2,
        num_heads=num_heads,
    )

    print("Testing transformer decoder attention mask...")

    # Generate random inputs
    print("Generating random test inputs...")
    tgt = torch.randint(0, vocab_size, (batch_size, tgt_len))
    memory = torch.randn(batch_size, src_len, embedding_dim)

    # Create a random boolean padding mask
    memory_key_padding_mask = torch.randint(0, 2, (batch_size, src_len)).bool()

    # Create causal attention mask
    tgt_mask = decoder.generate_square_subsequent_mask(tgt_len)

    try:
        print("Testing decoder forward pass...")
        # Forward pass with the masks
        output = decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )

        print(f"Success! Output shape: {output.shape}")
        print("The transformer decoder attention mask fix is working correctly.")
    except Exception as e:
        print(f"Error: {e}")
        print("The transformer decoder attention mask fix is not working correctly.")
        return False

    return True


if __name__ == "__main__":
    success = test_attention_mask()
    sys.exit(0 if success else 1)
