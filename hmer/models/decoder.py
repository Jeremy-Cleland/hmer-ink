"""
Decoder models for HMER-Ink.
"""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerDecoder(nn.Module):
    """
    Transformer decoder for generating LaTeX expressions from encoded ink.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_length: int = 128,
        padding_idx: int = 0,
    ):
        """
        Initialize transformer decoder.

        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of embeddings
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            dropout: Dropout probability
            max_length: Maximum sequence length
            padding_idx: Index of padding token
        """
        super(TransformerDecoder, self).__init__()

        # Token embedding
        self.token_embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=padding_idx
        )

        # Positional encoding
        self.pos_encoder = PositionalEncoding(embedding_dim, max_length, dropout)

        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=4 * embedding_dim,
            dropout=dropout,
            activation="relu",
            batch_first=True,
        )

        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_layers
        )

        # Output projection to vocabulary
        self.output_projection = nn.Linear(embedding_dim, vocab_size)

        # Store parameters
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        tgt_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Decode sequences.

        Args:
            tgt: Target sequence [batch_size, tgt_len]
            memory: Encoder output [batch_size, src_len, embedding_dim]
            tgt_mask: Target sequence mask to prevent attending to future positions
            tgt_padding_mask: Target padding mask [batch_size, tgt_len]
            memory_key_padding_mask: Memory padding mask [batch_size, src_len]

        Returns:
            Output logits [batch_size, tgt_len, vocab_size]
        """
        # Embed tokens
        tgt_embedded = self.token_embedding(tgt)

        # Add positional encoding
        tgt_embedded = self.pos_encoder(tgt_embedded)

        # Convert boolean mask to float mask to avoid type mismatches
        # This fixes the "mismatched key_padding_mask and attn_mask" warning
        memory_mask = None
        if memory_key_padding_mask is not None:
            # Use memory mask instead of memory_key_padding_mask
            if memory_key_padding_mask.dtype == torch.bool:
                # Create a cross-attention mask with shape [tgt_len, src_len]
                # For Transformer, the memory mask should have shape [tgt_len, src_len]
                tgt_len = tgt.size(1)
                src_len = memory.size(1)

                # First expand the padding mask to shape [batch_size, tgt_len, src_len]
                expanded_mask = memory_key_padding_mask.unsqueeze(1).expand(
                    -1, tgt_len, -1
                )

                # Then create a mask of shape [tgt_len, src_len] where each position
                # is masked if ANY batch element is masked at that position
                memory_mask = torch.zeros(
                    (tgt_len, src_len),
                    device=memory_key_padding_mask.device,
                    dtype=torch.float,
                )

                # If any batch element has padding at a source position, mask it for all batches
                # We first collapse the batch dimension with a logical OR
                combined_mask = expanded_mask.any(dim=0)
                memory_mask.masked_fill_(combined_mask, float("-inf"))
            else:
                # If the mask is already a float mask, reshape it to [tgt_len, src_len]
                tgt_len = tgt.size(1)
                src_len = memory.size(1)
                memory_mask = memory_key_padding_mask.view(-1, src_len).mean(dim=0)
                memory_mask = memory_mask.expand(tgt_len, src_len)

        # Apply transformer decoder using memory mask instead of key_padding_mask
        output = self.transformer_decoder(
            tgt_embedded,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_mask=memory_mask,
        )

        # Project to vocabulary
        logits = self.output_projection(output)

        return logits

    def generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """
        Generate a square mask for the sequence.
        The mask ensures that the prediction for position i
        can depend only on known elements in positions < i.

        Args:
            sz: Size of square matrix

        Returns:
            Mask tensor with ones on diagonal and upper triangle, zeros elsewhere
        """
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float("-inf"))
        return mask


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer models.
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        """
        Initialize positional encoding.

        Args:
            d_model: Dimension of the model
            max_len: Maximum sequence length
            dropout: Dropout probability
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute positional encoding once at initialization
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        # Register buffer (not a parameter, but part of the module)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.

        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]

        Returns:
            Output tensor with positional encoding added
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class LSTMDecoder(nn.Module):
    """
    LSTM decoder for generating LaTeX expressions from encoded ink.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.1,
        padding_idx: int = 0,
    ):
        """
        Initialize LSTM decoder.

        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of embeddings
            hidden_dim: Dimension of LSTM hidden state
            num_layers: Number of LSTM layers
            dropout: Dropout probability
            padding_idx: Index of padding token
        """
        super(LSTMDecoder, self).__init__()

        # Token embedding
        self.token_embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=padding_idx
        )

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )

        # Attention mechanism
        self.attention = BahdanauAttention(hidden_dim, embedding_dim)

        # Output projection
        self.output_projection = nn.Linear(hidden_dim + embedding_dim, vocab_size)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Store parameters
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Decode sequences.

        Args:
            tgt: Target sequence [batch_size, tgt_len]
            memory: Encoder output [batch_size, src_len, embedding_dim]
            hidden: Initial hidden state for LSTM
            memory_key_padding_mask: Memory padding mask [batch_size, src_len]

        Returns:
            Tuple of:
            - Output logits [batch_size, tgt_len, vocab_size]
            - Final hidden state
        """
        batch_size, tgt_len = tgt.shape

        # Embed tokens
        tgt_embedded = self.token_embedding(tgt)
        tgt_embedded = self.dropout(tgt_embedded)

        # Initialize hidden state if not provided
        if hidden is None:
            # Use encoder memory to initialize hidden state
            enc_hidden = torch.mean(memory, dim=1)
            h_0 = torch.zeros(
                self.lstm.num_layers, batch_size, self.hidden_dim, device=tgt.device
            )
            c_0 = torch.zeros(
                self.lstm.num_layers, batch_size, self.hidden_dim, device=tgt.device
            )

            # Copy encoder summary to all layers
            for i in range(self.lstm.num_layers):
                h_0[i] = enc_hidden

            hidden = (h_0, c_0)

        # Apply LSTM and get all outputs
        lstm_output, hidden = self.lstm(tgt_embedded, hidden)

        # Apply attention for each time step
        context_vectors = []
        for t in range(tgt_len):
            # Get hidden state for this step
            # Use only the top layer
            decoder_hidden = lstm_output[:, t : t + 1, :]

            # Apply attention
            context, _ = self.attention(decoder_hidden, memory, memory_key_padding_mask)

            context_vectors.append(context)

        # Concatenate all context vectors
        context = torch.cat(context_vectors, dim=1)

        # Concatenate LSTM output and context vector
        output = torch.cat([lstm_output, context], dim=2)

        # Project to vocabulary
        logits = self.output_projection(output)

        return logits, hidden


class BahdanauAttention(nn.Module):
    """
    Bahdanau (Additive) Attention mechanism.
    """

    def __init__(self, hidden_dim: int, enc_dim: int):
        """
        Initialize attention mechanism.

        Args:
            hidden_dim: Dimension of decoder hidden state
            enc_dim: Dimension of encoder output
        """
        super(BahdanauAttention, self).__init__()

        # Attention layers
        self.W_h = nn.Linear(enc_dim, hidden_dim)
        self.W_s = nn.Linear(hidden_dim, hidden_dim)
        self.V = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        hidden: torch.Tensor,
        encoder_outputs: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute attention weights and context vector.

        Args:
            hidden: Decoder hidden state [batch_size, 1, hidden_dim]
            encoder_outputs: Encoder outputs [batch_size, src_len, enc_dim]
            mask: Attention mask [batch_size, src_len]

        Returns:
            Tuple of:
            - Context vector [batch_size, 1, enc_dim]
            - Attention weights [batch_size, 1, src_len]
        """
        batch_size, src_len, enc_dim = encoder_outputs.shape

        # Calculate attention scores
        encoder_features = self.W_h(
            encoder_outputs
        )  # [batch_size, src_len, hidden_dim]
        decoder_features = self.W_s(hidden)  # [batch_size, 1, hidden_dim]

        # Expand decoder features to match encoder dimensions
        # decoder_features = decoder_features.expand(-1, src_len, -1)

        # Calculate attention score for each encoder position
        # e_ij = V(tanh(W_h * h_j + W_s * s_i))
        energy = self.V(
            torch.tanh(encoder_features + decoder_features)
        )  # [batch_size, src_len, 1]
        energy = energy.squeeze(2)  # [batch_size, src_len]

        # Apply mask if provided
        if mask is not None:
            energy = energy.masked_fill(mask, -1e9)

        # Calculate attention weights
        attention_weights = F.softmax(energy, dim=1)
        attention_weights = attention_weights.unsqueeze(1)  # [batch_size, 1, src_len]

        # Calculate context vector
        context = torch.bmm(
            attention_weights, encoder_outputs
        )  # [batch_size, 1, enc_dim]

        return context, attention_weights


def get_decoder(config: Dict, vocab_size: int) -> nn.Module:
    """
    Factory function to create a decoder based on configuration.

    Args:
        config: Configuration dictionary
        vocab_size: Size of vocabulary

    Returns:
        Decoder module
    """
    decoder_config = config["decoder"]
    decoder_type = decoder_config.get("type", "transformer")

    embedding_dim = decoder_config.get("embedding_dim", 256)
    num_layers = decoder_config.get("num_layers", 4)
    num_heads = decoder_config.get("num_heads", 8)
    dropout = decoder_config.get("dropout", 0.1)
    padding_idx = 0  # Assuming 0 is padding index

    if decoder_type == "transformer":
        max_length = decoder_config.get("max_length", 128)
        return TransformerDecoder(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            max_length=max_length,
            padding_idx=padding_idx,
        )
    elif decoder_type == "lstm":
        hidden_dim = decoder_config.get("hidden_dim", embedding_dim * 2)
        return LSTMDecoder(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            padding_idx=padding_idx,
        )
    else:
        raise ValueError(f"Unknown decoder type: {decoder_type}")
