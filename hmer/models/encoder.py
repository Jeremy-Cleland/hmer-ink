"""
Encoder models for HMER-Ink.
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class StrokeEmbedding(nn.Module):
    """
    Embedding layer for ink strokes.
    """

    def __init__(self, input_dim: int, embedding_dim: int, dropout: float = 0.1):
        """
        Initialize stroke embedding.

        Args:
            input_dim: Number of input features (e.g., x, y, t)
            embedding_dim: Dimension of the embedding
            dropout: Dropout probability
        """
        super(StrokeEmbedding, self).__init__()

        # Linear projection to embedding dimension
        self.embed = nn.Linear(input_dim, embedding_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Embed ink strokes.

        Args:
            x: Input tensor of shape [batch_size, seq_len, input_dim]

        Returns:
            Embedded tensor of shape [batch_size, seq_len, embedding_dim]
        """
        x = self.embed(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.norm(x)
        return x


class TransformerEncoder(nn.Module):
    """
    Transformer encoder for ink strokes.
    """

    def __init__(
        self,
        input_dim: int,
        embedding_dim: int,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_length: int = 1024,
        use_bbox_data: bool = False,
    ):
        """
        Initialize transformer encoder.

        Args:
            input_dim: Number of input features
            embedding_dim: Dimension of the embedding
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            dropout: Dropout probability
            max_length: Maximum sequence length for positional encoding
            use_bbox_data: Whether to use bounding box data
        """
        super(TransformerEncoder, self).__init__()

        # Flag to indicate this encoder can use bounding box data
        self.use_bbox_data = use_bbox_data

        # Stroke embedding
        self.embedding = StrokeEmbedding(input_dim, embedding_dim, dropout)

        # Positional encoding
        self.positional_encoding = PositionalEncoding(
            embedding_dim, max_length, dropout
        )

        # Bounding box embedding if enabled
        if use_bbox_data:
            # 4 values for bbox: x_min, y_min, x_max, y_max
            self.bbox_embedding = nn.Linear(4, embedding_dim)
            self.bbox_norm = nn.LayerNorm(embedding_dim)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=4 * embedding_dim,
            dropout=dropout,
            activation="relu",
            batch_first=True,
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        bbox_data: Optional[List] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode ink strokes.

        Args:
            x: Input tensor of shape [batch_size, seq_len, input_dim]
            lengths: Sequence lengths for masking
            bbox_data: Optional bounding box data for enhancing spatial understanding

        Returns:
            Tuple of:
            - Encoded representation [batch_size, seq_len, embedding_dim]
            - Padding mask [batch_size, seq_len]
        """
        batch_size, seq_len = x.shape[0], x.shape[1]

        # Create padding mask if sequence lengths are provided
        padding_mask = None
        if lengths is not None:
            # Create mask where 1 indicates padding position
            padding_mask = torch.arange(seq_len, device=x.device).expand(
                batch_size, seq_len
            ) >= lengths.unsqueeze(1)

        # Embed strokes
        x = self.embedding(x)

        # Add positional encoding
        x = self.positional_encoding(x)

        # Enhance with bounding box data if available and enabled
        if self.use_bbox_data and bbox_data is not None:
            for batch_idx, sample_bbox_data in enumerate(bbox_data):
                if not sample_bbox_data:
                    continue

                # Process each bounding box
                for bbox in sample_bbox_data:
                    # Extract bbox coordinates and normalize to [-1, 1] range
                    # Handle both naming conventions (snake_case or camelCase)
                    x_min = bbox.get("x_min", bbox.get("xMin", 0))
                    y_min = bbox.get("y_min", bbox.get("yMin", 0))
                    x_max = bbox.get("x_max", bbox.get("xMax", 1))
                    y_max = bbox.get("y_max", bbox.get("yMax", 1))

                    # Normalize coordinates if they're not already in the expected range
                    if (
                        abs(x_min) > 10
                        or abs(y_min) > 10
                        or abs(x_max) > 10
                        or abs(y_max) > 10
                    ):
                        # Assume these are unnormalized coordinates, normalize them to [-1, 1]
                        max_coord = max(abs(x_min), abs(y_min), abs(x_max), abs(y_max))
                        x_min = x_min / max_coord
                        y_min = y_min / max_coord
                        x_max = x_max / max_coord
                        y_max = y_max / max_coord

                    bbox_coords = torch.tensor(
                        [x_min, y_min, x_max, y_max],
                        dtype=torch.float32,
                        device=x.device,
                    )

                    # Convert bbox to embedding
                    bbox_embed = self.bbox_embedding(bbox_coords.view(1, -1))
                    bbox_embed = F.relu(bbox_embed)
                    bbox_embed = self.bbox_norm(bbox_embed)

                    # Check if we have point indices or token information
                    if "point_indices" in bbox:
                        # Use provided point indices
                        point_indices = bbox["point_indices"]
                        for idx in point_indices:
                            if idx < seq_len:
                                # Add bbox information to the point features
                                x[batch_idx, idx] = x[
                                    batch_idx, idx
                                ] + bbox_embed.squeeze(0)
                    elif "token" in bbox:
                        # For token-based bboxes, apply to a segment of the sequence
                        # This is a heuristic approach since we don't have explicit point mappings
                        segment_size = seq_len // 10  # Approximate segment size
                        start_idx = min(
                            int((x_min + 1) * seq_len / 2), seq_len - segment_size
                        )
                        end_idx = min(start_idx + segment_size, seq_len)

                        # Apply bbox embedding to this segment
                        for idx in range(start_idx, end_idx):
                            x[batch_idx, idx] = x[batch_idx, idx] + bbox_embed.squeeze(
                                0
                            )

        # Apply transformer encoder
        # Note: In PyTorch transformer, True in mask means to ignore that position
        encoded = self.transformer_encoder(x, src_key_padding_mask=padding_mask)

        return encoded, padding_mask


class BiLSTMEncoder(nn.Module):
    """
    Bidirectional LSTM encoder for ink strokes.
    """

    def __init__(
        self,
        input_dim: int,
        embedding_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        """
        Initialize BiLSTM encoder.

        Args:
            input_dim: Number of input features
            embedding_dim: Dimension of the embedding
            hidden_dim: Dimension of LSTM hidden state
            num_layers: Number of LSTM layers
            dropout: Dropout probability
        """
        super(BiLSTMEncoder, self).__init__()

        # Stroke embedding
        self.embedding = StrokeEmbedding(input_dim, embedding_dim, dropout)

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )

        # Output projection to match embedding dimension
        self.projection = nn.Linear(hidden_dim * 2, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embedding_dim)

    def forward(
        self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode ink strokes.

        Args:
            x: Input tensor of shape [batch_size, seq_len, input_dim]
            lengths: Sequence lengths for packing

        Returns:
            Tuple of:
            - Encoded representation [batch_size, seq_len, embedding_dim]
            - Padding mask [batch_size, seq_len]
        """
        batch_size, seq_len = x.shape[0], x.shape[1]

        # Create padding mask if sequence lengths are provided
        padding_mask = None
        if lengths is not None:
            padding_mask = torch.arange(seq_len, device=x.device).expand(
                batch_size, seq_len
            ) >= lengths.unsqueeze(1)

        # Embed strokes
        x = self.embedding(x)

        # Pack sequence for LSTM if lengths are provided
        if lengths is not None:
            packed_x = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )

            # Apply LSTM
            packed_output, _ = self.lstm(packed_x)

            # Unpack output
            output, _ = nn.utils.rnn.pad_packed_sequence(
                packed_output, batch_first=True
            )

            # Ensure correct output size
            if output.size(1) < seq_len:
                padding = torch.zeros(
                    batch_size,
                    seq_len - output.size(1),
                    output.size(2),
                    device=output.device,
                )
                output = torch.cat([output, padding], dim=1)
        else:
            # Apply LSTM without packing
            output, _ = self.lstm(x)

        # Project to embedding dimension
        output = self.projection(output)
        output = self.dropout(output)
        output = self.layer_norm(output)

        return output, padding_mask


class CNNEncoder(nn.Module):
    """
    Convolutional encoder for ink strokes.
    """

    def __init__(
        self,
        input_dim: int,
        embedding_dim: int,
        kernel_sizes: List[int] = [3, 5, 7],
        num_filters: int = 128,
        dropout: float = 0.1,
    ):
        """
        Initialize CNN encoder.

        Args:
            input_dim: Number of input features
            embedding_dim: Dimension of the embedding
            kernel_sizes: List of kernel sizes for convolutions
            num_filters: Number of filters per convolution
            dropout: Dropout probability
        """
        super(CNNEncoder, self).__init__()

        # Stroke embedding
        self.embedding = StrokeEmbedding(input_dim, embedding_dim, dropout)

        # Convolutional layers with different kernel sizes
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(embedding_dim, num_filters, k, padding=k // 2)
                for k in kernel_sizes
            ]
        )

        # Output projection
        total_filters = num_filters * len(kernel_sizes)
        self.projection = nn.Linear(total_filters, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embedding_dim)

    def forward(
        self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode ink strokes.

        Args:
            x: Input tensor of shape [batch_size, seq_len, input_dim]
            lengths: Sequence lengths for masking

        Returns:
            Tuple of:
            - Encoded representation [batch_size, seq_len, embedding_dim]
            - Padding mask [batch_size, seq_len]
        """
        batch_size, seq_len = x.shape[0], x.shape[1]

        # Create padding mask if sequence lengths are provided
        padding_mask = None
        if lengths is not None:
            padding_mask = torch.arange(seq_len, device=x.device).expand(
                batch_size, seq_len
            ) >= lengths.unsqueeze(1)

        # Embed strokes
        x = self.embedding(x)

        # Apply convolutions
        # Transpose to [batch_size, embedding_dim, seq_len] for convolutions
        x = x.transpose(1, 2)

        # Apply each convolution and activate
        conv_outputs = []
        for conv in self.convs:
            conv_output = F.relu(conv(x))
            conv_outputs.append(conv_output)

        # Concatenate along filter dimension
        combined = torch.cat(conv_outputs, dim=1)

        # Transpose back to [batch_size, seq_len, filters]
        combined = combined.transpose(1, 2)

        # Project to embedding dimension
        output = self.projection(combined)
        output = self.dropout(output)
        output = self.layer_norm(output)

        return output, padding_mask


def get_encoder(config: Dict) -> nn.Module:
    """
    Factory function to create an encoder based on configuration.

    Args:
        config: Configuration dictionary

    Returns:
        Encoder module
    """
    encoder_config = config["encoder"]
    encoder_type = encoder_config.get("type", "transformer")

    input_dim = encoder_config.get("input_dim", 3)
    embedding_dim = encoder_config.get("embedding_dim", 256)
    num_layers = encoder_config.get("num_layers", 4)
    num_heads = encoder_config.get("num_heads", 8)
    dropout = encoder_config.get("dropout", 0.1)

    # Check if bounding box data should be used
    use_bbox_data = encoder_config.get("use_bbox_data", False)

    if encoder_type == "transformer":
        return TransformerEncoder(
            input_dim=input_dim,
            embedding_dim=embedding_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            use_bbox_data=use_bbox_data,
        )
    elif encoder_type == "bilstm":
        hidden_dim = encoder_config.get("hidden_dim", embedding_dim)
        return BiLSTMEncoder(
            input_dim=input_dim,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
    elif encoder_type == "cnn":
        kernel_sizes = encoder_config.get("kernel_sizes", [3, 5, 7])
        num_filters = encoder_config.get("num_filters", 128)
        return CNNEncoder(
            input_dim=input_dim,
            embedding_dim=embedding_dim,
            kernel_sizes=kernel_sizes,
            num_filters=num_filters,
            dropout=dropout,
        )
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")
