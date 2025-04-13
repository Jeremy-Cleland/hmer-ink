"""
Main model for HMER combining encoder and decoder.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class HMERModel(nn.Module):
    """
    Handwritten Mathematical Expression Recognition model.
    Combines an encoder for ink data and a decoder for LaTeX generation.
    """

    def __init__(self, encoder: nn.Module, decoder: nn.Module, config: Dict):
        """
        Initialize HMER model.

        Args:
            encoder: Encoder module for ink data
            decoder: Decoder module for LaTeX generation
            config: Model configuration
        """
        super(HMERModel, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.config = config

        # Special token indices
        self.pad_token_id = 0
        self.sos_token_id = 1
        self.eos_token_id = 2

        # Whether decoder is autoregressive or not
        if hasattr(decoder, "generate_square_subsequent_mask"):
            self.autoregressive = True
        else:
            self.autoregressive = False

    def forward(
        self,
        input_seq: torch.Tensor,
        target_seq: torch.Tensor,
        input_lengths: Optional[torch.Tensor] = None,
        target_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            input_seq: Input ink sequence [batch_size, seq_len, input_dim]
            target_seq: Target token IDs [batch_size, target_len]
            input_lengths: Lengths of input sequences
            target_lengths: Lengths of target sequences

        Returns:
            Output logits [batch_size, target_len, vocab_size]
        """
        # Encode input sequence
        memory, src_padding_mask = self.encoder(input_seq, input_lengths)

        # Prepare masks for decoder
        tgt_padding_mask = None
        if target_lengths is not None:
            # Create mask where True indicates padding position
            batch_size, tgt_len = target_seq.shape
            tgt_padding_mask = torch.arange(tgt_len, device=target_seq.device).expand(
                batch_size, tgt_len
            ) >= target_lengths.unsqueeze(1)

        # Prepare autoregressive mask if needed
        tgt_mask = None
        if self.autoregressive:
            tgt_len = target_seq.shape[1]
            tgt_mask = self.decoder.generate_square_subsequent_mask(tgt_len).to(
                target_seq.device
            )

            # Apply decoder
            logits = self.decoder(
                target_seq,
                memory,
                tgt_mask=tgt_mask,
                tgt_padding_mask=tgt_padding_mask,
                memory_key_padding_mask=src_padding_mask,
            )
        else:
            # For non-autoregressive decoders like LSTM with attention
            logits, _ = self.decoder(
                target_seq, memory, memory_key_padding_mask=src_padding_mask
            )

        return logits

    def generate(
        self,
        input_seq: torch.Tensor,
        input_lengths: Optional[torch.Tensor] = None,
        max_length: int = 128,
        beam_size: int = 4,
    ) -> Tuple[List[List[int]], List[float]]:
        """
        Generate sequences using beam search.

        Args:
            input_seq: Input ink sequence [batch_size, seq_len, input_dim]
            input_lengths: Lengths of input sequences
            max_length: Maximum sequence length to generate
            beam_size: Beam size for beam search

        Returns:
            Tuple of:
            - Generated sequences [batch_size, beam_size, max_length]
            - Sequence scores [batch_size, beam_size]
        """
        # Encode input sequence
        with torch.no_grad():
            memory, src_padding_mask = self.encoder(input_seq, input_lengths)

            # Get batch size
            batch_size = input_seq.shape[0]

            # Initialize results
            all_beams = []
            all_scores = []

            # Process each sample in batch
            for batch_idx in range(batch_size):
                # Get encoder hidden state for this sample
                mem = memory[batch_idx : batch_idx + 1]

                # Get mask for this sample
                mask = None
                if src_padding_mask is not None:
                    mask = src_padding_mask[batch_idx : batch_idx + 1]

                # Run beam search
                beams, scores = self._beam_search(mem, mask, max_length, beam_size)

                all_beams.append(beams)
                all_scores.append(scores)

            return all_beams, all_scores

    def _beam_search(
        self,
        memory: torch.Tensor,
        memory_key_padding_mask: Optional[torch.Tensor],
        max_length: int,
        beam_size: int,
    ) -> Tuple[List[List[int]], List[float]]:
        """
        Beam search algorithm for sequence generation.

        Args:
            memory: Encoder memory for single sample [1, seq_len, hidden_dim]
            memory_key_padding_mask: Memory padding mask [1, seq_len]
            max_length: Maximum sequence length
            beam_size: Beam size

        Returns:
            Tuple of:
            - Generated sequences [beam_size, seq_len]
            - Sequence scores [beam_size]
        """
        device = memory.device

        # Initialize with <sos> tokens
        current_tokens = torch.full(
            (1, 1), self.sos_token_id, dtype=torch.long, device=device
        )

        # Keep track of beams and scores
        beams = [(current_tokens, 0.0)]
        completed_beams = []

        # Expand memory for beam search
        expanded_memory = memory
        expanded_mask = memory_key_padding_mask

        # Beam search
        for step in range(max_length):
            # Break if all beams are completed
            if len(beams) == 0:
                break

            # Collect candidates from all beams
            candidates = []

            # Get number of active beams
            num_beams = len(beams)

            # Expand encoder output if needed
            if expanded_memory.shape[0] < num_beams:
                expanded_memory = memory.expand(num_beams, -1, -1)
                if expanded_mask is not None:
                    expanded_mask = memory_key_padding_mask.expand(num_beams, -1)

            # Process all active beams
            for beam_idx, (tokens, score) in enumerate(beams):
                # Skip completed beams
                if tokens[0, -1].item() == self.eos_token_id:
                    completed_beams.append((tokens, score))
                    continue

                # Apply decoder
                if self.autoregressive:
                    tgt_mask = self.decoder.generate_square_subsequent_mask(
                        tokens.shape[1]
                    ).to(device)
                    
                    # Get memory key padding mask for this beam
                    beam_memory_mask = expanded_mask[beam_idx : beam_idx + 1] if expanded_mask is not None else None
                    
                    # Use the updated interface with proper mask shape handling
                    logits = self.decoder(
                        tokens,
                        expanded_memory[beam_idx : beam_idx + 1],
                        tgt_mask=tgt_mask,
                        memory_key_padding_mask=beam_memory_mask,
                    )
                else:
                    # For non-autoregressive decoders
                    logits, _ = self.decoder(
                        tokens,
                        expanded_memory[beam_idx : beam_idx + 1],
                        memory_key_padding_mask=expanded_mask[beam_idx : beam_idx + 1]
                        if expanded_mask is not None
                        else None,
                    )

                # Get probabilities for last position
                probs = F.softmax(logits[0, -1], dim=0)

                # Get top-k candidates
                topk_probs, topk_idx = torch.topk(probs, k=beam_size)

                # Add candidates
                for i in range(beam_size):
                    token_id = topk_idx[i].item()
                    token_prob = topk_probs[i].item()

                    # Create new token sequence
                    new_tokens = torch.cat(
                        [tokens, torch.tensor([[token_id]], device=device)], dim=1
                    )

                    # Calculate new score
                    new_score = score - np.log(token_prob)

                    candidates.append((new_tokens, new_score, token_id))

            # Sort candidates by score
            candidates.sort(key=lambda x: x[1])

            # Select top-k candidates
            beams = []
            for tokens, score, token_id in candidates[:beam_size]:
                # Check if sequence is completed
                if token_id == self.eos_token_id:
                    completed_beams.append((tokens, score))
                else:
                    beams.append((tokens, score))

                # Break if we have enough beams
                if len(beams) + len(completed_beams) >= beam_size:
                    break

        # Add incomplete beams to completed
        completed_beams.extend(beams)

        # Sort completed beams by score
        completed_beams.sort(key=lambda x: x[1])

        # Extract token sequences and scores
        beam_sequences = []
        beam_scores = []

        for tokens, score in completed_beams[:beam_size]:
            # Convert to list of integers
            token_ids = tokens.squeeze(0).tolist()
            beam_sequences.append(token_ids)
            beam_scores.append(score)

        # Pad results to beam_size
        while len(beam_sequences) < beam_size:
            beam_sequences.append([self.sos_token_id, self.eos_token_id])
            beam_scores.append(float("inf"))

        return beam_sequences, beam_scores

    def save_checkpoint(
        self,
        path: str,
        epoch: int,
        optimizer: Any = None,
        scheduler: Any = None,
        best_metric: float = None,
    ):
        """
        Save model checkpoint.

        Args:
            path: Path to save checkpoint
            epoch: Current epoch
            optimizer: Optimizer state
            scheduler: Scheduler state
            best_metric: Best validation metric
        """
        checkpoint = {"model": self.state_dict(), "config": self.config, "epoch": epoch}

        if optimizer is not None:
            checkpoint["optimizer"] = optimizer.state_dict()

        if scheduler is not None:
            checkpoint["scheduler"] = scheduler.state_dict()

        if best_metric is not None:
            checkpoint["best_metric"] = best_metric

        torch.save(checkpoint, path)

    @classmethod
    def load_checkpoint(cls, path: str, map_location: str = None):
        """
        Load model from checkpoint.

        Args:
            path: Path to load checkpoint from
            map_location: Device mapping

        Returns:
            Tuple of (model, checkpoint_dict)
        """
        from . import get_decoder, get_encoder

        checkpoint = torch.load(path, map_location=map_location)

        # Get configuration
        config = checkpoint.get("config", {})

        # Create encoder and decoder
        encoder = get_encoder(config)

        # Estimate vocab size from model weights
        output_layer_key = "decoder.output_projection.weight"
        if output_layer_key in checkpoint["model"]:
            vocab_size = checkpoint["model"][output_layer_key].shape[0]
        else:
            # Default fallback
            vocab_size = 1000

        decoder = get_decoder(config, vocab_size)

        # Create model
        model = cls(encoder, decoder, config)

        # Load state dict
        model.load_state_dict(checkpoint["model"])

        return model, checkpoint
