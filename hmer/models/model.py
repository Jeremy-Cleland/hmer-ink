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
        fast_mode: bool = True,
    ) -> Tuple[List[List[List[int]]], List[List[float]]]:
        """
        Generate sequences using beam search.

        Args:
            input_seq: Input ink sequence [batch_size, seq_len, input_dim]
            input_lengths: Lengths of input sequences
            max_length: Maximum sequence length to generate
            beam_size: Beam size for beam search
            fast_mode: Whether to use faster generation mode (recommended for validation)

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
            device = input_seq.device

            # If fast mode is enabled and we're using a small beam size, use a faster greedy search
            if fast_mode and beam_size <= 2:
                # For beam size 1 (greedy search) or 2 (simple beam) with small batch
                return self._fast_generate(
                    memory, src_padding_mask, max_length, beam_size, batch_size, device
                )

            # Otherwise, use standard beam search processing sample by sample
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

    def _fast_generate(
        self,
        memory: torch.Tensor,
        memory_padding_mask: Optional[torch.Tensor],
        max_length: int,
        beam_size: int,
        batch_size: int,
        device: torch.device,
    ) -> Tuple[List[List[List[int]]], List[List[float]]]:
        """
        Fast beam search implementation optimized for validation during training.

        Args:
            memory: Encoder outputs [batch_size, seq_len, dim]
            memory_padding_mask: Encoder padding mask
            max_length: Maximum sequence length to generate
            beam_size: Beam size (1 for greedy search, 2 for simple beam)
            batch_size: Batch size
            device: Device for computation

        Returns:
            Tuple of:
            - Generated sequences [batch_size, beam_size, seq_len]
            - Sequence scores [batch_size, beam_size]
        """
        # For simplicity, we'll implement a faster greedy search
        # when beam_size=1 and a simplified beam search when beam_size=2

        # Verify and adjust batch size if memory doesn't match
        actual_batch_size = memory.size(0)
        if actual_batch_size != batch_size:
            # Use the smaller batch size to avoid index errors
            batch_size = min(batch_size, actual_batch_size)

        # Initialize with <sos> tokens for all examples in batch
        sequences = torch.full(
            (batch_size, 1), self.sos_token_id, dtype=torch.long, device=device
        )

        # Keep track of finished sequences
        is_finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        # Storage for final results - always use the actual_batch_size for results
        all_beams = [[] for _ in range(actual_batch_size)]
        all_scores = [[] for _ in range(actual_batch_size)]

        # For beam search with beam_size=2
        if beam_size == 2:
            beam_sequences = [[] for _ in range(actual_batch_size)]
            beam_scores = [[] for _ in range(actual_batch_size)]

        # Generate tokens up to max_length
        for step in range(max_length):
            # If all sequences are finished, break
            if is_finished.all():
                break

            # Prepare mask for autoregressive decoding
            if self.autoregressive:
                tgt_mask = self.decoder.generate_square_subsequent_mask(
                    sequences.size(1)
                ).to(device)

                # Get decoder outputs - handle smaller batches to avoid memory issues
                try:
                    decoder_outputs = self.decoder(
                        sequences,
                        memory,
                        tgt_mask=tgt_mask,
                        memory_key_padding_mask=memory_padding_mask,
                    )
                except RuntimeError as e:
                    if "shape" in str(e) or "size" in str(e):
                        # Memory issue - fallback to sequential processing
                        all_outputs = []
                        for i in range(sequences.size(0)):
                            seq_i = sequences[i : i + 1]
                            mem_i = memory[i : i + 1] if memory.dim() > 2 else memory
                            mask_i = (
                                memory_padding_mask[i : i + 1]
                                if memory_padding_mask is not None
                                else None
                            )

                            out_i = self.decoder(
                                seq_i,
                                mem_i,
                                tgt_mask=tgt_mask,
                                memory_key_padding_mask=mask_i,
                            )
                            all_outputs.append(out_i)

                        decoder_outputs = torch.cat(all_outputs, dim=0)
                    else:
                        # Not a shape-related error, re-raise
                        raise
            else:
                # For non-autoregressive decoders
                try:
                    decoder_outputs, _ = self.decoder(
                        sequences, memory, memory_key_padding_mask=memory_padding_mask
                    )
                except RuntimeError as e:
                    if "shape" in str(e) or "size" in str(e):
                        # Memory issue - fallback to sequential processing
                        all_outputs = []
                        for i in range(sequences.size(0)):
                            seq_i = sequences[i : i + 1]
                            mem_i = memory[i : i + 1] if memory.dim() > 2 else memory
                            mask_i = (
                                memory_padding_mask[i : i + 1]
                                if memory_padding_mask is not None
                                else None
                            )

                            out_i, _ = self.decoder(
                                seq_i, mem_i, memory_key_padding_mask=mask_i
                            )
                            all_outputs.append(out_i)

                        decoder_outputs = torch.cat(all_outputs, dim=0)
                    else:
                        # Not a shape-related error, re-raise
                        raise

            # Get probabilities for next token (last position)
            logits = decoder_outputs[:, -1, :]
            probs = F.log_softmax(logits, dim=-1)

            if beam_size == 1:
                # Greedy search - just take the most likely token
                next_token_logprobs, next_tokens = torch.max(probs, dim=-1)

                # Add chosen tokens to sequences where not finished
                for i in range(batch_size):
                    if not is_finished[i]:
                        token = next_tokens[i].item()

                        # Check if sequence is finished
                        if token == self.eos_token_id:
                            is_finished[i] = True
                            # Store this completed sequence
                            all_beams[i] = [sequences[i].tolist()]
                            all_scores[i] = [
                                0.0
                            ]  # Simplified score for fast validation

                # Prepare next iteration - add tokens to sequences that aren't finished
                if not is_finished.all():
                    # Create mask for unfinished sequences
                    active_mask = ~is_finished

                    # Only update unfinished sequences
                    active_next_tokens = next_tokens[active_mask].unsqueeze(1)
                    sequences = torch.cat(
                        [sequences, torch.zeros_like(active_next_tokens)], dim=1
                    )
                    sequences[active_mask, -1] = next_tokens[active_mask]

            else:  # beam_size == 2
                # Simple beam search with 2 candidates
                next_token_logprobs, next_tokens = torch.topk(probs, k=2, dim=-1)

                # For each example in batch
                new_sequences = []
                for i in range(batch_size):
                    if is_finished[i]:
                        # Keep the same sequence for finished examples
                        new_sequences.append(sequences[i : i + 1])
                        continue

                    # Validate indices are in bounds
                    if i >= next_tokens.size(0) or i >= actual_batch_size:
                        # Handle out of bounds - just keep current sequence and mark as finished
                        new_sequences.append(
                            sequences[
                                min(i, sequences.size(0) - 1) : min(
                                    i, sequences.size(0) - 1
                                )
                                + 1
                            ]
                        )
                        is_finished[i] = True
                        if i < actual_batch_size:
                            all_beams[i] = [
                                sequences[min(i, sequences.size(0) - 1)].tolist()
                            ]
                            all_scores[i] = [0.0]
                        continue

                    # Get 2 best next tokens for this example
                    for j in range(min(2, next_tokens.size(1))):
                        token = next_tokens[i, j].item()
                        token_logprob = next_token_logprobs[i, j].item()

                        # Create new sequence with this token
                        new_seq = torch.cat(
                            [
                                sequences[i : i + 1],
                                torch.tensor([[token]], device=device),
                            ],
                            dim=1,
                        )

                        # Add to beam candidates
                        beam_sequences[i].append(new_seq[0].tolist())
                        beam_scores[i].append(
                            -token_logprob
                        )  # Negative log prob as score

                        # If EOS token and best candidate, mark as finished
                        if token == self.eos_token_id and j == 0:
                            is_finished[i] = True
                            # Store completed sequence
                            all_beams[i] = [new_seq[0].tolist()]
                            all_scores[i] = [-token_logprob]
                            break

                    # If not finished yet, continue with best sequence
                    if not is_finished[i]:
                        # Ensure we have beam candidates before accessing
                        if beam_sequences[i]:
                            new_sequences.append(
                                torch.tensor([beam_sequences[i][0]], device=device)
                            )
                        else:
                            # Fallback if no beam candidates
                            new_sequences.append(sequences[i : i + 1])
                            is_finished[i] = True

                # Update sequences for next iteration
                if new_sequences:
                    try:
                        sequences = torch.cat(new_sequences, dim=0)
                    except RuntimeError:
                        # Handle case where shapes aren't compatible
                        # This could happen if sequence lengths vary
                        sequences = torch.zeros(
                            (batch_size, sequences.size(1) + 1),
                            dtype=torch.long,
                            device=device,
                        )
                        for idx, seq in enumerate(new_sequences):
                            if idx < batch_size:
                                seq_len = min(seq.size(1), sequences.size(1))
                                sequences[idx, :seq_len] = seq[0, :seq_len]
                        # Mark all as finished to avoid further issues
                        is_finished = torch.ones(
                            batch_size, dtype=torch.bool, device=device
                        )

        # Handle any unfinished sequences
        for i in range(actual_batch_size):
            if not all_beams[i]:
                if i < sequences.size(0):
                    all_beams[i] = [sequences[i].tolist()]
                else:
                    # Handle case where i is out of bounds
                    all_beams[i] = [[self.sos_token_id, self.eos_token_id]]
                all_scores[i] = [0.0]

            # For beam_size=2, fill second beam with best alternative
            if beam_size == 2:
                if (
                    len(all_beams[i]) < 2
                    and i < len(beam_sequences)
                    and beam_sequences[i]
                ):
                    # Add the best alternative sequence
                    all_beams[i].append(
                        beam_sequences[i][1]
                        if len(beam_sequences[i]) > 1
                        else all_beams[i][0]
                    )
                    all_scores[i].append(
                        beam_scores[i][1]
                        if len(beam_scores[i]) > 1
                        else all_scores[i][0]
                    )

        # Ensure all examples have exactly beam_size results
        for i in range(actual_batch_size):
            # Duplicate the first beam if needed to reach beam_size
            if all_beams[i]:  # Check if there's at least one beam
                while len(all_beams[i]) < beam_size:
                    all_beams[i].append(all_beams[i][0])
                    all_scores[i].append(all_scores[i][0])
            else:  # Handle empty beam case
                all_beams[i] = [[self.sos_token_id, self.eos_token_id]] * beam_size
                all_scores[i] = [0.0] * beam_size

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
                    beam_memory_mask = (
                        expanded_mask[beam_idx : beam_idx + 1]
                        if expanded_mask is not None
                        else None
                    )

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
