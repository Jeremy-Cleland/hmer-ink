"""
Curriculum learning strategies for HMER-Ink datasets.

This module provides curriculum learning implementations that allow
batching samples by difficulty (easy to hard) or by length (short to long).
"""

import math
import os
import random
from typing import Dict, List

import torch
from torch.utils.data import Sampler

from .dataset import HMERDataset


class CurriculumSampler(Sampler):
    """
    Curriculum learning sampler that orders samples by difficulty.

    Difficulty can be measured by:
    1. Expression length (token count)
    2. Stroke count
    3. Sequence length
    4. Complexity score (if available)
    """

    def __init__(
        self,
        dataset: HMERDataset,
        difficulty_metric: str = "token_length",
        epochs_to_full_difficulty: int = 10,
        current_epoch: int = 0,
        shuffle: bool = True,
        seed: int = 42,
    ):
        """
        Initialize curriculum sampler.

        Args:
            dataset: HMER dataset to sample from
            difficulty_metric: Metric to use for difficulty ("token_length", "seq_length")
            epochs_to_full_difficulty: Number of epochs until full difficulty range is used
            current_epoch: Current epoch number (starts at 0)
            shuffle: Whether to shuffle samples within the same difficulty level
            seed: Random seed for reproducibility
        """
        self.dataset = dataset
        self.difficulty_metric = difficulty_metric
        self.epochs_to_full_difficulty = max(1, epochs_to_full_difficulty)
        self.current_epoch = current_epoch
        self.shuffle = shuffle
        self.seed = seed
        self.rng = random.Random(seed)

        # Calculate difficulties for all samples
        self.difficulties = self._calculate_difficulties()

        # Sort dataset indices by difficulty
        self.sorted_indices = self._sort_indices_by_difficulty()

        # Calculate current curriculum percentile based on epoch
        self.curr_percentile = self._get_current_percentile()

    def _calculate_difficulties(self) -> List[float]:
        """Calculate difficulty scores for all samples in the dataset."""
        difficulties = []

        for idx in range(len(self.dataset)):
            difficulty = self._get_sample_difficulty(idx)
            difficulties.append(difficulty)

        return difficulties

    def _get_sample_difficulty(self, idx: int) -> float:
        """
        Calculate difficulty for a single sample using various metrics.
        
        More sophisticated difficulty metrics provide better curriculum pacing.
        """
        # Get sample without loading full data to improve efficiency
        file_path = self.dataset.file_paths[idx]
        ink_data = self.dataset.parser.parse_inkml(file_path)
        label = ink_data.get("normalized_label", ink_data.get("label", ""))
        
        if self.difficulty_metric == "token_length":
            # More accurate token count - process with the tokenizer
            # First, try to use the tokenizer if available
            try:
                if hasattr(self.dataset, "tokenizer") and self.dataset.tokenizer:
                    tokens = self.dataset.tokenizer.tokenize(label)
                    # Count actual tokens for more accurate difficulty
                    return len(tokens)
            except:
                pass
                
            # Fallback to splitting, but with better heuristics
            tokens = []
            # Add simple regex split
            import re
            pattern = re.compile(r"(\\[a-zA-Z]+|[^a-zA-Z0-9\s]|\s+|[a-zA-Z0-9]+)")
            matches = pattern.findall(label)
            tokens = [m for m in matches if m.strip()]
            
            # Count characters in braces as added difficulty
            brace_depth = 0
            for char in label:
                if char == '{':
                    brace_depth += 1
                elif char == '}':
                    brace_depth -= 1
            
            # Longer expressions with more nesting are more difficult
            return len(tokens) + brace_depth * 0.5

        elif self.difficulty_metric == "expression_complexity":
            # Combined metric considering:
            # 1. Length of the expression
            # 2. Number of special LaTeX commands
            # 3. Nesting level of structures (fractions, square roots, etc.)
            
            # Simple version: count LaTeX commands as indicators of complexity
            special_commands = ["\\frac", "\\sqrt", "\\sum", "\\prod", "\\int", 
                              "\\lim", "\\begin", "\\end", "\\left", "\\right"]
            
            # Count tokens
            base_difficulty = len(label.split())
            
            # Add complexity for special LaTeX commands
            for cmd in special_commands:
                count = label.count(cmd)
                base_difficulty += count * 2  # Special commands add more difficulty
                
            # Count braces as a proxy for nesting
            brace_level = 0
            max_brace_level = 0
            for char in label:
                if char == '{':
                    brace_level += 1
                    max_brace_level = max(max_brace_level, brace_level)
                elif char == '}':
                    brace_level -= 1
                    
            # Add difficulty for nesting level
            base_difficulty += max_brace_level * 1.5
            
            return base_difficulty
            
        elif self.difficulty_metric == "seq_length":
            # Load the full sample to get input sequence length
            sample = self.dataset[idx]
            return len(sample["input"])
        
        elif self.difficulty_metric == "stroke_count":
            # Count strokes in the ink data
            strokes = ink_data.get("strokes", [])
            return len(strokes)
            
        else:
            # Default to simple token length if unknown metric
            return len(label.split())

    def _sort_indices_by_difficulty(self) -> List[int]:
        """Sort dataset indices by their difficulty scores."""
        # Create (index, difficulty) pairs
        index_difficulty_pairs = [(i, d) for i, d in enumerate(self.difficulties)]

        # Sort by difficulty (ascending - easy to hard)
        index_difficulty_pairs.sort(key=lambda x: x[1])

        # Extract sorted indices
        return [pair[0] for pair in index_difficulty_pairs]

    def _get_current_percentile(self) -> float:
        """
        Calculate the current curriculum percentile based on epoch.

        Uses a modified sigmoid function to create a more gradual pace
        at the beginning and end of the curriculum schedule.

        Returns a value between 0.2 and 1.0 representing how much of the
        difficulty range to include.
        """
        # Using a modified sigmoid to make a more gradual curve
        # We start at 20% samples to ensure we have enough data to learn from
        min_percentile = 0.2  # Start with 20% of the easiest samples
        
        if self.current_epoch >= self.epochs_to_full_difficulty:
            return 1.0
            
        # Linear progress
        # progress = min(1.0, self.current_epoch / self.epochs_to_full_difficulty)
        
        # Sigmoid progress (smoother transition at beginning and end)
        normalized_epoch = (self.current_epoch / self.epochs_to_full_difficulty) * 12 - 6
        sigmoid_value = 1 / (1 + math.exp(-normalized_epoch))  # Sigmoid between 0 and 1
        
        # Scale to range [min_percentile, 1.0]
        return min_percentile + (1.0 - min_percentile) * sigmoid_value

    def set_epoch(self, epoch: int) -> None:
        """Update the current epoch."""
        self.current_epoch = epoch
        self.curr_percentile = self._get_current_percentile()

        # Reseed the RNG for each epoch to ensure reproducible shuffling
        # but different shuffling per epoch
        self.rng = random.Random(self.seed + epoch)

    def __iter__(self):
        """
        Return an iterator over the indices with improved sampling strategy.
        
        This implementation uses a more balanced approach:
        1. Organizes samples into difficulty bins
        2. Maintains a proper progression from easy to hard
        3. Ensures diversity in training by mixing some harder examples
        4. Implements a better shuffling strategy that maintains curriculum properties
        """
        num_samples = len(self.sorted_indices)
        
        # Always include at least 20% of the dataset even at the beginning
        min_samples = max(int(num_samples * 0.2), 1)
        
        # Calculate number of samples to include based on current percentile
        max_index = max(
            min_samples, 
            min(num_samples, math.ceil(num_samples * self.curr_percentile))
        )
        
        # Get subset of indices up to current difficulty level
        curr_indices = self.sorted_indices[:max_index]
        
        # For early training, focus more on easier examples
        # For later training, sample more uniformly across the difficulty range
        final_indices = []
        
        if self.shuffle:
            # Group samples by difficulty - create bins for better grouping
            # This prevents having too many small groups with trivial differences
            difficulty_to_bin = {}
            difficulty_bins = {}
            
            # Create difficulty bins (10% of the max difficulty range per bin)
            bin_count = 10
            if len(self.difficulties) > 0:
                min_difficulty = min(self.difficulties)
                max_difficulty = max(self.difficulties)
                bin_width = max(1, (max_difficulty - min_difficulty) / bin_count)
                
                # Assign each sample to a difficulty bin
                for idx in curr_indices:
                    difficulty = self.difficulties[idx]
                    bin_idx = min(bin_count - 1, int((difficulty - min_difficulty) / bin_width))
                    
                    if bin_idx not in difficulty_bins:
                        difficulty_bins[bin_idx] = []
                    
                    difficulty_bins[bin_idx].append(idx)
                    difficulty_to_bin[idx] = bin_idx
            else:
                # Fallback if difficulties aren't properly calculated
                for idx in curr_indices:
                    difficulty_bins[0] = difficulty_bins.get(0, []) + [idx]
                    difficulty_to_bin[idx] = 0
            
            # Calculate bin sampling weights based on curriculum progression
            # Earlier in curriculum: focus on easier bins
            # Later in curriculum: more uniform sampling across bins
            bin_weights = {}
            
            # Calculate weights using a smoother transition curve
            progress = self.curr_percentile
            
            # In early stages, strongly favor easier examples
            # In later stages, sample more uniformly
            for bin_idx in sorted(difficulty_bins.keys()):
                # Normalized bin position (0 = easiest, 1 = hardest)
                bin_position = bin_idx / max(1, bin_count - 1)
                
                # Early in curriculum: exponential decay for harder bins
                # Later in curriculum: more uniform sampling
                if progress < 0.5:
                    # Early training phase: exponential decay for harder bins
                    decay_factor = 5 * (1 - progress)  # Stronger decay earlier
                    weight = math.exp(-decay_factor * bin_position)
                else:
                    # Later training phase: move toward uniform sampling
                    uniform_weight = 1.0
                    decay_weight = math.exp(-2.5 * bin_position)
                    # Blend between decay and uniform based on progress
                    blend_factor = (progress - 0.5) * 2  # 0 at progress=0.5, 1 at progress=1.0
                    weight = decay_weight * (1 - blend_factor) + uniform_weight * blend_factor
                
                bin_weights[bin_idx] = weight
            
            # Normalize weights to sum to 1.0
            weight_sum = sum(bin_weights.values())
            if weight_sum > 0:
                for bin_idx in bin_weights:
                    bin_weights[bin_idx] /= weight_sum
            
            # Determine how many samples to take from each bin
            bin_sample_counts = {}
            remaining = max_index
            
            # First pass: calculate how many samples per bin based on weights
            for bin_idx, weight in bin_weights.items():
                bin_samples = len(difficulty_bins.get(bin_idx, []))
                # Calculate desired samples based on weight
                desired_samples = int(max_index * weight)
                # Cannot take more samples than available in the bin
                bin_sample_counts[bin_idx] = min(desired_samples, bin_samples)
                remaining -= bin_sample_counts[bin_idx]
            
            # Second pass: distribute any remaining samples
            # Prioritize easier bins if early in curriculum, harder bins if later
            if remaining > 0:
                bin_indices = sorted(bin_weights.keys())
                if progress < 0.5:
                    # Early in curriculum: prioritize easier bins
                    pass  # bins already sorted from easy to hard
                else:
                    # Late in curriculum: prioritize harder bins
                    bin_indices = list(reversed(bin_indices))
                
                for bin_idx in bin_indices:
                    bin_samples = len(difficulty_bins.get(bin_idx, []))
                    can_add = bin_samples - bin_sample_counts[bin_idx]
                    if can_add > 0:
                        to_add = min(remaining, can_add)
                        bin_sample_counts[bin_idx] += to_add
                        remaining -= to_add
                    
                    if remaining <= 0:
                        break
            
            # Now sample from each bin based on the determined counts
            for bin_idx, count in bin_sample_counts.items():
                bin_samples = difficulty_bins.get(bin_idx, [])
                if bin_samples and count > 0:
                    # Shuffle samples within this bin
                    self.rng.shuffle(bin_samples)
                    # Take required number of samples
                    final_indices.extend(bin_samples[:count])
            
            # Final shuffle to mix samples from different bins while maintaining
            # a rough progression from easy to hard
            if len(final_indices) > 100:  # Only for datasets with significant size
                # Group into chunks and shuffle within chunks
                chunk_size = max(10, len(final_indices) // 20)  # 5% of total samples per chunk
                chunks = []
                
                # Sort by difficulty bin first
                final_indices.sort(key=lambda idx: difficulty_to_bin.get(idx, 0))
                
                # Split into chunks
                for i in range(0, len(final_indices), chunk_size):
                    chunk = final_indices[i:i+chunk_size]
                    self.rng.shuffle(chunk)  # Shuffle within chunk
                    chunks.append(chunk)
                    
                # Combine chunks back together
                final_indices = []
                for chunk in chunks:
                    final_indices.extend(chunk)
            else:
                # For smaller datasets, sort by difficulty but introduce some randomness
                final_indices.sort(key=lambda idx: self.difficulties[idx] + self.rng.random() * 0.5)
                
            return iter(final_indices)
        else:
            # When not shuffling, just return indices sorted by difficulty
            return iter(curr_indices)

    def __len__(self) -> int:
        """Return the number of samples in the current curriculum."""
        num_samples = len(self.sorted_indices)
        # Always include at least 20% of the dataset even at the beginning
        min_samples = max(int(num_samples * 0.2), 1)
        max_index = max(
            min_samples, 
            min(num_samples, math.ceil(num_samples * self.curr_percentile))
        )
        return max_index


class CurriculumDataset(HMERDataset):
    """
    Dataset wrapper that adds curriculum learning capabilities.

    This is primarily used to calculate and store difficulty metrics
    for each sample to be used by the CurriculumSampler.
    """

    def __init__(self, *args, difficulty_metric: str = "token_length", **kwargs):
        """Initialize curriculum dataset."""
        super().__init__(*args, **kwargs)
        self.difficulty_metric = difficulty_metric

        # Pre-calculate difficulties for efficient sampling
        self.difficulties = {}
        
        # Track distribution of difficulties for better curriculum planning
        self.difficulty_stats = {
            'min': float('inf'),
            'max': float('-inf'),
            'bins': {},  # Store count of samples in each difficulty range
            'percentiles': {}  # Store percentile cutoffs
        }

        print(f"Initializing curriculum dataset with {difficulty_metric} metric...")

    def calculate_difficulties(self) -> Dict[str, float]:
        """
        Pre-calculate difficulties for all samples and analyze the distribution.

        This enables more intelligent curriculum pacing based on the actual
        difficulty distribution in the dataset.
        """
        print(f"Calculating {self.difficulty_metric} difficulty for {len(self)} samples...")

        difficulties = {}
        all_difficulties = []  # Keep track of all difficulty values for statistics
        
        for idx in range(len(self)):
            file_id = os.path.basename(self.file_paths[idx]).split(".")[0]
            difficulty = self._calculate_sample_difficulty(idx)
            difficulties[file_id] = difficulty
            all_difficulties.append(difficulty)

        self.difficulties = difficulties
        print(f"Calculated difficulties for {len(difficulties)} samples")
        
        # Calculate statistics on difficulty distribution
        if all_difficulties:
            self._analyze_difficulty_distribution(all_difficulties)
            
        return difficulties

    def _analyze_difficulty_distribution(self, difficulties: List[float]) -> None:
        """
        Analyze the distribution of difficulties to improve curriculum pacing.
        
        This allows better planning of the curriculum progression by understanding
        the actual distribution of difficulty in the dataset.
        """
        import numpy as np
        
        # Basic statistics
        min_diff = min(difficulties)
        max_diff = max(difficulties)
        mean_diff = np.mean(difficulties)
        median_diff = np.median(difficulties)
        
        # Calculate percentiles
        percentiles = {}
        for p in [5, 10, 25, 50, 75, 90, 95]:
            percentiles[p] = np.percentile(difficulties, p)
        
        # Create difficulty bins for visualization
        num_bins = 10
        bins = {}
        bin_width = (max_diff - min_diff) / num_bins
        for d in difficulties:
            bin_idx = min(num_bins - 1, int((d - min_diff) / bin_width))
            bin_label = f"{min_diff + bin_idx * bin_width:.1f}-{min_diff + (bin_idx + 1) * bin_width:.1f}"
            bins[bin_label] = bins.get(bin_label, 0) + 1
        
        # Store statistics
        self.difficulty_stats = {
            'min': min_diff,
            'max': max_diff,
            'mean': mean_diff,
            'median': median_diff,
            'percentiles': percentiles,
            'bins': bins
        }
        
        # Print summary
        print(f"Difficulty distribution summary:")
        print(f"  Range: {min_diff:.2f} to {max_diff:.2f}")
        print(f"  Mean: {mean_diff:.2f}, Median: {median_diff:.2f}")
        print(f"  25th percentile: {percentiles[25]:.2f}, 75th percentile: {percentiles[75]:.2f}")
        
        # Log distribution histogram (optional)
        # print("Difficulty distribution:")
        # for bin_label, count in sorted(bins.items()):
        #     print(f"  {bin_label}: {count} samples")

    def _calculate_sample_difficulty(self, idx: int) -> float:
        """Calculate difficulty score for a single sample."""
        # Check if pre-calculated
        file_id = os.path.basename(self.file_paths[idx]).split(".")[0]
        if file_id in self.difficulties:
            return self.difficulties[file_id]

        # Calculate based on metric
        if self.difficulty_metric == "token_length":
            # Parse label without loading full data
            ink_data = self.parser.parse_inkml(self.file_paths[idx])
            label = ink_data.get("normalized_label", ink_data.get("label", ""))
            
            # Use tokenizer for more accurate token count if available
            try:
                tokens = self.tokenizer.tokenize(label)
                difficulty = len(tokens)
            except:
                # Fallback to basic splitting
                difficulty = len(label.split())
                
            # Add complexity factor for special commands
            special_commands = ["\\frac", "\\sqrt", "\\sum", "\\int"]
            for cmd in special_commands:
                if cmd in label:
                    difficulty += 2  # Add extra difficulty for complex symbols

        elif self.difficulty_metric == "expression_complexity":
            # Parse label
            ink_data = self.parser.parse_inkml(self.file_paths[idx])
            label = ink_data.get("normalized_label", ink_data.get("label", ""))
            
            # Combined metric considering multiple factors
            special_commands = ["\\frac", "\\sqrt", "\\sum", "\\prod", "\\int", 
                               "\\lim", "\\begin", "\\end", "\\left", "\\right"]
            
            # Base difficulty from token count
            difficulty = len(label.split())
            
            # Add complexity for special LaTeX commands
            for cmd in special_commands:
                count = label.count(cmd)
                difficulty += count * 2
                
            # Count nesting level through braces
            brace_level = 0
            max_brace_level = 0
            for char in label:
                if char == '{':
                    brace_level += 1
                    max_brace_level = max(max_brace_level, brace_level)
                elif char == '}':
                    brace_level -= 1
                    
            difficulty += max_brace_level * 1.5

        elif self.difficulty_metric == "seq_length":
            # This requires loading the full sample
            sample = super().__getitem__(idx)
            difficulty = len(sample["input"])
            
        elif self.difficulty_metric == "stroke_count":
            # Count strokes in the ink data
            ink_data = self.parser.parse_inkml(self.file_paths[idx])
            strokes = ink_data.get("strokes", [])
            difficulty = len(strokes)

        else:
            # Default to simple token length
            ink_data = self.parser.parse_inkml(self.file_paths[idx])
            label = ink_data.get("normalized_label", ink_data.get("label", ""))
            difficulty = len(label.split())

        # Store for future reference
        self.difficulties[file_id] = difficulty

        return difficulty

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get sample from dataset with difficulty information."""
        sample = super().__getitem__(idx)

        # Add difficulty score if not already present
        if "difficulty" not in sample:
            file_id = sample.get("file_id", "")
            if not file_id:
                # Extract file_id from the path if not in sample
                file_id = os.path.basename(self.file_paths[idx]).split(".")[0]
                
            if file_id in self.difficulties:
                difficulty = self.difficulties[file_id]
            else:
                difficulty = self._calculate_sample_difficulty(idx)
                self.difficulties[file_id] = difficulty

            sample["difficulty"] = difficulty

        return sample
