"""
Dataset classes for the HMER-Ink project.
"""

import os
import json
import torch
import random
import numpy as np
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Optional, Union, Callable
from .inkml import InkmlParser
from ..utils.tokenizer import LaTeXTokenizer


class HMERDataset(Dataset):
    """
    Dataset for Handwritten Mathematical Expression Recognition.
    """
    
    def __init__(self, 
                data_dir: str,
                split_dirs: List[str],
                tokenizer: LaTeXTokenizer,
                max_seq_length: int = 512,
                max_token_length: int = 128,
                transform: Optional[Callable] = None,
                normalize: bool = True,
                use_relative_coords: bool = True,
                x_range: Tuple[float, float] = (-1, 1),
                y_range: Tuple[float, float] = (-1, 1),
                time_range: Optional[Tuple[float, float]] = (0, 1),
                cache_dir: Optional[str] = None):
        """
        Initialize the HMER dataset.
        
        Args:
            data_dir: Root directory of the dataset
            split_dirs: List of directory names to include (e.g., ["train", "synthetic"])
            tokenizer: Tokenizer for LaTeX expressions
            max_seq_length: Maximum number of points to include
            max_token_length: Maximum number of tokens in the output sequence
            transform: Optional transform to apply to the data
            normalize: Whether to normalize coordinates
            use_relative_coords: Whether to use relative coordinates
            x_range: Range for normalized x coordinates
            y_range: Range for normalized y coordinates
            time_range: Range for normalized time values
            cache_dir: Directory to cache processed data
        """
        self.data_dir = data_dir
        self.split_dirs = split_dirs
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.max_token_length = max_token_length
        self.transform = transform
        self.normalize = normalize
        self.use_relative_coords = use_relative_coords
        self.x_range = x_range
        self.y_range = y_range
        self.time_range = time_range
        self.cache_dir = cache_dir
        
        # Parser for InkML files
        self.parser = InkmlParser()
        
        # Collect all file paths
        self.file_paths = []
        for split_dir in split_dirs:
            dir_path = os.path.join(data_dir, split_dir)
            if os.path.isdir(dir_path):
                for filename in os.listdir(dir_path):
                    if filename.endswith('.inkml'):
                        file_path = os.path.join(dir_path, filename)
                        self.file_paths.append(file_path)
        
        print(f"Found {len(self.file_paths)} files in {split_dirs}")
        
        # Cache for faster loading (optional)
        self.cache = {}
        if cache_dir and os.path.exists(cache_dir):
            self._load_cache()
    
    def _load_cache(self):
        """Load cached data if available."""
        if not self.cache_dir:
            return
            
        cache_path = os.path.join(self.cache_dir, f"cache_{'_'.join(self.split_dirs)}.pt")
        if os.path.exists(cache_path):
            try:
                self.cache = torch.load(cache_path)
                print(f"Loaded cache with {len(self.cache)} items")
            except Exception as e:
                print(f"Error loading cache: {e}")
                self.cache = {}
    
    def _save_cache(self):
        """Save processed data to cache."""
        if not self.cache_dir:
            return
            
        os.makedirs(self.cache_dir, exist_ok=True)
        cache_path = os.path.join(self.cache_dir, f"cache_{'_'.join(self.split_dirs)}.pt")
        torch.save(self.cache, cache_path)
        print(f"Saved cache with {len(self.cache)} items")
    
    def __len__(self) -> int:
        return len(self.file_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        file_path = self.file_paths[idx]
        file_id = os.path.basename(file_path).split('.')[0]
        
        # Check if already cached
        if file_id in self.cache:
            return self.cache[file_id]
        
        # Parse the file
        ink_data = self.parser.parse_inkml(file_path)
        
        # Get strokes
        strokes = ink_data['strokes']
        
        # Normalize if requested
        if self.normalize:
            strokes = self.parser.normalize_strokes(
                strokes, 
                x_range=self.x_range, 
                y_range=self.y_range, 
                time_range=self.time_range
            )
        
        # Convert to relative coordinates if requested
        if self.use_relative_coords:
            strokes = self.parser.get_relative_coordinates(strokes)
        
        # Flatten strokes
        points = self.parser.flatten_strokes(strokes)
        
        # Apply additional transforms if provided
        if self.transform:
            points = self.transform(points)
        
        # Truncate or pad to max_seq_length
        if len(points) > self.max_seq_length:
            points = points[:self.max_seq_length]
        
        # Convert to tensor
        points_tensor = torch.tensor(points, dtype=torch.float32)
        
        # Get the label
        label = ink_data['normalized_label']
        if 'symbols' in file_path and not label:
            label = ink_data['label']  # For symbols, use 'label' instead

        # Tokenize the label
        tokenized = self.tokenizer.encode(label, max_length=self.max_token_length)
        
        # Create the sample
        sample = {
            'input': points_tensor,
            'target': torch.tensor(tokenized, dtype=torch.long),
            'target_length': torch.tensor(len(tokenized), dtype=torch.long),
            'file_id': file_id,
            'label': label,
        }
        
        # Cache the result if using cache
        if self.cache_dir:
            self.cache[file_id] = sample
            
            # Save cache periodically
            if len(self.cache) % 1000 == 0 and len(self.cache) > 0:
                self._save_cache()
        
        return sample
    
    def collate_fn(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Custom collate function for padding sequences in a batch.
        
        Args:
            batch: List of samples
            
        Returns:
            Dictionary with batched tensors
        """
        # Get maximum sequence length in this batch
        max_input_len = max(sample['input'].shape[0] for sample in batch)
        max_target_len = max(sample['target'].shape[0] for sample in batch)
        
        # Get feature dimension
        input_dim = batch[0]['input'].shape[1]
        
        # Prepare padded tensors
        padded_inputs = torch.zeros(len(batch), max_input_len, input_dim)
        padded_targets = torch.zeros(len(batch), max_target_len, dtype=torch.long)
        input_lengths = torch.zeros(len(batch), dtype=torch.long)
        target_lengths = torch.zeros(len(batch), dtype=torch.long)
        
        # Fill padded tensors
        file_ids = []
        labels = []
        for i, sample in enumerate(batch):
            input_seq = sample['input']
            target_seq = sample['target']
            
            seq_len = input_seq.shape[0]
            padded_inputs[i, :seq_len] = input_seq
            input_lengths[i] = seq_len
            
            target_len = target_seq.shape[0]
            padded_targets[i, :target_len] = target_seq
            target_lengths[i] = target_len
            
            file_ids.append(sample['file_id'])
            labels.append(sample['label'])
        
        return {
            'input': padded_inputs,
            'target': padded_targets,
            'input_lengths': input_lengths,
            'target_lengths': target_lengths,
            'file_ids': file_ids,
            'labels': labels,
        }


class HMERSyntheticDataset(HMERDataset):
    """
    Dataset extension for handling synthetic data with bounding boxes.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize the synthetic dataset."""
        super().__init__(*args, **kwargs)
        self.bboxes = self._load_bboxes()
    
    def _load_bboxes(self) -> Dict:
        """Load bounding box information for synthetic data."""
        bbox_file = os.path.join(self.data_dir, 'synthetic-bboxes.jsonl')
        if not os.path.exists(bbox_file):
            print(f"Warning: Could not find bounding box file at {bbox_file}")
            return {}
            
        bboxes = {}
        with open(bbox_file, 'r') as f:
            for line in f:
                try:
                    bbox_data = json.loads(line.strip())
                    label = bbox_data.get('label', '')
                    if label:
                        bboxes[label] = bbox_data
                except json.JSONDecodeError:
                    continue
                    
        print(f"Loaded {len(bboxes)} bounding box entries")
        return bboxes
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample from the dataset, enhanced with bounding box information if available.
        """
        sample = super().__getitem__(idx)
        
        # Add bounding box information if available
        file_path = self.file_paths[idx]
        if 'synthetic' in file_path:
            ink_data = self.parser.parse_inkml(file_path)
            label = ink_data['label']
            
            if label in self.bboxes:
                bbox_data = self.bboxes[label]
                sample['bbox_data'] = bbox_data.get('bboxes', [])
        
        return sample