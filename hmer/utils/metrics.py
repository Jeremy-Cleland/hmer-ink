"""
Evaluation metrics for HMER.
"""

from typing import Dict, List

import Levenshtein
import numpy as np


def compute_edit_distance(pred: str, target: str) -> int:
    """
    Compute Levenshtein edit distance between two strings.

    Args:
        pred: Predicted string
        target: Target string

    Returns:
        Edit distance
    """
    return Levenshtein.distance(pred, target)


def compute_normalized_edit_distance(pred: str, target: str) -> float:
    """
    Compute normalized edit distance between two strings.

    Args:
        pred: Predicted string
        target: Target string

    Returns:
        Normalized edit distance (0 to 1)
    """
    if len(target) == 0:
        return 1.0 if len(pred) > 0 else 0.0

    return Levenshtein.distance(pred, target) / len(target)


def compute_exact_match(pred: str, target: str) -> int:
    """
    Compute exact match between prediction and target.

    Args:
        pred: Predicted string
        target: Target string

    Returns:
        1 if exact match, 0 otherwise
    """
    return 1 if pred == target else 0


def compute_token_accuracy(pred_tokens: List[str], target_tokens: List[str]) -> float:
    """
    Compute token-level accuracy.

    Args:
        pred_tokens: List of predicted tokens
        target_tokens: List of target tokens

    Returns:
        Token accuracy (0 to 1)
    """
    if not target_tokens:
        return 1.0 if not pred_tokens else 0.0

    # Align sequences using Levenshtein
    alignment = Levenshtein.opcodes(pred_tokens, target_tokens)

    correct = 0
    total = len(target_tokens)

    for tag, i1, i2, j1, j2 in alignment:
        if tag == "equal":
            correct += i2 - i1

    return correct / total


def compute_expression_recognition_rate(
    predictions: List[str], targets: List[str]
) -> float:
    """
    Compute Expression Recognition Rate (ERR).

    Args:
        predictions: List of predicted expressions
        targets: List of target expressions

    Returns:
        Expression Recognition Rate (0 to 1)
    """
    exact_matches = sum(p == t for p, t in zip(predictions, targets))
    return exact_matches / len(targets) if targets else 0.0


def compute_symbol_accuracy(predictions: List[str], targets: List[str]) -> float:
    """
    Compute symbol-level accuracy.

    Args:
        predictions: List of predicted expressions
        targets: List of target expressions

    Returns:
        Symbol accuracy (0 to 1)
    """

    # Tokenize expressions to symbols
    def tokenize(expr):
        tokens = []
        i = 0
        while i < len(expr):
            if expr[i : i + 1] == "\\":
                # Command token
                j = i + 1
                while j < len(expr) and expr[j].isalpha():
                    j += 1
                if i + 1 < j:  # Command has at least one letter
                    tokens.append(expr[i:j])
                    i = j
                else:
                    tokens.append(expr[i])
                    i += 1
            else:
                tokens.append(expr[i])
                i += 1
        return tokens

    pred_tokens = [tokenize(p) for p in predictions]
    target_tokens = [tokenize(t) for t in targets]

    total_correct = 0
    total_symbols = 0

    for p_tokens, t_tokens in zip(pred_tokens, target_tokens):
        total_correct += sum(
            p == t for p, t in zip(p_tokens[: len(t_tokens)], t_tokens)
        )
        total_symbols += len(t_tokens)

    return total_correct / total_symbols if total_symbols > 0 else 0.0


def compute_metrics(
    predictions: List[str], targets: List[str], metrics: List[str] = None
) -> Dict[str, float]:
    """
    Compute multiple metrics for a batch of predictions.

    Args:
        predictions: List of predicted expressions
        targets: List of target expressions
        metrics: List of metric names to compute (default: all)

    Returns:
        Dictionary of metrics
    """
    available_metrics = {
        "edit_distance": lambda p, t: compute_edit_distance(p, t),
        "normalized_edit_distance": lambda p, t: compute_normalized_edit_distance(p, t),
        "exact_match": lambda p, t: compute_exact_match(p, t),
        "expression_recognition_rate": lambda ps,
        ts: compute_expression_recognition_rate(ps, ts),
        "symbol_accuracy": lambda ps, ts: compute_symbol_accuracy(ps, ts),
    }

    if metrics is None:
        metrics = available_metrics.keys()

    results = {}

    # Compute sample-level metrics
    sample_metrics = ["edit_distance", "normalized_edit_distance", "exact_match"]
    for metric in metrics:
        if metric in sample_metrics:
            metric_values = [
                available_metrics[metric](p, t) for p, t in zip(predictions, targets)
            ]
            results[metric] = np.mean(metric_values)

    # Compute dataset-level metrics
    dataset_metrics = ["expression_recognition_rate", "symbol_accuracy"]
    for metric in metrics:
        if metric in dataset_metrics:
            results[metric] = available_metrics[metric](predictions, targets)

    return results
