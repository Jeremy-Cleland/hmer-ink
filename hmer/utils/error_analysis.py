"""
Error analysis utilities for HMER models.
"""

from typing import Dict, List, Optional

from hmer.utils.metrics import compute_metrics


def analyze_errors(
    predictions: List[str],
    references: List[str],
    image_paths: Optional[List[str]] = None,
) -> Dict:
    """Analyzes common error patterns between predicted and reference sequences.

    Args:
        predictions: List of predicted LaTeX strings
        references: List of reference LaTeX strings
        image_paths: Optional list of image paths for each example

    Returns:
        Dictionary containing detailed error analysis results
    """
    error_details = []
    cer_scores = []
    exact_matches = 0

    if image_paths is None:
        image_paths = ["N/A"] * len(predictions)

    for i, (pred, ref, path) in enumerate(zip(predictions, references, image_paths)):
        pred_tokens = pred.split()
        ref_tokens = ref.split()

        # Calculate metrics for this example
        sample_metrics = compute_metrics([pred], [ref])
        edit_distance = sample_metrics["edit_distance"]
        normalized_edit_distance = sample_metrics["normalized_edit_distance"]
        is_match = 1 if pred == ref else 0

        cer_scores.append(normalized_edit_distance)
        exact_matches += is_match

        # Check for specific LaTeX structure errors
        error_details.append({
            "index": i,
            "image_path": path,
            "prediction": pred,
            "reference": ref,
            "edit_distance": edit_distance,
            "normalized_edit_distance": normalized_edit_distance,
            "is_match": is_match,
            "pred_len": len(pred_tokens),
            "ref_len": len(ref_tokens),
            "len_diff": len(pred_tokens) - len(ref_tokens),
            "braces_balanced": check_balanced_delimiters(pred),
            "has_frac": 1 if "\\frac" in ref else 0,
            "pred_has_frac": 1 if "\\frac" in pred else 0,
            "has_sqrt": 1 if "\\sqrt" in ref else 0,
            "pred_has_sqrt": 1 if "\\sqrt" in pred else 0,
        })

    # Aggregate Analysis
    avg_cer = sum(cer_scores) / max(len(cer_scores), 1)
    accuracy = exact_matches / max(len(predictions), 1)

    # Structure-specific metrics
    frac_samples = [d for d in error_details if d["has_frac"] == 1]
    sqrt_samples = [d for d in error_details if d["has_sqrt"] == 1]

    frac_accuracy = sum(d["is_match"] for d in frac_samples) / max(len(frac_samples), 1)
    sqrt_accuracy = sum(d["is_match"] for d in sqrt_samples) / max(len(sqrt_samples), 1)

    # Length-based slicing
    short_expr = [d for d in error_details if d["ref_len"] < 10]
    medium_expr = [d for d in error_details if 10 <= d["ref_len"] < 30]
    long_expr = [d for d in error_details if d["ref_len"] >= 30]

    short_accuracy = sum(d["is_match"] for d in short_expr) / max(len(short_expr), 1)
    medium_accuracy = sum(d["is_match"] for d in medium_expr) / max(len(medium_expr), 1)
    long_accuracy = sum(d["is_match"] for d in long_expr) / max(len(long_expr), 1)

    # Common error detection
    structure_errors = sum(1 for d in error_details if not d["braces_balanced"])
    frac_detection_errors = sum(
        1 for d in error_details if d["has_frac"] != d["pred_has_frac"]
    )

    results = {
        "overall": {
            "accuracy": accuracy,
            "cer": avg_cer,
            "exact_matches": exact_matches,
            "total_samples": len(predictions),
        },
        "structure_slices": {
            "frac_accuracy": frac_accuracy,
            "sqrt_accuracy": sqrt_accuracy,
            "structure_errors": structure_errors,
            "frac_detection_errors": frac_detection_errors,
        },
        "length_slices": {
            "short_accuracy": short_accuracy,
            "medium_accuracy": medium_accuracy,
            "long_accuracy": long_accuracy,
        },
        "details": error_details,
    }

    return results


def check_balanced_delimiters(latex_str: str) -> bool:
    """Check if LaTeX string has balanced delimiters (braces, brackets, parentheses)."""
    stack = []
    delimiters = {"{": "}", "[": "]", "(": ")"}

    for char in latex_str:
        if char in delimiters:
            stack.append(char)
        elif char in delimiters.values():
            if not stack:
                return False

            # Check if closing delimiter matches the most recent opening delimiter
            last_open = stack.pop()
            if delimiters[last_open] != char:
                return False

    return len(stack) == 0  # Stack should be empty if delimiters are balanced