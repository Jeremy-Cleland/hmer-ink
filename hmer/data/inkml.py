"""
InkML parsing utilities for the HMER-Ink dataset.
"""

import os
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Tuple

import numpy as np


class InkmlParser:
    """Parser for InkML files."""

    @staticmethod
    def parse_inkml(file_path: str) -> Dict:
        """
        Parse an InkML file and return a dictionary with its contents.

        Args:
            file_path: Path to the InkML file.

        Returns:
            Dictionary with parsed content, including:
            - strokes: List of strokes, each a numpy array of [x, y, t] points
            - label: Original LaTeX label
            - normalized_label: Normalized LaTeX label (ground truth)
            - metadata: Additional metadata
        """
        tree = ET.parse(file_path)
        root = tree.getroot()

        # XML namespace
        ns = {"inkml": "http://www.w3.org/2003/InkML"}

        # Extract annotations (metadata)
        annotations = {}
        for annotation in root.findall("inkml:annotation", ns):
            annotation_type = annotation.get("type")
            text = annotation.text
            annotations[annotation_type] = text

        # Extract strokes (traces)
        strokes = []
        for trace in root.findall("inkml:trace", ns):
            # trace_id = trace.get("id")  # Not used currently
            points_str = trace.text.strip()
            points = []

            for point_str in points_str.split(","):
                coords = point_str.strip().split()
                if len(coords) >= 3:  # x, y, t
                    x, y, t = float(coords[0]), float(coords[1]), float(coords[2])
                    points.append([x, y, t])
                elif len(coords) >= 2:  # x, y (no time)
                    x, y = float(coords[0]), float(coords[1])
                    points.append([x, y, 0.0])  # Default time to 0

            if points:  # Only add non-empty traces
                strokes.append(np.array(points))

        # Prepare the result dictionary
        result = {
            "strokes": strokes,
            "label": annotations.get("label", ""),
            "normalized_label": annotations.get(
                "normalizedLabel", annotations.get("label", "")
            ),
            "metadata": {
                "sample_id": annotations.get(
                    "sampleId", os.path.basename(file_path).split(".")[0]
                ),
                "split": annotations.get("splitTagOriginal", ""),
                "creation_method": annotations.get("inkCreationMethod", ""),
                "label_creation_method": annotations.get("labelCreationMethod", ""),
            },
        }

        return result

    @staticmethod
    def normalize_strokes(
        strokes: List[np.ndarray],
        x_range: Tuple[float, float] = (-1, 1),
        y_range: Tuple[float, float] = (-1, 1),
        time_range: Optional[Tuple[float, float]] = None,
        preserve_aspect_ratio: bool = True,
    ) -> List[np.ndarray]:
        """
        Normalize stroke coordinates to specified ranges.

        Args:
            strokes: List of strokes, each a numpy array of [x, y, t] points
            x_range: Target range for x coordinates
            y_range: Target range for y coordinates
            time_range: Target range for time values (if None, keep original)
            preserve_aspect_ratio: Whether to preserve the aspect ratio of the original strokes

        Returns:
            List of normalized strokes
        """
        if not strokes:
            return strokes

        # Concatenate all strokes to find global min/max
        all_points = np.vstack(strokes)

        # Compute min and max for x and y coordinates
        x_min, x_max = all_points[:, 0].min(), all_points[:, 0].max()
        y_min, y_max = all_points[:, 1].min(), all_points[:, 1].max()

        # Avoid division by zero
        x_scale = (x_max - x_min) if x_max > x_min else 1.0
        y_scale = (y_max - y_min) if y_max > y_min else 1.0

        # Calculate target width and height
        target_width = x_range[1] - x_range[0]
        target_height = y_range[1] - y_range[0]

        # If preserving aspect ratio, use the same scale for both dimensions
        if preserve_aspect_ratio and x_scale > 0 and y_scale > 0:
            # Calculate original aspect ratio
            original_aspect_ratio = x_scale / y_scale
            
            # Calculate the scaling factor to fit within the target bounds
            # while preserving aspect ratio
            if original_aspect_ratio > 1:  # wider than tall
                # Scale based on width, adjust height
                scale_factor = target_width / x_scale
                effective_height = y_scale * scale_factor
                # Center within the target height
                y_offset = (target_height - effective_height) / 2
            else:  # taller than wide or square
                # Scale based on height, adjust width
                scale_factor = target_height / y_scale
                effective_width = x_scale * scale_factor
                # Center within the target width
                x_offset = (target_width - effective_width) / 2
        else:
            # If not preserving aspect ratio, use different scales for each dimension
            scale_factor_x = target_width / x_scale if x_scale > 0 else 1.0
            scale_factor_y = target_height / y_scale if y_scale > 0 else 1.0
            x_offset = 0
            y_offset = 0

        normalized_strokes = []
        for stroke in strokes:
            normalized_stroke = stroke.copy()

            if preserve_aspect_ratio:
                # Apply the same scaling factor to both dimensions and center within target range
                if original_aspect_ratio > 1:  # wider than tall
                    # Normalize to [0, 1] first
                    normalized_stroke[:, 0] = (stroke[:, 0] - x_min) / x_scale
                    normalized_stroke[:, 1] = (stroke[:, 1] - y_min) / y_scale
                    
                    # Scale to target size while preserving aspect ratio
                    normalized_stroke[:, 0] = normalized_stroke[:, 0] * target_width + x_range[0]
                    normalized_stroke[:, 1] = normalized_stroke[:, 1] * effective_height + y_range[0] + y_offset
                else:
                    # Normalize to [0, 1] first
                    normalized_stroke[:, 0] = (stroke[:, 0] - x_min) / x_scale
                    normalized_stroke[:, 1] = (stroke[:, 1] - y_min) / y_scale
                    
                    # Scale to target size while preserving aspect ratio
                    normalized_stroke[:, 0] = normalized_stroke[:, 0] * effective_width + x_range[0] + x_offset
                    normalized_stroke[:, 1] = normalized_stroke[:, 1] * target_height + y_range[0]
            else:
                # Standard normalization without preserving aspect ratio
                normalized_stroke[:, 0] = (stroke[:, 0] - x_min) / x_scale
                normalized_stroke[:, 0] = (
                    normalized_stroke[:, 0] * target_width + x_range[0]
                )

                normalized_stroke[:, 1] = (stroke[:, 1] - y_min) / y_scale
                normalized_stroke[:, 1] = (
                    normalized_stroke[:, 1] * target_height + y_range[0]
                )

            # Normalize time if requested
            if time_range and stroke.shape[1] > 2:
                t_min, t_max = stroke[:, 2].min(), stroke[:, 2].max()
                t_scale = (t_max - t_min) if t_max > t_min else 1.0

                normalized_stroke[:, 2] = (stroke[:, 2] - t_min) / t_scale
                normalized_stroke[:, 2] = (
                    normalized_stroke[:, 2] * (time_range[1] - time_range[0])
                    + time_range[0]
                )

            normalized_strokes.append(normalized_stroke)

        return normalized_strokes

    @staticmethod
    def get_relative_coordinates(strokes: List[np.ndarray]) -> List[np.ndarray]:
        """
        Convert absolute coordinates to relative displacements within each stroke.
        Also adds a binary "pen-up" feature to indicate the end of a stroke.

        Args:
            strokes: List of strokes, each a numpy array of [x, y, t] points

        Returns:
            List of strokes with relative coordinates and pen-up features
        """
        relative_strokes = []

        for stroke in strokes:
            # Get stroke with relative coordinates
            rel_stroke = np.zeros((stroke.shape[0], stroke.shape[1] + 1))

            # Copy the original coordinates for the first point
            rel_stroke[0, : stroke.shape[1]] = stroke[0]

            # Compute deltas for subsequent points
            if stroke.shape[0] > 1:
                rel_stroke[1:, : stroke.shape[1]] = stroke[1:] - stroke[:-1]

            # Set pen-up feature (1 for the last point in the stroke)
            rel_stroke[-1, -1] = 1.0

            relative_strokes.append(rel_stroke)

        return relative_strokes

    @staticmethod
    def flatten_strokes(strokes: List[np.ndarray]) -> np.ndarray:
        """
        Flatten a list of strokes into a single sequence of points.

        Args:
            strokes: List of strokes, each a numpy array

        Returns:
            Numpy array of shape (N, D) where N is the total number of points
            and D is the dimension of each point
        """
        if not strokes:
            return np.array([])

        return np.vstack(strokes)


# Utility function to test the parser
def test_parser(inkml_file: str) -> None:
    """Test the InkML parser on a sample file and print results."""
    parser = InkmlParser()
    result = parser.parse_inkml(inkml_file)

    print(f"Parsed {inkml_file}")
    print(f"Label: {result['label']}")
    print(f"Normalized Label: {result['normalized_label']}")
    print(f"Metadata: {result['metadata']}")
    print(f"Number of strokes: {len(result['strokes'])}")

    # Normalize strokes
    normalized_strokes = parser.normalize_strokes(result["strokes"])

    # Get relative coordinates
    relative_strokes = parser.get_relative_coordinates(normalized_strokes)

    # Flatten the strokes
    flattened = parser.flatten_strokes(relative_strokes)

    print(f"Flattened shape: {flattened.shape}")


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) > 1:
        test_parser(sys.argv[1])
    else:
        print("Please provide an InkML file path as an argument.")
