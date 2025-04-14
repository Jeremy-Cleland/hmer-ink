"""
Transformations for ink data processing and augmentation.
"""

from typing import Callable, Dict, List, Tuple

import numpy as np


class Compose:
    """Compose multiple transforms."""

    def __init__(self, transforms: List[Callable]):
        """
        Args:
            transforms: List of transforms to apply in sequence
        """
        self.transforms = transforms

    def __call__(self, data):
        """Apply all transforms in sequence."""
        for t in self.transforms:
            data = t(data)
        return data


class RandomScale:
    """Randomly scale the ink data."""

    def __init__(self, scale_range: Tuple[float, float] = (0.8, 1.2)):
        """
        Args:
            scale_range: Range of scaling factors (min, max)
        """
        self.scale_range = scale_range

    def __call__(self, data: np.ndarray) -> np.ndarray:
        """
        Apply random scaling to the ink data.

        Args:
            data: Numpy array of shape (N, D) where N is the number of points
                 and D is the dimension of each point

        Returns:
            Transformed data
        """
        if len(data) == 0:
            return data

        # Generate random scaling factors for x and y
        scale_x = np.random.uniform(self.scale_range[0], self.scale_range[1])
        scale_y = np.random.uniform(self.scale_range[0], self.scale_range[1])

        # Apply scaling
        result = data.copy()

        # If using absolute coordinates
        if data.shape[1] >= 2 and data.shape[1] <= 4:  # Basic x, y coordinates
            result[:, 0] *= scale_x
            result[:, 1] *= scale_y

        # If using relative coordinates
        elif data.shape[1] >= 4:  # Relative (dx, dy) coordinates
            # Scale only the delta values, not the pen state
            result[:, 0] *= scale_x
            result[:, 1] *= scale_y

        return result


class RandomRotation:
    """Randomly rotate the ink data."""

    def __init__(
        self, angle_range: Tuple[float, float] = (-15, 15), max_probability: float = 0.8
    ):
        """
        Args:
            angle_range: Range of rotation angles in degrees (min, max)
            max_probability: Maximum probability of applying the rotation
                            (introduces some unrotated samples)
        """
        self.angle_range = angle_range
        self.max_probability = max_probability

    def __call__(self, data: np.ndarray) -> np.ndarray:
        """
        Apply random rotation to the ink data.

        Args:
            data: Numpy array of shape (N, D) where N is the number of points
                 and D is the dimension of each point

        Returns:
            Transformed data
        """
        if len(data) == 0:
            return data

        # Sometimes skip rotation entirely - especially useful for complex expressions
        if np.random.random() > self.max_probability:
            return data.copy()

        # Generate random angle in degrees - use smaller rotations for bigger samples
        # (Complex expressions with many points typically need gentler rotations)
        point_count_factor = min(
            1.0, 100 / max(len(data), 1)
        )  # Scale down for larger samples
        effective_min = self.angle_range[0] * point_count_factor
        effective_max = self.angle_range[1] * point_count_factor

        angle = np.random.uniform(effective_min, effective_max)

        # Convert to radians
        theta = np.radians(angle)

        # Create rotation matrix
        c, s = np.cos(theta), np.sin(theta)
        rotation_matrix = np.array([[c, -s], [s, c]])

        # Apply rotation
        result = data.copy()

        # If using absolute coordinates
        if data.shape[1] >= 2 and data.shape[1] <= 4:
            # Apply rotation to x, y coordinates
            coords = result[:, :2]
            result[:, :2] = np.dot(coords, rotation_matrix.T)

        # If using relative coordinates
        elif data.shape[1] >= 4:
            # Apply rotation to dx, dy values
            deltas = result[:, :2]
            result[:, :2] = np.dot(deltas, rotation_matrix.T)

        return result


class RandomTranslation:
    """Randomly translate the ink data."""

    def __init__(self, translation_range: Tuple[float, float] = (-0.1, 0.1)):
        """
        Args:
            translation_range: Range of translation factors as proportion of
                               overall width/height (min, max)
        """
        self.translation_range = translation_range

    def __call__(self, data: np.ndarray) -> np.ndarray:
        """
        Apply random translation to the ink data.

        Args:
            data: Numpy array of shape (N, D) where N is the number of points
                 and D is the dimension of each point

        Returns:
            Transformed data
        """
        if len(data) == 0:
            return data

        result = data.copy()

        # Only apply to absolute coordinates
        if data.shape[1] >= 2 and data.shape[1] <= 4:
            # Compute data ranges
            x_min, x_max = data[:, 0].min(), data[:, 0].max()
            y_min, y_max = data[:, 1].min(), data[:, 1].max()

            # Compute translation amounts
            width = max(x_max - x_min, 1e-6)
            height = max(y_max - y_min, 1e-6)

            tx = (
                np.random.uniform(self.translation_range[0], self.translation_range[1])
                * width
            )
            ty = (
                np.random.uniform(self.translation_range[0], self.translation_range[1])
                * height
            )

            # Apply translation
            result[:, 0] += tx
            result[:, 1] += ty

        # For relative coordinates, add translation to first point only
        elif data.shape[1] >= 4 and len(data) > 0:
            # Add translation to the first point only
            # This effectively shifts the starting position
            width = 2.0  # Assuming normalized to [-1, 1]
            height = 2.0

            tx = (
                np.random.uniform(self.translation_range[0], self.translation_range[1])
                * width
            )
            ty = (
                np.random.uniform(self.translation_range[0], self.translation_range[1])
                * height
            )

            result[0, 0] += tx
            result[0, 1] += ty

        return result


class RandomStrokeDropout:
    """Randomly drop strokes for augmentation."""

    def __init__(
        self,
        dropout_prob: float = 0.05,
        pen_up_value: float = 1.0,
        max_dropout_ratio: float = 0.2,
    ):
        """
        Args:
            dropout_prob: Probability of dropping a stroke
            pen_up_value: Value indicating pen up in the data
            max_dropout_ratio: Maximum ratio of strokes that can be dropped (safety measure)
        """
        self.dropout_prob = dropout_prob
        self.pen_up_value = pen_up_value
        self.max_dropout_ratio = max_dropout_ratio

    def __call__(self, data: np.ndarray) -> np.ndarray:
        """
        Randomly drop strokes from the ink data.

        Args:
            data: Numpy array with pen state in the last column

        Returns:
            Transformed data with some strokes dropped
        """
        if len(data) == 0 or data.shape[1] < 3:  # Need at least pen state
            return data

        # Count strokes to determine max number that can be dropped
        stroke_count = 0
        for i, point in enumerate(data):
            # Check if this is the end of a stroke
            pen_up = False
            if data.shape[1] >= 4:  # Relative coordinates with pen state
                pen_up = point[-1] >= self.pen_up_value / 2

            if pen_up or i == len(data) - 1:
                stroke_count += 1

        # For very few strokes, don't drop any
        if stroke_count <= 3:
            return data.copy()

        # Adapt dropout probability based on stroke count
        # Fewer strokes → lower dropout probability
        adjusted_prob = min(
            self.dropout_prob, self.dropout_prob * (stroke_count / 10.0)
        )

        # Calculate maximum strokes to drop (as a safety check)
        max_strokes_to_drop = int(stroke_count * self.max_dropout_ratio)
        strokes_dropped = 0

        result = []
        stroke = []
        drop_current = False

        # Split data into strokes and randomly drop some
        for i, point in enumerate(data):
            stroke.append(point)

            # Check if this is the end of a stroke
            pen_up = False
            if data.shape[1] >= 4:  # Relative coordinates with pen state
                pen_up = point[-1] >= self.pen_up_value / 2

            if pen_up or i == len(data) - 1:
                should_drop = drop_current or (
                    np.random.random() < adjusted_prob
                    and strokes_dropped < max_strokes_to_drop
                )

                # Decide whether to drop this stroke
                if not should_drop:
                    result.extend(stroke)
                else:
                    strokes_dropped += 1

                stroke = []
                drop_current = np.random.random() < adjusted_prob

        # Handle case where no strokes remain or too many are dropped
        if not result or strokes_dropped >= stroke_count:
            return data  # Return original data if all strokes would be dropped

        return np.array(result)


class RandomJitter:
    """Add random jitter to ink data."""

    def __init__(self, jitter_scale: float = 0.01, max_probability: float = 0.9):
        """
        Args:
            jitter_scale: Scale of the random jitter to add
            max_probability: Maximum probability of applying jitter to each point
        """
        self.jitter_scale = jitter_scale
        self.max_probability = max_probability

    def __call__(self, data: np.ndarray) -> np.ndarray:
        """
        Add random noise to the ink data.

        Args:
            data: Numpy array of shape (N, D) where N is the number of points
                 and D is the dimension of each point

        Returns:
            Transformed data with jitter
        """
        if len(data) == 0:
            return data

        # For complex expressions, use even more conservative jitter
        point_count = len(data)
        # Scale down jitter for larger expressions (more points)
        scaling_factor = min(1.0, 100 / max(point_count, 1))
        effective_jitter = self.jitter_scale * scaling_factor

        # Apply jitter only to some points (not all) to maintain structure better
        jitter_mask = np.random.random(point_count) < self.max_probability

        result = data.copy()

        # Add jitter to x and y coordinates
        if data.shape[1] >= 2:
            # Determine scale of jitter based on data range
            x_range = data[:, 0].max() - data[:, 0].min()
            y_range = data[:, 1].max() - data[:, 1].min()

            # Generate jitter for all points, but only apply to selected ones
            jitter_x = np.random.normal(
                0, effective_jitter * x_range, size=data.shape[0]
            )
            jitter_y = np.random.normal(
                0, effective_jitter * y_range, size=data.shape[0]
            )

            # Apply jitter only to selected points using the mask
            result[:, 0] += jitter_x * jitter_mask
            result[:, 1] += jitter_y * jitter_mask

        return result


class Resample:
    """Resample ink data to have a fixed number of points."""

    def __init__(self, num_points: int = 512, preserve_stroke_ends: bool = True):
        """
        Args:
            num_points: Target number of points after resampling
            preserve_stroke_ends: Whether to preserve the endpoints of strokes
        """
        self.num_points = num_points
        self.preserve_stroke_ends = preserve_stroke_ends

    def __call__(self, data: np.ndarray) -> np.ndarray:
        """
        Resample the ink data to have a fixed number of points.

        Args:
            data: Numpy array of shape (N, D) where N is the number of points
                 and D is the dimension of each point

        Returns:
            Resampled data with self.num_points points
        """
        if len(data) == 0 or len(data) <= self.num_points:
            # Pad with zeros if necessary
            if len(data) < self.num_points:
                padding = np.zeros((self.num_points - len(data), data.shape[1]))
                return np.vstack([data, padding])
            return data

        # Identify stroke boundaries if pen state is available
        stroke_ends = []
        if data.shape[1] >= 3:  # At least x, y, and pen state
            pen_state_idx = -1  # Last column is assumed to be pen state
            for i in range(len(data)):
                if data[i, pen_state_idx] > 0.5:  # Pen up
                    stroke_ends.append(i)

        if not stroke_ends:
            stroke_ends = [len(data) - 1]  # Just the end of the sequence

        # Determine how many points to keep
        if self.preserve_stroke_ends:
            # Reserve points for stroke ends
            reserved_points = len(stroke_ends)
            points_to_sample = self.num_points - reserved_points

            if points_to_sample <= 0:
                # Too many strokes, can't preserve all ends
                # Just use uniform sampling
                indices = np.linspace(0, len(data) - 1, self.num_points).astype(int)
                return data[indices]

            # Sample remaining points uniformly, excluding stroke ends
            mask = np.ones(len(data), dtype=bool)
            mask[stroke_ends] = False

            # Get indices of non-stroke-end points
            non_end_indices = np.arange(len(data))[mask]

            if len(non_end_indices) > points_to_sample:
                # Randomly sample from non-end points
                sampled_indices = np.sort(
                    np.random.choice(non_end_indices, points_to_sample, replace=False)
                )
            else:
                # Not enough non-end points, use what we have
                sampled_indices = non_end_indices

                # Add more points if needed by duplicating
                remaining = points_to_sample - len(sampled_indices)
                if remaining > 0:
                    extra_indices = np.random.choice(
                        non_end_indices, remaining, replace=True
                    )
                    sampled_indices = np.sort(
                        np.concatenate([sampled_indices, extra_indices])
                    )

            # Combine stroke end indices with sampled indices
            indices = np.sort(np.concatenate([sampled_indices, stroke_ends]))

        else:
            # Simple uniform sampling
            indices = np.linspace(0, len(data) - 1, self.num_points).astype(int)

        return data[indices]


# Create augmentation pipeline
def get_train_transforms(config: Dict) -> Callable:
    """
    Create a transform pipeline for training data.

    Args:
        config: Configuration dictionary with augmentation settings

    Returns:
        Composed transform function
    """
    aug_config = config.get("augmentation", {})
    enabled = aug_config.get("enabled", True)

    if not enabled:
        return None

    transforms = []

    # Add random scaling (more conservative range)
    scale_range = aug_config.get("scale_range", (0.9, 1.1))
    transforms.append(RandomScale(scale_range))

    # Add random rotation (reduced angle range with adaptive behavior)
    rotation_range = aug_config.get("rotation_range", (-10, 10))
    rotation_prob = aug_config.get("rotation_probability", 0.7)
    transforms.append(RandomRotation(rotation_range, max_probability=rotation_prob))

    # Add random translation (smaller shifts)
    translation_range = aug_config.get("translation_range", (-0.05, 0.05))
    transforms.append(RandomTranslation(translation_range))

    # Add random stroke dropout (lower probability with safety checks)
    dropout_prob = aug_config.get("stroke_dropout_prob", 0.03)
    max_dropout_ratio = aug_config.get("max_dropout_ratio", 0.2)
    transforms.append(
        RandomStrokeDropout(dropout_prob, max_dropout_ratio=max_dropout_ratio)
    )

    # Add random jitter (smaller noise with adaptive behavior)
    jitter_scale = aug_config.get("jitter_scale", 0.005)
    jitter_prob = aug_config.get("jitter_probability", 0.7)
    transforms.append(RandomJitter(jitter_scale, max_probability=jitter_prob))

    return Compose(transforms)


def get_eval_transforms(config: Dict) -> Callable:
    """
    Create a transform pipeline for evaluation data.

    Args:
        config: Configuration dictionary

    Returns:
        Composed transform function or None
    """
    # No augmentation for evaluation
    return None
