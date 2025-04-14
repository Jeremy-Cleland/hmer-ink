"""
Plotting utilities and theme management for HMER-Ink visualizations.
"""

import logging
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

from ..data.inkml import InkmlParser
from ..data.transforms import (
    RandomJitter, 
    RandomRotation,
    RandomScale, 
    RandomStrokeDropout,
    RandomTranslation
)

logger = logging.getLogger(__name__)

# --- Theme Configuration ---


class ThemeName(Enum):
    """Enum for theme names to avoid string literals."""

    LIGHT = "light"
    DARK = "dark"
    PUBLICATION = "publication"

    def __str__(self):
        return self.value


DEFAULT_THEMES = {
    "light": {
        "style": "seaborn-v0_8-whitegrid",
        "colormap": "viridis",
        "sequential_palette": "YlOrRd",
        "categorical_palette": "Set1",
        "diverging_cmap": "RdBu_r",
        "background_color": "#FFFFFF",
        "text_color": "#000000",
        "grid_color": "#CCCCCC",
        "main_color": "#66C2A6",
        "dpi": 300,
        "figure_size": [10, 6],
        "font_scale": 1.2,
        "bar_colors": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"],
    },
    "dark": {
        "style": "dark_background",
        "colormap": "plasma",
        "sequential_palette": "plasma",
        "categorical_palette": "Set2",
        "diverging_cmap": "coolwarm",
        "background_color": "#121212",
        "text_color": "#FFFFFF",
        "grid_color": "#333333",
        "main_color": "#66C2A6",
        "dpi": 300,
        "figure_size": [10, 6],
        "font_scale": 1.2,
        "bar_colors": [
            "#4e79a7",
            "#f28e2c",
            "#e15759",
            "#76b7b2",
            "#59a14f",
            "#edc949",
            "#af7aa1",
            "#ff9da7",
            "#9c755f",
            "#bab0ab",
        ],
    },
    "publication": {
        "style": "seaborn-v0_8-whitegrid",  # Start with a clean grid style
        "colormap": "viridis",
        "sequential_palette": "YlGnBu",
        "categorical_palette": "colorblind",
        "diverging_cmap": "PiYG",
        "background_color": "#FFFFFF",
        "text_color": "#000000",
        "grid_color": "#CCCCCC",
        "main_color": "#4878d0",
        "dpi": 400,  # Increased DPI for publication
        "figure_size": [8, 5],  # Slightly adjusted typical publication size
        "font_scale": 1.1,  # Slightly smaller font for denser plots if needed
        "bar_colors": [
            "#4878d0",
            "#ee854a",
            "#6acc64",
            "#d65f5f",
            "#956cb4",
            "#8c613c",
            "#dc7ec0",
            "#797979",
            "#d5bb67",
            "#82c6e2",
        ],  # Colorblind friendly often good
    },
}


class ThemeRegistry:
    """Central registry for visualization themes."""

    def __init__(self, default_themes: Dict[str, Dict[str, Any]] = DEFAULT_THEMES):
        """Initialize the theme registry."""
        self.themes = default_themes
        self.current_theme_name = ThemeName.DARK
        self.current_theme = self.themes[str(self.current_theme_name)].copy()
        self.is_dark = True
        self.set_theme(self.current_theme_name)  # Apply default theme initially

    def set_theme(self, theme_name: Union[str, ThemeName], **overrides) -> None:
        """
        Set the active theme by name, with optional overrides.

        Args:
            theme_name: Name of the theme to activate ('light', 'dark', 'publication')
            **overrides: Any specific theme values to override
        """
        if isinstance(theme_name, ThemeName):
            theme_name = str(theme_name)

        if theme_name not in self.themes:
            logger.warning(
                f"Theme '{theme_name}' not found, falling back to light theme"
            )
            theme_name = ThemeName.LIGHT

        # Get the base theme settings
        theme_settings = self.themes[theme_name].copy()
        theme_settings.update(overrides)  # Apply overrides

        # Update the current theme state
        self.current_theme = theme_settings
        self.current_theme_name = ThemeName(theme_name)
        self.is_dark = theme_name == ThemeName.DARK

        # Apply theme settings
        self._apply_matplotlib_settings(self.current_theme)
        logger.info(
            f"Set active theme to '{theme_name}' with {len(overrides)} overrides"
        )

    def _apply_matplotlib_settings(self, theme: Dict[str, Any]) -> None:
        """Apply theme settings to matplotlib rcParams."""
        try:
            # Use the defined base style
            plt.style.use(theme.get("style", "default"))
            # Set seaborn context for font scaling etc.
            sns.set_context("paper", font_scale=theme.get("font_scale", 1.2))
        except Exception as e:
            logger.warning(
                f"Could not apply base style '{theme.get('style')}': {e}. Using default."
            )
            plt.style.use("default")
            sns.set_context("paper", font_scale=theme.get("font_scale", 1.2))

        # Convert figure_size from list to tuple if needed
        figure_size = theme.get("figure_size", [10, 6])
        if isinstance(figure_size, list):
            figure_size = tuple(figure_size)

        # Configure matplotlib rcParams directly
        mpl.rcParams.update(
            {
                # Figure settings
                "figure.figsize": figure_size,
                "figure.facecolor": theme.get("background_color", "#FFFFFF"),
                "figure.edgecolor": theme.get("background_color", "#FFFFFF"),
                "savefig.dpi": theme.get("dpi", 300),
                "savefig.facecolor": theme.get("background_color", "#FFFFFF"),
                "savefig.edgecolor": theme.get("background_color", "#FFFFFF"),
                "savefig.transparent": False,
                # Axes settings
                "axes.facecolor": theme.get("background_color", "#FFFFFF"),
                "axes.edgecolor": theme.get(
                    "grid_color", "#CCCCCC"
                ),  # Use grid color for edge
                "axes.labelcolor": theme.get("text_color", "#000000"),
                "axes.titlecolor": theme.get("text_color", "#000000"),
                "axes.grid": True,
                "axes.titlesize": 14 * theme.get("font_scale", 1.2),  # Scale titles
                "axes.labelsize": 12 * theme.get("font_scale", 1.2),  # Scale labels
                # Grid settings
                "grid.color": theme.get("grid_color", "#CCCCCC"),
                "grid.linestyle": "--",
                "grid.linewidth": 0.8,
                "grid.alpha": 0.6,
                # Tick settings
                "xtick.color": theme.get("text_color", "#000000"),
                "ytick.color": theme.get("text_color", "#000000"),
                "xtick.labelsize": 10 * theme.get("font_scale", 1.2),
                "ytick.labelsize": 10 * theme.get("font_scale", 1.2),
                # Legend settings
                "legend.facecolor": theme.get("background_color", "#FFFFFF"),
                "legend.edgecolor": theme.get(
                    "grid_color", "#CCCCCC"
                ),  # Use grid color for edge
                "legend.fontsize": 10 * theme.get("font_scale", 1.2),
                "legend.title_fontsize": 11 * theme.get("font_scale", 1.2),
                "legend.framealpha": 0.8,
                "legend.labelcolor": theme.get("text_color", "#000000"),
                # Text / Font settings (adjust as needed)
                # 'font.family': 'sans-serif',
                # 'font.sans-serif': ['DejaVu Sans', 'Helvetica', 'Arial'], # Example font stack
                "text.color": theme.get("text_color", "#000000"),
            }
        )

    def get_palette(
        self, n_colors: Optional[int] = None, palette_type: str = "categorical"
    ) -> List[str]:
        """Get a color palette of the specified type and length from the current theme.

        Args:
            n_colors: Number of colors to generate. If None for categorical, returns theme's default list.
            palette_type: Type of palette ("categorical", "sequential", or "diverging")

        Returns:
            List of color hex codes
        """
        if palette_type == "categorical":
            palette_name = self.current_theme.get("categorical_palette", "Set1")
            # Use the bar_colors if specified and n_colors is within its range or None
            if "bar_colors" in self.current_theme:
                bar_colors = self.current_theme["bar_colors"]
                if n_colors is None:
                    return bar_colors  # Return full list if n_colors is None
                elif n_colors <= len(bar_colors):
                    return bar_colors[:n_colors]
                else:  # Need more colors than defined, fall back to seaborn generation
                    logger.warning(
                        f"Requested {n_colors} categorical colors, but theme only defines {len(bar_colors)}. Generating with seaborn '{palette_name}'."
                    )
                    pass  # Let seaborn handle generation below
            # If bar_colors not defined or n_colors > len(bar_colors)
            if n_colors is None:
                # Need a default number if not specified and not using bar_colors list
                n_colors = 8
        elif palette_type == "sequential":
            palette_name = self.current_theme.get("sequential_palette", "YlOrRd")
            if n_colors is None:
                n_colors = 6  # Default length for sequential
        elif palette_type == "diverging":
            palette_name = self.current_theme.get("diverging_cmap", "RdBu_r")
            if n_colors is None:
                n_colors = 6  # Default length for diverging
        else:
            logger.warning(
                f"Unknown palette type '{palette_type}', using categorical instead"
            )
            palette_name = self.current_theme.get("categorical_palette", "Set1")
            if n_colors is None:
                n_colors = 8

        # Use seaborn to generate a color palette
        try:
            return sns.color_palette(palette_name, n_colors=n_colors).as_hex()
        except ValueError as e:
            logger.warning(
                f"Could not generate palette '{palette_name}' with {n_colors} colors: {e}. Returning default palette."
            )
            return sns.color_palette("Set1", n_colors=n_colors).as_hex()

    def get_sequential_cmap(self) -> str:
        """Get current sequential colormap name."""
        return self.current_theme.get("sequential_palette", "YlOrRd")

    def get_diverging_cmap(self) -> str:
        """Get current diverging colormap name."""
        return self.current_theme.get("diverging_cmap", "RdBu_r")

    def style_axis(self, ax: plt.Axes):
        """
        Apply theme styling to a matplotlib axis.

        Args:
            ax: The matplotlib axis to style
        """
        # Settings applied via rcParams generally cover this,
        # but we can add specific overrides if needed.
        # Example: Customize spines further
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(self.current_theme["grid_color"])
        ax.spines["bottom"].set_color(self.current_theme["grid_color"])
        ax.grid(True, linestyle="--", alpha=0.6, color=self.current_theme["grid_color"])

    def style_plot(self, fig: plt.Figure, ax: Union[plt.Axes, np.ndarray]):
        """
        Apply theme styling to a matplotlib figure and axis/axes.

        Args:
            fig: The matplotlib figure to style
            ax: The matplotlib axis or array of axes to style
        """
        # Figure styling is handled by rcParams ('figure.facecolor')

        # Apply axis styling to single or multiple axes
        if isinstance(ax, np.ndarray):
            for sub_ax in ax.flatten():
                self.style_axis(sub_ax)
        else:
            self.style_axis(ax)


# --- Global Theme Registry Instance ---
# Create a single instance for the project to use
theme_registry = ThemeRegistry()


# --- Convenience Functions ---
def set_theme(theme_name: Union[str, ThemeName], **overrides) -> None:
    """Set the active theme using the global theme registry."""
    theme_registry.set_theme(theme_name, **overrides)


def get_palette(
    n_colors: Optional[int] = None, palette_type: str = "categorical"
) -> List[str]:
    """Get a color palette using the global theme registry."""
    return theme_registry.get_palette(n_colors, palette_type)


def get_sequential_cmap() -> str:
    """Get current sequential colormap name from the global theme registry."""
    return theme_registry.get_sequential_cmap()


def get_diverging_cmap() -> str:
    """Get current diverging colormap name from the global theme registry."""
    return theme_registry.get_diverging_cmap()


def style_plot(fig: plt.Figure, ax: Union[plt.Axes, np.ndarray]):
    """Apply theme styling to a plot using the global registry."""
    theme_registry.style_plot(fig, ax)


# --- HMER Ink Visualization Functions ---

def get_sample_inkml_files(data_dir: str, split: str = "test", num_samples: int = 5, seed: Optional[int] = None) -> List[str]:
    """
    Get a list of sample InkML files from a specific split.
    
    Args:
        data_dir: Root directory of the dataset
        split: Dataset split to sample from ('train', 'test', 'valid', etc.)
        num_samples: Number of samples to return
        seed: Random seed for reproducibility
        
    Returns:
        List of absolute paths to InkML files
    """
    import glob
    import os
    import random
    
    # Set random seed if provided
    if seed is not None:
        random.seed(seed)
    
    # Get all InkML files in the directory
    split_dir = os.path.join(data_dir, split)
    if not os.path.exists(split_dir):
        logger.warning(f"Split directory {split_dir} does not exist")
        return []
        
    inkml_files = glob.glob(os.path.join(split_dir, "*.inkml"))
    
    if not inkml_files:
        logger.warning(f"No InkML files found in {split_dir}")
        return []
        
    # Sample randomly
    if len(inkml_files) <= num_samples:
        return inkml_files
    else:
        return random.sample(inkml_files, num_samples)

def plot_strokes(
    ax: plt.Axes, 
    strokes: List[np.ndarray], 
    invert_y: bool = True, 
    show_points: bool = True,
    colorful: bool = True,
    alpha: float = 0.8
):
    """
    Plot a list of strokes on the given axes.
    
    Args:
        ax: Matplotlib axes to plot on
        strokes: List of strokes as numpy arrays with (x, y, t) coordinates
        invert_y: Whether to invert the y-axis (default: True for natural writing orientation)
        show_points: Whether to show start and end points of strokes
        colorful: Whether to use different colors for strokes
        alpha: Transparency of the stroke lines
    """
    if not strokes:
        return
        
    if colorful:
        colors = plt.cm.jet(np.linspace(0, 1, len(strokes)))
    else:
        colors = ['blue'] * len(strokes)
        
    for i, stroke in enumerate(strokes):
        # Extract x and y coordinates
        x_coords = stroke[:, 0]
        y_coords = stroke[:, 1] * (-1 if invert_y else 1)
        
        # Plot the stroke
        ax.plot(x_coords, y_coords, color=colors[i], alpha=alpha, linewidth=2)
        
        # Show start and end points if requested
        if show_points:
            ax.scatter(x_coords[0], y_coords[0], color='green', s=30, zorder=5)  # Start point
            ax.scatter(x_coords[-1], y_coords[-1], color='red', s=30, zorder=5)  # End point
            
    # Set equal aspect ratio and remove axes
    ax.set_aspect('equal')
    ax.set_axis_off()

def visualize_ink_normalization(
    inkml_file: str,
    output_path: Optional[str] = None,
    x_range: Tuple[float, float] = (-1, 1),
    y_range: Tuple[float, float] = (-1, 1),
    time_range: Optional[Tuple[float, float]] = None,
    show: bool = False
) -> plt.Figure:
    """
    Visualize the normalization process for an InkML file.
    
    Args:
        inkml_file: Path to the InkML file
        output_path: Path to save the visualization (PDF format)
        x_range: Target range for x coordinates normalization
        y_range: Target range for y coordinates normalization
        time_range: Target range for time values normalization
        show: Whether to display the plot
        
    Returns:
        Matplotlib figure
    """
    # Parse the InkML file
    parser = InkmlParser()
    ink_data = parser.parse_inkml(inkml_file)
    
    # Get raw strokes
    raw_strokes = ink_data["strokes"]
    
    # Normalize strokes with aspect ratio preservation
    normalized_strokes = parser.normalize_strokes(
        raw_strokes, x_range=x_range, y_range=y_range, time_range=time_range,
        preserve_aspect_ratio=True
    )
    
    # Convert to relative coordinates for visualization
    relative_strokes = parser.get_relative_coordinates(normalized_strokes)
    
    # Recreate absolute coordinates from relative for visualization
    reconstructed_strokes = []
    for rel_stroke in relative_strokes:
        abs_stroke = np.zeros((rel_stroke.shape[0], 3))  # x, y, t
        
        # First point is absolute
        abs_stroke[0, :] = rel_stroke[0, :3]  # Copy first point
        
        # Reconstruct subsequent points
        for i in range(1, rel_stroke.shape[0]):
            abs_stroke[i, :] = abs_stroke[i-1, :] + rel_stroke[i, :3]
            
        reconstructed_strokes.append(abs_stroke)
    
    # Create figure
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    # Set title
    fig.suptitle(
        f"Ink Normalization Visualization: {ink_data.get('normalized_label', '')}",
        fontsize=14
    )
    
    # Plot original strokes
    axs[0].set_title("Original Strokes")
    plot_strokes(axs[0], raw_strokes)
    
    # Plot normalized strokes
    axs[1].set_title(f"Normalized Strokes\n(Range: {x_range}, {y_range})")
    plot_strokes(axs[1], normalized_strokes)
    
    # Plot reconstructed from relative strokes
    axs[2].set_title("Reconstructed from Relative")
    plot_strokes(axs[2], reconstructed_strokes)
    
    # Add grid lines to the normalized and reconstructed plots
    axs[1].grid(True, linestyle='--', alpha=0.3)
    axs[2].grid(True, linestyle='--', alpha=0.3)
    
    # Add coordinate bounds to normalized plot
    axs[1].axhline(y=y_range[0], color='gray', linestyle='--', alpha=0.5)
    axs[1].axhline(y=y_range[1], color='gray', linestyle='--', alpha=0.5)
    axs[1].axvline(x=x_range[0], color='gray', linestyle='--', alpha=0.5)
    axs[1].axvline(x=x_range[1], color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    # Save if output path is provided
    if output_path:
        with PdfPages(output_path) as pdf:
            pdf.savefig(fig)
        logger.info(f"Visualization saved to {output_path}")
            
    # Show if requested
    if show:
        plt.show()
    
    return fig

def visualize_augmentations(
    inkml_file: str,
    output_path: Optional[str] = None,
    x_range: Tuple[float, float] = (-1, 1),
    y_range: Tuple[float, float] = (-1, 1),
    time_range: Optional[Tuple[float, float]] = None,
    show: bool = False,
    seed: Optional[int] = None
) -> plt.Figure:
    """
    Visualize the effects of different augmentations on an InkML file.
    
    Args:
        inkml_file: Path to the InkML file
        output_path: Path to save the visualization (PDF format)
        x_range: Target range for x coordinates normalization
        y_range: Target range for y coordinates normalization
        time_range: Target range for time values normalization
        show: Whether to display the plot
        seed: Random seed for reproducibility
        
    Returns:
        Matplotlib figure
    """
    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)
    
    # Parse the InkML file
    parser = InkmlParser()
    ink_data = parser.parse_inkml(inkml_file)
    
    # Get raw strokes
    raw_strokes = ink_data["strokes"]
    
    # Normalize strokes with aspect ratio preservation
    normalized_strokes = parser.normalize_strokes(
        raw_strokes, x_range=x_range, y_range=y_range, time_range=time_range,
        preserve_aspect_ratio=True
    )
    
    # Convert to relative coordinates
    relative_strokes = parser.get_relative_coordinates(normalized_strokes)
    
    # Flatten strokes
    flattened = parser.flatten_strokes(relative_strokes)
    
    # Apply different augmentations with adaptive parameters based on sample complexity
    scale_aug = RandomScale(scale_range=(0.95, 1.05))(flattened.copy())  # Very mild scaling
    
    # Rotation with adaptive behavior based on sequence length
    rotation_aug = RandomRotation(
        angle_range=(-10, 10), 
        max_probability=0.7
    )(flattened.copy())
    
    # Mild translations
    translation_aug = RandomTranslation(
        translation_range=(-0.05, 0.05)
    )(flattened.copy())
    
    # Jitter with protective measures for complex expressions
    jitter_aug = RandomJitter(
        jitter_scale=0.005,
        max_probability=0.7
    )(flattened.copy())
    
    # Stroke dropout with safety measures
    dropout_aug = RandomStrokeDropout(
        dropout_prob=0.03,
        max_dropout_ratio=0.2
    )(flattened.copy())
    
    # Create figure
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    axs = axs.flatten()
    
    # Set title
    fig.suptitle(
        f"Data Augmentation Visualization: {ink_data.get('normalized_label', '')}",
        fontsize=14
    )
    
    # Helper function to convert flattened relative coordinates back to strokes for visualization
    def rel_to_abs_for_vis(rel_data):
        abs_data = np.zeros_like(rel_data)
        abs_data[0] = rel_data[0]  # First point is absolute
        
        # Reconstruct subsequent points
        for i in range(1, rel_data.shape[0]):
            abs_data[i, :2] = abs_data[i-1, :2] + rel_data[i, :2]
            abs_data[i, 2:] = rel_data[i, 2:]  # Keep time and pen state
            
        # Split into strokes based on pen-up signal
        strokes = []
        stroke_start = 0
        
        for i in range(abs_data.shape[0]):
            # Check if pen is up (last dimension)
            if abs_data[i, -1] >= 0.5 or i == abs_data.shape[0] - 1:
                # Extract stroke up to this point
                stroke = abs_data[stroke_start:i+1, :3]  # Keep only x, y, t
                strokes.append(stroke)
                stroke_start = i + 1
                
        return strokes
    
    # Plot original
    normalized_abs = rel_to_abs_for_vis(flattened)
    axs[0].set_title("Original (Normalized)")
    plot_strokes(axs[0], normalized_abs)
    
    # Plot scaled
    scale_abs = rel_to_abs_for_vis(scale_aug)
    axs[1].set_title("Random Scaling")
    plot_strokes(axs[1], scale_abs)
    
    # Plot rotated
    rotation_abs = rel_to_abs_for_vis(rotation_aug)
    axs[2].set_title("Random Rotation")
    plot_strokes(axs[2], rotation_abs)
    
    # Plot translated
    translation_abs = rel_to_abs_for_vis(translation_aug)
    axs[3].set_title("Random Translation")
    plot_strokes(axs[3], translation_abs)
    
    # Plot jittered
    jitter_abs = rel_to_abs_for_vis(jitter_aug)
    axs[4].set_title("Random Jitter")
    plot_strokes(axs[4], jitter_abs)
    
    # Plot with dropout
    dropout_abs = rel_to_abs_for_vis(dropout_aug)
    axs[5].set_title("Stroke Dropout")
    plot_strokes(axs[5], dropout_abs)
    
    # Add grid to all plots
    for ax in axs:
        ax.grid(True, linestyle='--', alpha=0.3)
        
        # Add coordinate bounds
        ax.axhline(y=y_range[0], color='gray', linestyle='--', alpha=0.3)
        ax.axhline(y=y_range[1], color='gray', linestyle='--', alpha=0.3)
        ax.axvline(x=x_range[0], color='gray', linestyle='--', alpha=0.3)
        ax.axvline(x=x_range[1], color='gray', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    
    # Save if output path is provided
    if output_path:
        with PdfPages(output_path) as pdf:
            pdf.savefig(fig)
        logger.info(f"Visualization saved to {output_path}")
            
    # Show if requested
    if show:
        plt.show()
    
    return fig

def batch_visualize_samples(
    data_dir: str, 
    output_dir: str,
    split: str = "test",
    num_samples: int = 5,
    visualization_types: List[str] = ["basic", "normalization", "augmentation"],
    seed: Optional[int] = None
) -> None:
    """
    Generate visualizations for multiple samples in batch mode.
    
    Args:
        data_dir: Root directory of the dataset
        output_dir: Directory to save visualizations
        split: Dataset split to sample from ('train', 'test', 'valid', etc.)
        num_samples: Number of samples to visualize
        visualization_types: List of visualization types to generate ('basic', 'normalization', 'augmentation')
        seed: Random seed for reproducibility
    """
    import os
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get sample files
    sample_files = get_sample_inkml_files(data_dir, split, num_samples, seed)
    
    if not sample_files:
        logger.warning(f"No sample files found in {data_dir}/{split}")
        return
    
    logger.info(f"Generating visualizations for {len(sample_files)} samples from {split} split")
    
    # Generate visualizations for each sample
    for i, inkml_file in enumerate(sample_files):
        file_id = os.path.basename(inkml_file).split('.')[0]
        logger.info(f"Processing sample {i+1}/{len(sample_files)}: {file_id}")
        
        # Parse the file to get the label
        parser = InkmlParser()
        ink_data = parser.parse_inkml(inkml_file)
        label = ink_data.get("normalized_label", "") or ink_data.get("label", "")
        label_safe = label.replace("\\", "").replace("/", "").replace("$", "")[:30]
        
        # Basic visualization
        if "basic" in visualization_types:
            output_path = os.path.join(output_dir, f"{i+1}_{file_id}_basic.pdf")
            try:
                # Create basic visualization
                fig, ax = plt.subplots(figsize=(10, 6))
                plot_strokes(ax, ink_data["strokes"], invert_y=True, show_points=True)
                if label:
                    ax.set_title(f"Label: {label}")
                plt.tight_layout()
                with PdfPages(output_path) as pdf:
                    pdf.savefig(fig)
                plt.close(fig)
                logger.info(f"  - Basic visualization saved to {output_path}")
            except Exception as e:
                logger.error(f"  - Error creating basic visualization: {e}")
        
        # Normalization visualization
        if "normalization" in visualization_types:
            output_path = os.path.join(output_dir, f"{i+1}_{file_id}_normalization.pdf")
            try:
                fig = visualize_ink_normalization(
                    inkml_file=inkml_file,
                    output_path=output_path,
                    show=False
                )
                logger.info(f"  - Normalization visualization saved to {output_path}")
            except Exception as e:
                logger.error(f"  - Error creating normalization visualization: {e}")
        
        # Augmentation visualization
        if "augmentation" in visualization_types:
            output_path = os.path.join(output_dir, f"{i+1}_{file_id}_augmentation.pdf")
            try:
                fig = visualize_augmentations(
                    inkml_file=inkml_file,
                    output_path=output_path,
                    seed=seed,
                    show=False
                )
                logger.info(f"  - Augmentation visualization saved to {output_path}")
            except Exception as e:
                logger.error(f"  - Error creating augmentation visualization: {e}")
    
    logger.info(f"Visualization generation completed. Files saved to {output_dir}")

# Example Usage (can be removed or kept for testing)
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # --- Example 1: Set theme and plot ---
    set_theme(ThemeName.PUBLICATION)

    fig1, ax1 = plt.subplots()
    x = np.linspace(0, 10, 100)
    y1 = np.sin(x)
    y2 = np.cos(x)
    palette = get_palette(2)
    ax1.plot(x, y1, label="Sin(x)", color=palette[0], lw=2)
    ax1.plot(x, y2, label="Cos(x)", color=palette[1], lw=2)
    ax1.set_title("Example Plot (Publication Theme)")
    ax1.set_xlabel("X-axis")
    ax1.set_ylabel("Y-axis")
    ax1.legend()
    style_plot(fig1, ax1)  # Apply spine/grid styling
    plt.tight_layout()
    plt.savefig("example_publication_plot.png")
    plt.close(fig1)
    logger.info("Saved example_publication_plot.png")

    # Test ink visualization if a command line argument is provided
    import sys
    if len(sys.argv) > 1:
        if sys.argv[1] == "--batch":
            # Batch mode
            data_dir = sys.argv[2] if len(sys.argv) > 2 else "data"
            output_dir = sys.argv[3] if len(sys.argv) > 3 else "sample_visualizations"
            split = sys.argv[4] if len(sys.argv) > 4 else "test"
            num_samples = int(sys.argv[5]) if len(sys.argv) > 5 else 5
            
            logger.info(f"Batch visualizing {num_samples} samples from {data_dir}/{split}")
            batch_visualize_samples(
                data_dir=data_dir,
                output_dir=output_dir,
                split=split,
                num_samples=num_samples,
                seed=42
            )
        else:
            # Single file mode
            inkml_file = sys.argv[1]
            logger.info(f"Testing visualization with {inkml_file}")
            
            # Normalize visualization
            visualize_ink_normalization(
                inkml_file, 
                output_path="normalization_visualization.pdf",
                show=False
            )
            
            # Augmentation visualization
            visualize_augmentations(
                inkml_file, 
                output_path="augmentation_visualization.pdf",
                seed=42,
                show=False
            )
