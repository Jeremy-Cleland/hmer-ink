"""
Plotting utilities and theme management for HMER-Ink visualizations.
"""

import logging
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

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

    # --- Example 2: Switch to dark theme ---
    set_theme(ThemeName.DARK)

    fig2, ax2 = plt.subplots()
    categories = ["A", "B", "C", "D", "E"]
    values = [23, 45, 56, 12, 39]
    colors = get_palette(len(categories))
    ax2.bar(categories, values, color=colors)
    ax2.set_title("Example Bar Chart (Dark Theme)")
    ax2.set_xlabel("Category")
    ax2.set_ylabel("Value")
    style_plot(fig2, ax2)
    plt.tight_layout()
    plt.savefig("example_dark_plot.png")
    plt.close(fig2)
    logger.info("Saved example_dark_plot.png")

    # --- Example 3: Back to light theme ---
    set_theme(ThemeName.LIGHT)
    fig3, ax3 = plt.subplots()
    data = np.random.rand(10, 10)
    cmap = get_sequential_cmap()
    im = ax3.imshow(data, cmap=cmap)
    fig3.colorbar(im, ax=ax3)
    ax3.set_title("Example Heatmap (Light Theme)")
    style_plot(fig3, ax3)
    plt.tight_layout()
    plt.savefig("example_light_heatmap.png")
    plt.close(fig3)
    logger.info("Saved example_light_heatmap.png")
