"""
Training Monitor for HMER-Ink that provides comprehensive visualization of training metrics.
"""

import glob  # Added for finding error analysis files
import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MaxNLocator  # Added for integer ticks

# Import the new plotting utility
from hmer.utils.plotting import theme_registry


def convert_numpy_types(obj: Any) -> Any:
    """
    Convert numpy data types to Python native types for JSON serialization.

    Args:
        obj: The object to convert

    Returns:
        The converted object with numpy types replaced by Python native types
    """
    # Handle numpy scalars (np.int64, np.float64, etc.)
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return convert_numpy_types(obj.tolist())
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    else:
        return obj


class TrainingMonitor:
    """Monitor and visualize training metrics."""

    def __init__(self, log_dir: str = "outputs/training_metrics"):
        """
        Initialize the training monitor.

        Args:
            log_dir: Directory to save metrics and visualizations
        """
        self.log_dir = log_dir
        self.metrics_file = os.path.join(log_dir, "training_metrics.json")
        self.error_analysis_dir = log_dir  # Store the base directory for error jsons
        self.metrics_history: List[Dict[str, Any]] = []

        os.makedirs(log_dir, exist_ok=True)
        self.plot_dir = os.path.join(log_dir, "plots")  # Define plot_dir
        os.makedirs(self.plot_dir, exist_ok=True)

        # Load existing metrics if available
        if os.path.exists(self.metrics_file):
            with open(self.metrics_file, "r") as f:
                data = json.load(f)
                if isinstance(data, list):
                    self.metrics_history = data

    def log_metrics(
        self,
        metrics: Dict[str, Union[float, int, str]],
        error_examples: Optional[List[Dict]] = None,
        error_analysis_file: Optional[str] = None,
        val_ned_scores_path: Optional[str] = None,
    ):
        """
        Log training metrics.

        Args:
            metrics: Dictionary of metrics to log
            error_examples: List of error examples (optional)
            error_analysis_file: Path to error analysis file (optional)
            val_ned_scores_path: Path to saved validation NED scores (optional)
        """
        # Add timestamp
        metrics["timestamp"] = datetime.now().isoformat()

        # Convert NumPy types to Python native types
        metrics = convert_numpy_types(metrics)

        # Add to history
        self.metrics_history.append(metrics)

        # Save to file
        with open(self.metrics_file, "w") as f:
            json.dump(self.metrics_history, f, indent=2)

        # Generate visualizations, pass the error analysis file path if available
        self.generate_dashboard(error_analysis_file, val_ned_scores_path)

    def generate_dashboard(
        self,
        error_analysis_file: Optional[str] = None,
        val_ned_scores_path: Optional[str] = None,
    ):
        """Generate comprehensive dashboard of training progress."""
        if not self.metrics_history:
            return

        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(self.metrics_history)

        # Only keep numeric columns for plotting
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        plot_df = df[numeric_cols].copy()

        # Create dashboard figure
        self._plot_training_curves(plot_df)
        self._plot_learning_rate(plot_df)
        self._plot_metric_correlations(plot_df)
        self._plot_error_analysis(error_analysis_file)
        self._plot_validation_distribution(val_ned_scores_path)

    def _plot_training_curves(self, df: pd.DataFrame):
        """Plot key training curves using the theme."""
        if "epoch" not in df.columns:
            print("No epoch data to plot training curves.")
            return

        epochs = df["epoch"]

        # Define metrics to plot
        loss_metrics = ["train_loss", "val_loss"]
        other_metrics = [
            "val_expression_recognition_rate",
            "val_symbol_accuracy",
            "val_edit_distance",
            "val_normalized_edit_distance",  # Added Normalized ED
        ]

        # Filter available metrics
        available_loss = [m for m in loss_metrics if m in df.columns]
        available_other = [m for m in other_metrics if m in df.columns]

        if not available_loss and not available_other:
            print("No metrics found to plot training curves.")
            return

        # Use theme settings for figure size etc.
        fig, axes = plt.subplots(
            2,
            1,
            sharex=True,
            figsize=theme_registry.current_theme.get("figure_size", (12, 10)),
        )
        fig.suptitle("Training Progress", fontsize=16)
        theme_registry.style_plot(fig, axes[0])  # Style figure and first axis
        theme_registry.style_axis(axes[1])  # Style second axis

        # --- Plot Loss ---
        ax1 = axes[0]
        palette = theme_registry.get_palette(len(available_loss))
        for i, metric in enumerate(available_loss):
            ax1.plot(
                epochs,
                df[metric],
                label=metric.replace("_", " ").title(),
                color=palette[i],
                linewidth=2,
            )
        ax1.set_ylabel("Loss")
        ax1.set_title("Training & Validation Loss")
        ax1.legend()
        ax1.grid(True, linestyle="--", alpha=0.6)

        # --- Plot Other Metrics ---
        ax2 = axes[1]
        palette = theme_registry.get_palette(len(available_other))
        lines = []
        labels = []
        for i, metric in enumerate(available_other):
            # Use secondary y-axis for edit distance if its scale is very different?
            # For now, plot on the same axis.
            (line,) = ax2.plot(
                epochs,
                df[metric],
                label=metric.replace("_", " ").title(),
                color=palette[i],
                linewidth=2,
            )
            lines.append(line)
            labels.append(metric.replace("_", " ").title())

        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Metric Value")
        ax2.set_title("Validation Metrics")
        ax2.legend(lines, labels)
        ax2.grid(True, linestyle="--", alpha=0.6)
        # Ensure x-axis shows integer epochs
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax2.xaxis.set_major_locator(MaxNLocator(integer=True))

        plt.tight_layout(
            rect=[0, 0.03, 1, 0.97]
        )  # Adjust layout to prevent title overlap
        save_path = os.path.join(self.plot_dir, "training_curves.png")
        plt.savefig(save_path, dpi=theme_registry.current_theme.get("dpi", 300))
        plt.close(fig)
        print(f"Saved training curves plot to {save_path}")

    def _plot_learning_rate(self, df: pd.DataFrame):
        """Plot the learning rate schedule over epochs."""
        if "epoch" not in df.columns or "learning_rate" not in df.columns:
            print("No epoch or learning_rate data to plot.")
            return

        fig, ax = plt.subplots(
            figsize=theme_registry.current_theme.get("figure_size", (10, 5))
        )
        theme_registry.style_plot(fig, ax)

        palette = theme_registry.get_palette(1)
        ax.plot(
            df["epoch"],
            df["learning_rate"],
            label="Learning Rate",
            color=palette[0],
            linewidth=2,
        )

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Learning Rate")
        ax.set_title("Learning Rate Schedule")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        # Use scientific notation if values are very small
        ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

        plt.tight_layout()
        save_path = os.path.join(self.plot_dir, "learning_rate_schedule.png")
        plt.savefig(save_path, dpi=theme_registry.current_theme.get("dpi", 300))
        plt.close(fig)
        print(f"Saved learning rate plot to {save_path}")

    def _plot_metric_correlations(self, df: pd.DataFrame):
        """Plot correlations between key metrics using the theme."""
        corr_metrics = [
            "train_loss",
            "val_loss",
            "val_expression_recognition_rate",
            "val_symbol_accuracy",
            "val_edit_distance",
            "val_normalized_edit_distance",
        ]
        available_metrics = [m for m in corr_metrics if m in df.columns]

        if len(available_metrics) < 2:
            return

        corr = df[available_metrics].corr()

        fig, ax = plt.subplots(
            figsize=theme_registry.current_theme.get("figure_size", (10, 8))
        )
        theme_registry.style_plot(fig, ax)  # Apply theme

        cmap = theme_registry.get_diverging_cmap()
        sns.heatmap(
            corr, annot=True, cmap=cmap, vmin=-1, vmax=1, center=0, ax=ax, fmt=".2f"
        )
        ax.set_title("Metric Correlations")
        plt.tight_layout()
        save_path = os.path.join(self.plot_dir, "metric_correlations.png")
        plt.savefig(save_path, dpi=theme_registry.current_theme.get("dpi", 300))
        plt.close(fig)
        print(f"Saved metric correlations plot to {save_path}")

    def _plot_error_analysis(self, error_analysis_file: Optional[str] = None):
        """Plot error analysis results from the latest JSON file."""
        if error_analysis_file is None:
            # Find the latest error analysis file if not provided
            error_files = glob.glob(
                os.path.join(self.error_analysis_dir, "error_analysis_epoch_*.json")
            )
            if not error_files:
                print("No error analysis JSON files found.")
                return
            # Sort by epoch number (assuming format '..._epoch_N.json')
            try:
                error_files.sort(
                    key=lambda f: int(
                        os.path.splitext(os.path.basename(f))[0].split("_")[-1]
                    )
                )
                error_analysis_file = error_files[-1]
            except (IndexError, ValueError):
                print("Could not determine the latest error analysis file.")
                return

        if not os.path.exists(error_analysis_file):
            print(f"Error analysis file not found: {error_analysis_file}")
            return

        try:
            with open(error_analysis_file, "r") as f:
                analysis_data = json.load(f)
        except json.JSONDecodeError:
            print(f"Error decoding JSON from {error_analysis_file}")
            return
        except Exception as e:
            print(f"Error reading error analysis file {error_analysis_file}: {e}")
            return

        # --- Extract Error Type Distribution ---
        # **ASSUMPTION:** Looking for a structure like: analysis_data['error_types'] = {'Error A': 10, 'Error B': 5}
        # Adjust this section based on the actual structure of your error_analysis.json
        error_types_data = analysis_data.get("error_types")
        if not isinstance(error_types_data, dict) or not error_types_data:
            # Fallback: Check if structure_slices has counts we can plot
            structure_errors = analysis_data.get("structure_slices", {})
            error_types_data = {
                "Unbalanced Delimiters": structure_errors.get("structure_errors", 0),
                "Frac Detection Mismatch": structure_errors.get(
                    "frac_detection_errors", 0
                ),
                # Add more based on what `analyze_errors` calculates if needed
            }
            # Filter out zero counts
            error_types_data = {k: v for k, v in error_types_data.items() if v > 0}

            if not error_types_data:
                print(
                    f"Could not find suitable error type data in {error_analysis_file}"
                )
                return

        error_labels = list(error_types_data.keys())
        error_counts = list(error_types_data.values())

        # --- Plotting ---
        fig, ax = plt.subplots(
            figsize=theme_registry.current_theme.get("figure_size", (10, 6))
        )
        theme_registry.style_plot(fig, ax)  # Apply theme

        palette = theme_registry.get_palette(len(error_labels))
        bars = ax.bar(error_labels, error_counts, color=palette)

        ax.set_xlabel("Error Type")
        ax.set_ylabel("Count")
        ax.set_title(
            f"Error Type Distribution ({os.path.basename(error_analysis_file)})"
        )
        ax.tick_params(axis="x", rotation=45)

        # Add counts on top of bars
        for bar in bars:
            yval = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                yval,
                int(yval),
                va="bottom",
                ha="center",
            )  # va='bottom' places text above bar

        # Ensure y-axis shows integer counts
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_ylim(bottom=0)  # Start y-axis at 0

        plt.tight_layout()
        save_path = os.path.join(self.plot_dir, "error_analysis_latest.png")
        plt.savefig(save_path, dpi=theme_registry.current_theme.get("dpi", 300))
        plt.close(fig)
        print(f"Saved error analysis plot to {save_path}")

    def _plot_validation_distribution(self, ned_scores_file: Optional[str] = None):
        """Plot the distribution of validation NED scores from the latest JSON file."""
        if ned_scores_file is None:
            # Find the latest NED scores file if not provided
            score_files = glob.glob(
                os.path.join(self.error_analysis_dir, "val_ned_scores_epoch_*.json")
            )
            if not score_files:
                print("No validation NED score JSON files found.")
                return
            try:
                score_files.sort(
                    key=lambda f: int(
                        os.path.splitext(os.path.basename(f))[0].split("_")[-1]
                    )
                )
                ned_scores_file = score_files[-1]
            except (IndexError, ValueError):
                print("Could not determine the latest NED scores file.")
                return

        if not os.path.exists(ned_scores_file):
            print(f"NED scores file not found: {ned_scores_file}")
            return

        try:
            with open(ned_scores_file, "r") as f:
                ned_scores = json.load(f)
            if not isinstance(ned_scores, list) or not ned_scores:
                print(f"No valid NED scores found in {ned_scores_file}")
                return
        except json.JSONDecodeError:
            print(f"Error decoding JSON from {ned_scores_file}")
            return
        except Exception as e:
            print(f"Error reading NED scores file {ned_scores_file}: {e}")
            return

        # --- Plotting Histogram ---
        fig, ax = plt.subplots(
            figsize=theme_registry.current_theme.get("figure_size", (10, 6))
        )
        theme_registry.style_plot(fig, ax)

        palette = theme_registry.get_palette(1, palette_type="sequential")
        # Use seaborn's histplot for better automatic binning
        sns.histplot(ned_scores, bins="auto", kde=True, ax=ax, color=palette[0])

        ax.set_xlabel("Normalized Edit Distance (NED)")
        ax.set_ylabel("Frequency")
        ax.set_title(
            f"Validation NED Distribution ({os.path.basename(ned_scores_file)})"
        )
        ax.grid(True, linestyle="--", alpha=0.6)
        # Set x-axis limits (optional, adjust as needed)
        ax.set_xlim(
            left=0, right=max(1.0, np.max(ned_scores) * 1.05) if ned_scores else 1.0
        )

        plt.tight_layout()
        save_path = os.path.join(self.plot_dir, "validation_ned_distribution.png")
        plt.savefig(save_path, dpi=theme_registry.current_theme.get("dpi", 300))
        plt.close(fig)
        print(f"Saved validation NED distribution plot to {save_path}")


def extract_from_wandb(wandb_dir: str, output_dir: str = "outputs/training_metrics"):
    """
    Extract metrics from wandb files and generate visualizations.

    Args:
        wandb_dir: Directory containing wandb files
        output_dir: Directory to save extracted metrics and visualizations
    """
    try:
        import wandb

        api = wandb.Api()
    except ImportError:
        print("Wandb not installed. Please install with: pip install wandb")
        return

    # Create training monitor
    monitor = TrainingMonitor(output_dir)

    print(f"Extracting metrics from wandb directory: {wandb_dir}")

    # Find all run directories
    run_dirs = []
    for root, dirs, files in os.walk(wandb_dir):
        for d in dirs:
            if "run-" in d:
                run_dirs.append(os.path.join(root, d))

    if not run_dirs:
        print("No wandb run directories found.")
        return

    print(f"Found {len(run_dirs)} wandb runs.")

    # Process each run
    for run_dir in run_dirs:
        # Find wandb run ID from directory name
        run_id = os.path.basename(run_dir).split("-")[-1]

        try:
            # Load run from wandb
            run = api.run(f"jdcl-umd/hmer-ink-fast/{run_id}")

            # Get history
            history = run.scan_history()

            # Extract metrics
            metrics_list = []
            for row in history:
                metrics_list.append(row)

            # Log to our monitor
            for metrics in metrics_list:
                # Convert from wandb types to Python native types
                clean_metrics = {}
                for k, v in metrics.items():
                    if hasattr(v, "item"):
                        clean_metrics[k] = v.item()
                    else:
                        clean_metrics[k] = v

                # Additionally convert any NumPy types that might still be present
                clean_metrics = convert_numpy_types(clean_metrics)

                # Extract error examples if available
                error_examples = None
                if "error_examples" in clean_metrics:
                    error_examples = clean_metrics.pop("error_examples")

                monitor.log_metrics(clean_metrics, error_examples)

            print(f"Processed run: {run_id}")

        except Exception as e:
            print(f"Error processing run {run_id}: {e}")

    print(f"Metrics extracted and visualized in: {output_dir}")


def watch_training(metrics_file: str, refresh_rate: int = 300):
    """
    Watch training progress and update visualizations periodically.

    Args:
        metrics_file: Path to JSON file containing metrics
        refresh_rate: How often to refresh visualizations (in seconds)
    """
    import time

    monitor = TrainingMonitor(os.path.dirname(metrics_file))

    print(f"Watching training metrics: {metrics_file}")
    print("Press Ctrl+C to stop")

    try:
        while True:
            # Load metrics and regenerate visualizations
            if os.path.exists(metrics_file):
                try:
                    with open(metrics_file, "r") as f:
                        metrics_history = json.load(f)

                    monitor.metrics_history = metrics_history
                    monitor.generate_dashboard()
                    print(
                        f"Updated visualizations at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                    )
                except Exception as e:
                    print(f"Error updating visualizations: {e}")

            # Wait for next refresh
            time.sleep(refresh_rate)

    except KeyboardInterrupt:
        print("Stopping watch mode.")


def capture_training_metrics(
    metrics: Dict,
    error_examples: Optional[List[Dict]] = None,
    error_analysis_data: Optional[Dict] = None,
    val_ned_scores_path: Optional[str] = None,
    output_dir: str = "outputs/training_metrics",
):
    """
    Logs training metrics and optionally saves and plots error analysis/distributions.

    Args:
        metrics: Dictionary of metrics from the current epoch/step.
        error_examples: Optional list of dictionaries containing error examples.
        error_analysis_data: Optional dictionary containing results from analyze_errors.
        val_ned_scores_path: Optional path to the saved validation NED scores JSON file.
        output_dir: The base directory for saving metrics and plots.
    """
    monitor = TrainingMonitor(log_dir=output_dir)

    error_analysis_file = None
    if error_analysis_data is not None:
        epoch = metrics.get("epoch", "unknown")
        error_analysis_path = os.path.join(
            output_dir, f"error_analysis_epoch_{epoch}.json"
        )
        try:
            error_analysis_data = convert_numpy_types(error_analysis_data)
            with open(error_analysis_path, "w") as f:
                json.dump(error_analysis_data, f, indent=2)
            print(f"Saved error analysis data to {error_analysis_path}")
            error_analysis_file = error_analysis_path
        except Exception as e:
            print(f"Error saving error analysis data: {e}")

    # Log metrics and trigger dashboard update, passing all paths
    monitor.log_metrics(
        metrics, error_examples, error_analysis_file, val_ned_scores_path
    )


def create_dashboard(log_dir: str = "outputs/training_metrics"):
    """Generates the dashboard from existing metrics data."""
    monitor = TrainingMonitor(log_dir=log_dir)
    # generate_dashboard will find the latest files automatically if paths are None
    monitor.generate_dashboard(error_analysis_file=None, val_ned_scores_path=None)


if __name__ == "__main__":
    # Example usage: Generate dashboard from existing logs
    import argparse

    parser = argparse.ArgumentParser(description="Generate Training Dashboard")
    parser.add_argument(
        "--log-dir",
        type=str,
        default="outputs/hmer-ink-v1/metrics",  # Example path
        help="Directory containing training_metrics.json",
    )
    args = parser.parse_args()

    if not os.path.exists(os.path.join(args.log_dir, "training_metrics.json")):
        print(f"Error: Metrics file not found in {args.log_dir}")
    else:
        create_dashboard(log_dir=args.log_dir)
        print(f"Dashboard generated in {os.path.join(args.log_dir, 'plots')}")
