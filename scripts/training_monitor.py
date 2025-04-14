"""
Training Monitor for HMER-Ink that provides comprehensive visualization of training metrics.
"""

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
from matplotlib.gridspec import GridSpec


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
        self.metrics_history: List[Dict[str, Any]] = []
        self.error_examples_history: List[Dict[str, Any]] = []

        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(os.path.join(log_dir, "plots"), exist_ok=True)

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
    ):
        """
        Log training metrics.

        Args:
            metrics: Dictionary of metrics to log
            error_examples: List of error examples (optional)
        """
        # Add timestamp
        metrics["timestamp"] = datetime.now().isoformat()

        # Convert NumPy types to Python native types
        metrics = convert_numpy_types(metrics)

        # Add to history
        self.metrics_history.append(metrics)

        # Add error examples if provided
        if error_examples:
            # Convert NumPy types in error examples
            error_examples = convert_numpy_types(error_examples)
            self.error_examples_history.append(
                {"epoch": metrics.get("epoch", 0), "examples": error_examples}
            )

        # Save to file
        with open(self.metrics_file, "w") as f:
            json.dump(self.metrics_history, f, indent=2)

        # Generate visualizations
        self.generate_dashboard()

    def generate_dashboard(self):
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
        self._plot_metric_correlations(plot_df)
        self._create_summary_table(df)

        # If we have error examples, visualize them
        if self.error_examples_history:
            self._visualize_error_examples()

    def _plot_training_curves(self, df: pd.DataFrame):
        """Plot key training curves."""
        # Only use these if they exist
        possible_metrics = [
            # Main metrics
            ("loss", "val_loss"),
            ("val_accuracy", "val_exprate"),
            ("val_cer", "val_ter"),  # Error rates
        ]

        # Create a 2x2 grid of plots
        fig = plt.figure(figsize=(15, 10))
        gs = GridSpec(2, 2, figure=fig)

        # Plot main training curve (always plot if exists)
        if "loss" in df.columns or "val_loss" in df.columns:
            ax1 = fig.add_subplot(gs[0, :])
            if "loss" in df.columns:
                ax1.plot(df["epoch"], df["loss"], "b-", label="Training Loss")
            if "val_loss" in df.columns:
                ax1.plot(df["epoch"], df["val_loss"], "r-", label="Validation Loss")
            ax1.set_title("Training & Validation Loss")
            ax1.set_xlabel("Epoch")
            ax1.set_ylabel("Loss")
            ax1.legend()
            ax1.grid(True)

        # Plot additional metric pairs
        plot_idx = 0
        for i, (metric1, metric2) in enumerate(possible_metrics[1:]):
            if metric1 in df.columns or metric2 in df.columns:
                row, col = 1, plot_idx
                ax = fig.add_subplot(gs[row, col])

                if metric1 in df.columns:
                    ax.plot(df["epoch"], df[metric1], "g-", label=metric1)
                if metric2 in df.columns:
                    ax.plot(df["epoch"], df[metric2], "m-", label=metric2)

                ax.set_title(f"{metric1} & {metric2}")
                ax.set_xlabel("Epoch")
                ax.set_ylabel("Value")
                ax.legend()
                ax.grid(True)

                plot_idx += 1
                if plot_idx >= 2:  # We only have 2 slots in the bottom row
                    break

        plt.tight_layout()
        plt.savefig(
            os.path.join(self.log_dir, "plots", "training_curves.png"),
            dpi=100,
            bbox_inches="tight",
        )
        plt.close()

    def _plot_metric_correlations(self, df: pd.DataFrame):
        """Plot correlations between key metrics."""
        # Select relevant metrics for correlation
        corr_metrics = ["val_loss", "val_accuracy", "val_exprate", "val_cer", "val_ter"]
        available_metrics = [m for m in corr_metrics if m in df.columns]

        if len(available_metrics) < 2:
            return  # Need at least 2 metrics for correlation

        # Calculate correlation matrix
        corr = df[available_metrics].corr()

        # Plot correlation heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1, center=0)
        plt.title("Metric Correlations")
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.log_dir, "plots", "metric_correlations.png"),
            dpi=100,
            bbox_inches="tight",
        )
        plt.close()

    def _create_summary_table(self, df: pd.DataFrame):
        """Create a summary table of key metrics."""
        # Get the latest metrics and best metrics
        latest = df.iloc[-1].to_dict()

        # Find best values for each metric
        best = {}
        for col in df.columns:
            if col.startswith("val_"):
                if "loss" in col or "cer" in col or "ter" in col:
                    # Lower is better for these metrics
                    best[col] = df[col].min()
                else:
                    # Higher is better for accuracy, exprate
                    best[col] = df[col].max()

        # Create a summary table
        summary = {"Latest": latest, "Best": best}

        # Convert NumPy types to Python native types for JSON serialization
        summary = convert_numpy_types(summary)

        # Save as JSON
        with open(os.path.join(self.log_dir, "latest_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        # Also create a markdown table for easy viewing
        with open(os.path.join(self.log_dir, "latest_summary.md"), "w") as f:
            f.write("# Training Summary\n\n")
            f.write(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write(
                "## Latest Metrics (Epoch {})\n\n".format(latest.get("epoch", "N/A"))
            )
            f.write("| Metric | Value |\n")
            f.write("|--------|-------|\n")
            for k, v in latest.items():
                if k not in ["timestamp", "epoch"] and not isinstance(v, str):
                    f.write(f"| {k} | {v:.4f} |\n")

            f.write("\n## Best Metrics\n\n")
            f.write("| Metric | Best Value | Epoch |\n")
            f.write("|--------|------------|-------|\n")
            for k, v in best.items():
                # Find which epoch had this best value
                best_epoch = df[df[k] == v]["epoch"].iloc[0]
                f.write(f"| {k} | {v:.4f} | {best_epoch} |\n")

    def _visualize_error_examples(self):
        """Visualize error examples from the latest epoch."""
        if not self.error_examples_history:
            return

        # Get the latest error examples
        latest = self.error_examples_history[-1]
        examples = latest.get("examples", [])

        if not examples:
            return

        # Create a visualization of common error patterns
        error_types = {}

        for ex in examples:
            pred = ex.get("prediction", "")
            target = ex.get("target", "")

            # Simple error categorization
            if pred == target:
                category = "Correct"
            elif len(pred) == 0:
                category = "Empty Prediction"
            elif all(c == pred[0] for c in pred):
                category = "Repeating Character"
            elif len(pred) < len(target) / 2:
                category = "Too Short"
            elif len(pred) > len(target) * 2:
                category = "Too Long"
            else:
                category = "Other Error"

            if category not in error_types:
                error_types[category] = 0
            error_types[category] += 1

        # Plot error type distribution
        plt.figure(figsize=(10, 6))
        bars = plt.bar(error_types.keys(), error_types.values())
        plt.title(f"Error Type Distribution (Epoch {latest.get('epoch', 0)})")
        plt.xlabel("Error Type")
        plt.ylabel("Count")
        plt.xticks(rotation=45, ha="right")

        # Add counts on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.1,
                f"{height:.0f}",
                ha="center",
                va="bottom",
            )

        plt.tight_layout()
        plt.savefig(
            os.path.join(
                self.log_dir,
                "plots",
                f"error_analysis_epoch_{latest.get('epoch', 0)}.png",
            ),
            dpi=100,
            bbox_inches="tight",
        )
        plt.close()

        # Save the detailed examples
        # Convert any NumPy types before serialization
        examples = convert_numpy_types(examples)
        with open(
            os.path.join(
                self.log_dir, f"error_examples_epoch_{latest.get('epoch', 0)}.json"
            ),
            "w",
        ) as f:
            json.dump(examples, f, indent=2)


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
    output_dir: str = "outputs/training_metrics",
):
    """
    Utility function to call from training script to log metrics during training.

    Args:
        metrics: Dictionary of metrics
        error_examples: List of error examples
        output_dir: Directory to save metrics
    """
    monitor = TrainingMonitor(output_dir)
    monitor.log_metrics(metrics, error_examples)


def create_dashboard():
    """
    Create a Streamlit dashboard script for interactive monitoring.
    """
    dashboard_code = """
import os
import sys
import json
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
import typer

# Create Typer app
app = typer.Typer(help="HMER-Ink Training Dashboard")

@app.callback()
def main(
    metrics_dir: str = typer.Option(
        "outputs/training_metrics",
        "--metrics-dir",
        "-d",
        help="Directory containing training metrics",
    ),
):
    \"\"\"Dashboard for monitoring HMER-Ink training.\"\"\"
    # We only use this for argument parsing
    # All dashboard logic is in run_dashboard
    pass

def run_dashboard(metrics_dir: str = "outputs/training_metrics"):
    \"\"\"Run the dashboard with the given metrics directory.\"\"\"
    # Set page config
    st.set_page_config(
        page_title="HMER-Ink Training Monitor",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Title and sidebar
    st.title("HMER-Ink Training Monitor")
    st.sidebar.title("Controls")

    # Function to load metrics
    @st.cache_data(ttl=60)  # Cache for 60 seconds
    def load_metrics(metrics_file):
        try:
            with open(metrics_file, "r") as f:
                data = json.load(f)
            return pd.DataFrame(data)
        except Exception as e:
            st.error(f"Error loading metrics: {e}")
            return pd.DataFrame()

    # Path to metrics file
    metrics_file = os.path.join(metrics_dir, "training_metrics.json")
    metrics_exist = os.path.exists(metrics_file)

    # Sidebar auto-refresh
    auto_refresh = st.sidebar.checkbox("Auto-refresh", value=True)
    refresh_interval = st.sidebar.slider("Refresh interval (seconds)", 5, 300, 60)

    if auto_refresh:
        st.sidebar.write(f"Dashboard will refresh every {refresh_interval} seconds")
        st.empty()  # Placeholder for refresh

    # Load metrics
    if metrics_exist:
        df = load_metrics(metrics_file)
        
        if len(df) > 0:
            # Extract latest metrics
            latest = df.iloc[-1]
            
            # Create multicolumn layout
            col1, col2, col3 = st.columns(3)
            
            # Display latest metrics
            with col1:
                st.subheader("Latest Metrics")
                st.metric("Epoch", int(latest.get("epoch", 0)))
                st.metric("Training Loss", f"{latest.get('train_loss', 0):.4f}")
                
            with col2:
                st.subheader("Validation Metrics")
                for k, v in latest.items():
                    if k.startswith("val_") and k != "val_loss" and isinstance(v, (int, float)):
                        st.metric(k.replace("val_", "").replace("_", " ").title(), f"{v:.4f}")
                
            with col3:
                st.subheader("Training Stats")
                st.metric("Validation Loss", f"{latest.get('val_loss', 0):.4f}")
                st.metric("Samples Processed", int(latest.get("val_num_samples", 0)))
            
            # Plot training curves
            st.subheader("Training Progress")
            
            # Create interactive chart with Altair
            chart_data = df.reset_index()
            
            # Identify numeric columns for plotting
            numeric_cols = chart_data.select_dtypes(include=[np.number]).columns.tolist()
            
            # Metrics selection
            selected_metrics = st.multiselect(
                "Select metrics to display",
                options=[col for col in numeric_cols if col != "index" and col != "epoch"],
                default=["train_loss", "val_loss"] if "train_loss" in numeric_cols and "val_loss" in numeric_cols else [],
            )
            
            if selected_metrics:
                # Prepare data for Altair
                plot_data = pd.melt(
                    chart_data, 
                    id_vars=["epoch"],
                    value_vars=selected_metrics,
                    var_name="Metric",
                    value_name="Value"
                )
                
                # Create line chart
                chart = alt.Chart(plot_data).mark_line(point=True).encode(
                    x=alt.X("epoch:Q", title="Epoch"),
                    y=alt.Y("Value:Q", title="Value"),
                    color=alt.Color("Metric:N", title="Metric"),
                    tooltip=["epoch", "Metric", "Value"]
                ).interactive()
                
                st.altair_chart(chart, use_container_width=True)
                
            # Show error examples if available
            error_examples_files = [f for f in os.listdir(metrics_dir) if f.startswith("error_examples_epoch_") and f.endswith(".json")]
            
            if error_examples_files:
                st.subheader("Error Examples")
                
                # Sort by epoch and get the latest
                error_examples_files.sort(key=lambda f: int(f.split("_")[-1].split(".")[0]), reverse=True)
                latest_error_file = error_examples_files[0]
                
                try:
                    with open(os.path.join(metrics_dir, latest_error_file), "r") as f:
                        error_examples = json.load(f)
                    
                    if error_examples:
                        for i, example in enumerate(error_examples[:5]):  # Show up to 5 examples
                            with st.expander(f"Example {i+1} - CER: {example.get('cer', 'N/A')}"):
                                st.code(f"Target:     {example.get('target', '')}")
                                st.code(f"Prediction: {example.get('prediction', '')}")
                except Exception as e:
                    st.warning(f"Error loading error examples: {e}")
            
            # Raw data view
            with st.expander("View Raw Data"):
                st.dataframe(df)
        else:
            st.warning("No metrics data available yet.")
    else:
        st.warning(f"No metrics file found at {metrics_file}. Start training to generate metrics.")

    # Auto-refresh script
    if auto_refresh:
        st.markdown(f"<meta http-equiv='refresh' content='{refresh_interval}'>", unsafe_allow_html=True)
        st.sidebar.write(f"Last refreshed: {datetime.now().strftime('%H:%M:%S')}")

# This runs when directly executed
if __name__ == "__main__":
    typer.run(run_dashboard)
"""

    # Write the dashboard script
    dashboard_path = os.path.join("scripts", "dashboard.py")
    os.makedirs(os.path.dirname(dashboard_path), exist_ok=True)

    with open(dashboard_path, "w") as f:
        f.write(dashboard_code)

    return dashboard_path


if __name__ == "__main__":
    # This script is now meant to be used through the CLI interface
    # See cli.py for the command-line interface
    print(
        "This script is now integrated with the CLI. Use 'python cli.py monitor' instead."
    )
    print("For help, run: python cli.py monitor --help")
