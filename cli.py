"""
Command-line interface for HMER-Ink.
"""

import os
import sys
from typing import Optional

import typer

# Create Typer app with subcommands
app = typer.Typer(help="HMER-Ink: Handwritten Mathematical Expression Recognition")
monitor_app = typer.Typer(help="Monitor training metrics")

# Register subcommands
app.add_typer(monitor_app, name="monitor", help="Monitor and visualize training metrics")

# Add import path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))


@app.command("train")
def train_command(
    config: str = typer.Option(
        "configs/default.yaml", "--config", "-c", help="Path to configuration file"
    ),
    checkpoint: Optional[str] = typer.Option(
        None, "--checkpoint", "-ckpt", help="Path to checkpoint for resuming training"
    ),
    output_dir: Optional[str] = typer.Option(
        None, "--output-dir", "-o", help="Directory to save outputs"
    ),
):
    """Train a HMER model."""
    from scripts.train import train

    typer.echo(f"Training model with config from {config}")
    if checkpoint:
        typer.echo(f"Resuming from checkpoint {checkpoint}")
    if output_dir:
        typer.echo(f"Saving outputs to {output_dir}")

    model = train(config, checkpoint, output_dir)
    
    # Print location of metrics file if metrics recording is enabled
    config_data = train.__globals__["load_config"](config)
    if config_data["output"].get("record_metrics", False):
        metrics_dir = config_data["output"].get("metrics_dir", "outputs/training_metrics")
        metrics_file = os.path.join(metrics_dir, "training_metrics.json")
        typer.echo(f"Training metrics recorded to {metrics_file}")
        typer.echo(f"Use 'make watch-training' or 'make dashboard' to view metrics")

    typer.echo("Training completed")


@app.command("evaluate")
def evaluate_command(
    model: str = typer.Option(..., "--model", "-m", help="Path to model checkpoint"),
    config: str = typer.Option(
        "configs/default.yaml", "--config", "-c", help="Path to configuration file"
    ),
    output: Optional[str] = typer.Option(
        None, "--output", "-o", help="Path to save evaluation results"
    ),
    split: str = typer.Option(
        "test", "--split", "-s", help="Data split to evaluate on"
    ),
    beam_size: int = typer.Option(
        4, "--beam-size", "-b", help="Beam size for generation"
    ),
    batch_size: int = typer.Option(
        16, "--batch-size", help="Batch size for evaluation"
    ),
):
    """Evaluate a trained HMER model."""
    from scripts.evaluate import evaluate

    typer.echo(f"Evaluating model {model} on {split} split")

    metrics = evaluate(model, config, output, split, beam_size, batch_size)

    # Print metrics
    typer.echo("Evaluation results:")
    for key, value in metrics.items():
        typer.echo(f"  {key}: {value:.4f}")


@app.command("predict")
def predict_command(
    model: str = typer.Option(..., "--model", "-m", help="Path to model checkpoint"),
    inkml_file: str = typer.Option(..., "--input", "-i", help="Path to InkML file"),
    config: str = typer.Option(
        "configs/default.yaml", "--config", "-c", help="Path to configuration file"
    ),
    beam_size: int = typer.Option(
        4, "--beam-size", "-b", help="Beam size for generation"
    ),
    visualize: bool = typer.Option(
        False, "--visualize", "-v", help="Visualize input and prediction"
    ),
):
    """Make a prediction for a single InkML file."""
    import torch

    from hmer.config import get_device, load_config
    from hmer.data.inkml import InkmlParser
    from hmer.models import HMERModel
    from hmer.utils.tokenizer import LaTeXTokenizer

    # Load configuration
    config_data = load_config(config)

    # Get device
    device = get_device(config_data["training"])

    # Load model
    typer.echo(f"Loading model from {model}")
    model_instance, _ = HMERModel.load_checkpoint(model, map_location=device)
    model_instance = model_instance.to(device)
    model_instance.eval()

    # Load tokenizer
    tokenizer_path = os.path.join(os.path.dirname(model), "vocab.json")
    if not os.path.exists(tokenizer_path):
        tokenizer_path = os.path.join(
            config_data["output"].get("checkpoint_dir", "outputs/checkpoints"),
            "vocab.json",
        )

    tokenizer = LaTeXTokenizer()
    tokenizer.load_vocab(tokenizer_path)

    # Parse InkML file
    parser = InkmlParser()
    ink_data = parser.parse_inkml(inkml_file)

    # Get strokes
    strokes = ink_data["strokes"]

    # Get normalization parameters
    data_config = config_data["data"]
    x_range = data_config.get("normalization", {}).get("x_range", (-1, 1))
    y_range = data_config.get("normalization", {}).get("y_range", (-1, 1))
    time_range = data_config.get("normalization", {}).get("time_range", (0, 1))

    # Normalize strokes
    normalized_strokes = parser.normalize_strokes(
        strokes, x_range=x_range, y_range=y_range, time_range=time_range
    )

    # Convert to relative coordinates
    relative_strokes = parser.get_relative_coordinates(normalized_strokes)

    # Flatten strokes
    points = parser.flatten_strokes(relative_strokes)

    # Convert to tensor
    input_seq = torch.tensor(points, dtype=torch.float32).unsqueeze(0).to(device)
    input_lengths = torch.tensor([len(points)], dtype=torch.long).to(device)

    # Generate prediction
    with torch.no_grad():
        beam_results, _ = model_instance.generate(
            input_seq, input_lengths, max_length=128, beam_size=beam_size
        )

        # Decode prediction
        prediction = tokenizer.decode(beam_results[0][0], skip_special_tokens=True)

    # Get ground truth if available
    ground_truth = ink_data.get("normalized_label", "") or ink_data.get("label", "")

    # Print results
    typer.echo(f"Input file: {inkml_file}")
    if ground_truth:
        typer.echo(f"Ground truth: {ground_truth}")
    typer.echo(f"Prediction: {prediction}")

    # Visualize if requested
    if visualize:
        try:
            import matplotlib.pyplot as plt

            # numpy imported but unused
            from matplotlib.backends.backend_pdf import PdfPages

            # Create figure
            fig, ax = plt.subplots(figsize=(10, 6))

            # Plot strokes
            for stroke in strokes:
                ax.plot(stroke[:, 0], -stroke[:, 1], "b-")
                ax.scatter(
                    stroke[0, 0], -stroke[0, 1], color="green", s=20
                )  # Start point
                ax.scatter(
                    stroke[-1, 0], -stroke[-1, 1], color="red", s=20
                )  # End point

            # Set title
            title = f"Prediction: {prediction}"
            if ground_truth:
                title += f"\nGround truth: {ground_truth}"
            ax.set_title(title)

            # Remove axes
            ax.set_axis_off()

            # Save or show
            output_path = os.path.splitext(inkml_file)[0] + "_prediction.pdf"
            with PdfPages(output_path) as pdf:
                pdf.savefig(fig)

            typer.echo(f"Visualization saved to {output_path}")
            plt.close(fig)

        except ImportError:
            typer.echo(
                "Visualization requires matplotlib. Install it with 'pip install matplotlib'"
            )


@app.command("visualize")
def visualize_command(
    inkml_file: str = typer.Option(..., "--input", "-i", help="Path to InkML file"),
    output: Optional[str] = typer.Option(
        None, "--output", "-o", help="Path to save visualization"
    ),
    show: bool = typer.Option(False, "--show", help="Show visualization"),
):
    """Visualize an InkML file."""
    from hmer.data.inkml import InkmlParser

    try:
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
    except ImportError:
        typer.echo(
            "Visualization requires matplotlib. Install it with 'pip install matplotlib'"
        )
        return

    # Parse InkML file
    parser = InkmlParser()
    ink_data = parser.parse_inkml(inkml_file)

    # Get strokes
    strokes = ink_data["strokes"]

    # Get label
    label = ink_data.get("normalized_label", "") or ink_data.get("label", "")

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot strokes with different colors
    import numpy as np

    colors = plt.cm.jet(np.linspace(0, 1, len(strokes)))

    for i, stroke in enumerate(strokes):
        ax.plot(stroke[:, 0], -stroke[:, 1], color=colors[i])
        ax.scatter(stroke[0, 0], -stroke[0, 1], color="green", s=20)  # Start point
        ax.scatter(stroke[-1, 0], -stroke[-1, 1], color="red", s=20)  # End point

    # Set title
    if label:
        ax.set_title(f"Label: {label}")

    # Remove axes
    ax.set_axis_off()

    # Save if output path is provided
    if output:
        with PdfPages(output) as pdf:
            pdf.savefig(fig)
        typer.echo(f"Visualization saved to {output}")

    # Show if requested
    if show:
        plt.show()

    plt.close(fig)


@monitor_app.command("extract")
def extract_metrics_command(
    wandb_dir: str = typer.Option(
        "wandb", "--wandb-dir", "-w", help="Directory containing wandb files"
    ),
    output_dir: str = typer.Option(
        "outputs/training_metrics",
        "--output-dir",
        "-o",
        help="Directory to save extracted metrics and visualizations",
    ),
):
    """Extract training metrics from wandb and generate visualizations."""
    from scripts.training_monitor import extract_from_wandb

    typer.echo(f"Extracting training metrics from {wandb_dir} to {output_dir}")
    extract_from_wandb(wandb_dir, output_dir)


@monitor_app.command("watch")
def watch_metrics_command(
    metrics_file: str = typer.Option(
        "outputs/training_metrics/training_metrics.json",
        "--metrics-file",
        "-m",
        help="Path to metrics JSON file",
    ),
    refresh_rate: int = typer.Option(
        300, "--refresh", "-r", help="Refresh rate in seconds"
    ),
):
    """Watch training metrics and update visualizations periodically."""
    from scripts.training_monitor import watch_training

    typer.echo(f"Watching training metrics in {metrics_file}")
    typer.echo("Press Ctrl+C to stop")
    watch_training(metrics_file, refresh_rate)


@monitor_app.command("dashboard")
def dashboard_command(
    metrics_dir: str = typer.Option(
        "outputs/training_metrics",
        "--metrics-dir",
        "-d",
        help="Directory containing training metrics",
    ),
    port: int = typer.Option(8501, "--port", "-p", help="Port to run dashboard on"),
):
    """Launch an interactive dashboard for monitoring training."""
    try:
        import streamlit.web.cli as stcli
        import sys
        from scripts.training_monitor import create_dashboard

        # Create or update the dashboard script
        dashboard_path = create_dashboard()
        
        # Launch the dashboard with streamlit
        typer.echo(f"Launching dashboard for {metrics_dir} on port {port}")
        typer.echo("Press Ctrl+C to stop the dashboard")
        
        # We need to set sys.argv for streamlit
        sys.argv = [
            "streamlit", 
            "run", 
            dashboard_path,
            "--server.port", str(port),
            "--",  # Arguments after this go to the dashboard script
            "--metrics-dir", metrics_dir
        ]
        
        # Run streamlit
        sys.exit(stcli.main())
    except ImportError:
        typer.echo("Required dependencies not found. Install with:")
        typer.echo("  pip install streamlit altair")
        sys.exit(1)


if __name__ == "__main__":
    app()
