"""
Generate a comprehensive report of model performance and visualizations.
"""

import json
import os
import random
import sys
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from hmer.config import get_device, load_config
from hmer.data.dataset import HMERDataset
from hmer.data.inkml import InkmlParser
from hmer.models import HMERModel
from hmer.utils.tokenizer import LaTeXTokenizer


def generate_report(
    model_path: str, config_path: str, output_path: str, num_samples: int = 20
):
    """
    Generate a comprehensive HTML report for model evaluation.

    Args:
        model_path: Path to model checkpoint
        config_path: Path to configuration file
        output_path: Path to save report HTML
        num_samples: Number of random samples to include in report
    """
    try:
        from jinja2 import Template
    except ImportError:
        print("Please install jinja2: pip install jinja2")
        return

    # Load configuration
    config = load_config(config_path)

    # Get device
    device = get_device(config["training"])
    print(f"Using device: {device}")

    # Load model
    print(f"Loading model from {model_path}")
    model, checkpoint = HMERModel.load_checkpoint(model_path, map_location=device)
    model = model.to(device)
    model.eval()

    # Load tokenizer
    tokenizer_path = os.path.join(os.path.dirname(model_path), "vocab.json")
    if not os.path.exists(tokenizer_path):
        tokenizer_path = os.path.join(
            config["output"].get("checkpoint_dir", "outputs/checkpoints"), "vocab.json"
        )

    if not os.path.exists(tokenizer_path):
        raise ValueError(f"Tokenizer not found at {tokenizer_path}")

    tokenizer = LaTeXTokenizer()
    tokenizer.load_vocab(tokenizer_path)
    print(f"Loaded tokenizer with vocabulary size {len(tokenizer)}")

    # Load test dataset
    data_dir = config["data"].get("data_dir", "data")
    split = "test"

    # Get normalization parameters
    normalize = True
    data_config = config["data"]
    x_range = data_config.get("normalization", {}).get("x_range", (-1, 1))
    y_range = data_config.get("normalization", {}).get("y_range", (-1, 1))
    time_range = data_config.get("normalization", {}).get("time_range", (0, 1))

    # Create test dataset
    test_dataset = HMERDataset(
        data_dir=data_dir,
        split_dirs=[split],
        tokenizer=tokenizer,
        max_seq_length=config["data"].get("max_seq_length", 512),
        max_token_length=config["data"].get("max_token_length", 128),
        transform=None,
        normalize=normalize,
        use_relative_coords=True,
        x_range=x_range,
        y_range=y_range,
        time_range=time_range,
    )

    print(f"Created dataset with {len(test_dataset)} samples")

    # Try to load existing evaluation results
    eval_results_path = os.path.join(
        os.path.dirname(model_path), "evaluation_test.json"
    )
    evaluation_data = None
    metrics = {}

    if os.path.exists(eval_results_path):
        print(f"Loading evaluation results from {eval_results_path}")
        try:
            with open(eval_results_path, "r") as f:
                evaluation_data = json.load(f)
            metrics = evaluation_data.get("metrics", {})
        except Exception as e:
            print(f"Error loading evaluation results: {e}")

    # If evaluation results not available, run basic metrics calculation
    if not metrics:
        print("No pre-computed metrics found. Computing basic metrics...")

        # Sample some examples to calculate metrics
        sample_indices = random.sample(
            range(len(test_dataset)), min(100, len(test_dataset))
        )
        predictions = []
        targets = []

        with torch.no_grad():
            for idx in tqdm(sample_indices):
                sample = test_dataset[idx]
                input_seq = sample["input"].unsqueeze(0).to(device)
                input_lengths = torch.tensor([input_seq.shape[1]], dtype=torch.long).to(
                    device
                )

                # Generate prediction
                beam_results, _ = model.generate(
                    input_seq, input_lengths, max_length=128, beam_size=4
                )

                # Decode prediction
                prediction = tokenizer.decode(
                    beam_results[0][0], skip_special_tokens=True
                )
                target = sample["label"]

                predictions.append(prediction)
                targets.append(target)

        # Calculate metrics
        from hmer.utils.metrics import compute_metrics

        metrics = compute_metrics(predictions, targets)

    # Sample examples for visualization
    print("Sampling examples for visualization...")
    sample_indices = random.sample(
        range(len(test_dataset)), min(num_samples, len(test_dataset))
    )
    sample_examples = []

    with torch.no_grad():
        for idx in tqdm(sample_indices):
            sample = test_dataset[idx]
            file_id = sample["file_id"]
            label = sample["label"]

            # Generate prediction
            input_seq = sample["input"].unsqueeze(0).to(device)
            input_lengths = torch.tensor([input_seq.shape[1]], dtype=torch.long).to(
                device
            )

            beam_results, _ = model.generate(
                input_seq, input_lengths, max_length=128, beam_size=4
            )

            # Decode prediction
            prediction = tokenizer.decode(beam_results[0][0], skip_special_tokens=True)

            # Get original ink file path
            ink_file_path = None
            for dir_name in [split, "test", "valid", "train", "synthetic", "symbols"]:
                possible_path = os.path.join(data_dir, dir_name, f"{file_id}.inkml")
                if os.path.exists(possible_path):
                    ink_file_path = possible_path
                    break

            if not ink_file_path:
                print(f"Warning: InkML file not found for {file_id}")
                continue

            # Parse original ink file to get strokes
            parser = InkmlParser()
            ink_data = parser.parse_inkml(ink_file_path)
            strokes = ink_data["strokes"]

            # Generate visualization
            fig = plt.figure(figsize=(6, 4))
            for stroke in strokes:
                plt.plot(stroke[:, 0], -stroke[:, 1], "b-")
                plt.scatter(
                    stroke[0, 0], -stroke[0, 1], color="green", s=20
                )  # Start point
                plt.scatter(
                    stroke[-1, 0], -stroke[-1, 1], color="red", s=20
                )  # End point

            plt.axis("off")
            plt.tight_layout()

            # Save figure to a file
            fig_dir = os.path.join(os.path.dirname(output_path), "figures")
            os.makedirs(fig_dir, exist_ok=True)
            fig_path = os.path.join(fig_dir, f"{file_id}.png")
            plt.savefig(fig_path, dpi=100, bbox_inches="tight")
            plt.close(fig)

            # Calculate correctness
            correct = prediction == label

            # Add to samples
            sample_examples.append(
                {
                    "file_id": file_id,
                    "image_path": fig_path,
                    "target": label,
                    "prediction": prediction,
                    "correct": correct,
                }
            )

    # Generate confusion matrix for common symbols
    confusion_data = {}
    if evaluation_data and "results" in evaluation_data:
        from collections import Counter

        # Tokenize predictions and targets
        def simple_tokenize(expr):
            import re

            # Extract LaTeX commands and symbols
            tokens = re.findall(r"\\[a-zA-Z]+|[^\\a-zA-Z0-9\s]|[a-zA-Z0-9]", expr)
            return tokens

        all_target_tokens = []
        all_pred_tokens = []

        for result in evaluation_data["results"]:
            target_tokens = simple_tokenize(result["target"])
            pred_tokens = simple_tokenize(result["prediction"])

            all_target_tokens.extend(target_tokens)
            all_pred_tokens.extend(pred_tokens)

        # Find most common symbols
        target_counter = Counter(all_target_tokens)
        most_common = [token for token, _ in target_counter.most_common(30)]

        # Create confusion matrix data
        confusion_labels = []
        confusion_values = []

        for i, token in enumerate(most_common):
            row = []
            for j, other_token in enumerate(most_common):
                # Count how many times token was predicted as other_token
                count = 0
                for result in evaluation_data["results"]:
                    target_tokens = simple_tokenize(result["target"])
                    pred_tokens = simple_tokenize(result["prediction"])

                    # Find matching positions
                    for k, t in enumerate(target_tokens):
                        if t == token and k < len(pred_tokens):
                            if pred_tokens[k] == other_token:
                                count += 1

                row.append(count)
            confusion_values.append(row)
            confusion_labels.append(token)

        confusion_data = {"labels": confusion_labels, "values": confusion_values}

    # Prepare template data
    template_data = {
        "title": "HMER-Ink Model Evaluation Report",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_path": model_path,
        "metrics": metrics,
        "examples": sample_examples,
        "confusion_data": confusion_data,
        "config": config,
    }

    # Load HTML template
    template_string = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>{{ title }}</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                line-height: 1.6;
                margin: 0;
                padding: 20px;
                color: #333;
            }
            h1, h2, h3 {
                color: #2c3e50;
            }
            .header {
                background-color: #f8f9fa;
                padding: 20px;
                margin-bottom: 20px;
                border-radius: 5px;
            }
            .metrics {
                display: flex;
                flex-wrap: wrap;
                gap: 20px;
                margin-bottom: 30px;
            }
            .metric-card {
                background-color: #fff;
                border: 1px solid #ddd;
                border-radius: 5px;
                padding: 15px;
                width: 200px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .metric-card h3 {
                margin-top: 0;
                color: #7f8c8d;
                font-size: 14px;
                text-transform: uppercase;
            }
            .metric-value {
                font-size: 24px;
                font-weight: bold;
                color: #2980b9;
            }
            .examples {
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(400px, 1fr));
                gap: 20px;
                margin-bottom: 30px;
            }
            .example-card {
                border: 1px solid #ddd;
                border-radius: 5px;
                overflow: hidden;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .example-card.correct {
                border-color: #27ae60;
            }
            .example-card.incorrect {
                border-color: #e74c3c;
            }
            .example-image {
                width: 100%;
                height: 200px;
                background-color: #f5f5f5;
                display: flex;
                align-items: center;
                justify-content: center;
            }
            .example-image img {
                max-width: 100%;
                max-height: 100%;
                object-fit: contain;
            }
            .example-details {
                padding: 15px;
            }
            .example-details p {
                margin: 5px 0;
            }
            .correct-prediction {
                color: #27ae60;
            }
            .incorrect-prediction {
                color: #e74c3c;
            }
            pre {
                background-color: #f5f5f5;
                padding: 10px;
                border-radius: 5px;
                overflow-x: auto;
            }
            table {
                border-collapse: collapse;
                width: 100%;
                margin-bottom: 20px;
            }
            th, td {
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
            }
            th {
                background-color: #f2f2f2;
            }
            tr:nth-child(even) {
                background-color: #f9f9f9;
            }
            .confusion-matrix {
                margin-top: 30px;
                overflow-x: auto;
            }
            .config-section {
                margin-top: 30px;
            }
            .footer {
                margin-top: 50px;
                text-align: center;
                color: #7f8c8d;
                font-size: 12px;
            }
        </style>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    </head>
    <body>
        <div class="header">
            <h1>{{ title }}</h1>
            <p>Generated on: {{ timestamp }}</p>
            <p>Model: {{ model_path }}</p>
        </div>
        
        <h2>Metrics</h2>
        <div class="metrics">
            {% for name, value in metrics.items() %}
            <div class="metric-card">
                <h3>{{ name|replace('_', ' ')|title }}</h3>
                <div class="metric-value">{{ "%.4f"|format(value) }}</div>
            </div>
            {% endfor %}
        </div>
        
        <h2>Example Predictions</h2>
        <p>Showing {{ examples|length }} random examples from the test set</p>
        <div class="examples">
            {% for example in examples %}
            <div class="example-card {{ 'correct' if example.correct else 'incorrect' }}">
                <div class="example-image">
                    <img src="{{ example.image_path }}" alt="Ink Sample">
                </div>
                <div class="example-details">
                    <p><strong>File ID:</strong> {{ example.file_id }}</p>
                    <p><strong>Target:</strong> <code>{{ example.target }}</code></p>
                    <p><strong>Prediction:</strong> 
                        <code class="{{ 'correct-prediction' if example.correct else 'incorrect-prediction' }}">
                            {{ example.prediction }}
                        </code>
                    </p>
                </div>
            </div>
            {% endfor %}
        </div>
        
        {% if confusion_data %}
        <h2>Symbol Confusion Analysis</h2>
        <div class="confusion-matrix">
            <canvas id="confusionMatrix" width="800" height="800"></canvas>
        </div>
        
        <script>
            const ctx = document.getElementById('confusionMatrix').getContext('2d');
            
            const labels = {{ confusion_data.labels|tojson }};
            const values = {{ confusion_data.values|tojson }};
            
            // Create data for heatmap
            const data = {
                labels: labels,
                datasets: [{
                    label: 'Confusion Matrix',
                    data: values.flat(),
                    backgroundColor: 'rgba(54, 162, 235, 0.2)',
                    borderColor: 'rgba(54, 162, 235, 1)',
                    borderWidth: 1
                }]
            };
            
            // Config for the chart
            const config = {
                type: 'heatmap',
                data: data,
                options: {
                    plugins: {
                        title: {
                            display: true,
                            text: 'Symbol Confusion Matrix'
                        },
                        tooltip: {
                            callbacks: {
                                title: function(tooltipItems) {
                                    const i = Math.floor(tooltipItems[0].dataIndex / labels.length);
                                    const j = tooltipItems[0].dataIndex % labels.length;
                                    return `Target: ${labels[i]}, Predicted: ${labels[j]}`;
                                },
                                label: function(tooltipItem) {
                                    return `Count: ${tooltipItem.raw}`;
                                }
                            }
                        }
                    },
                    scales: {
                        x: {
                            title: {
                                display: true,
                                text: 'Predicted'
                            }
                        },
                        y: {
                            title: {
                                display: true,
                                text: 'Actual'
                            }
                        }
                    }
                }
            };
            
            // Create the chart
            const myChart = new Chart(ctx, config);
        </script>
        {% endif %}
        
        <div class="config-section">
            <h2>Model Configuration</h2>
            <pre>{{ config|tojson(indent=2) }}</pre>
        </div>
        
        <div class="footer">
            <p>HMER-Ink - Handwritten Mathematical Expression Recognition</p>
        </div>
    </body>
    </html>
    """

    # Create Jinja2 template
    template = Template(template_string)

    # Render HTML
    html_content = template.render(**template_data)

    # Write HTML file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(html_content)

    print(f"Report generated at {output_path}")


if __name__ == "__main__":
    import typer
    
    def main(
        model: str = typer.Argument(..., help="Path to model checkpoint"),
        config: str = typer.Option(
            "configs/default.yaml", "--config", "-c", help="Path to configuration file"
        ),
        output: str = typer.Option(
            "outputs/report.html", "--output", "-o", help="Path to save report HTML"
        ),
        samples: int = typer.Option(
            20, "--samples", "-s", help="Number of random samples to include in report"
        ),
    ):
        """Generate a comprehensive HTML report for model evaluation."""
        generate_report(model, config, output, samples)
    
    typer.run(main)
