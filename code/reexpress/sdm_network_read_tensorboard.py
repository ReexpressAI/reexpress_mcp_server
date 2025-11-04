# Copyright Reexpress AI, Inc. All rights reserved.

"""
Command-line utility to read and analyze TensorBoard logs without the web interface.
"""

import argparse
import os
import json
import glob
from collections import defaultdict
from typing import Dict, List, Tuple
import warnings

warnings.filterwarnings('ignore')

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("Warning: tensorboard not installed. Install with: pip install tensorboard")

try:
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


def find_event_files(log_dir: str) -> List[str]:
    """Find all TensorBoard event files in a directory."""
    pattern = os.path.join(log_dir, "**", "events.out.tfevents.*")
    files = glob.glob(pattern, recursive=True)
    return sorted(files)


def read_tensorboard_logs(log_dir: str, tags: List[str] = None) -> Dict:
    """
    Read TensorBoard logs and extract metrics.

    Args:
        log_dir: Directory containing TensorBoard logs
        tags: Specific tags to extract (None = all tags)

    Returns:
        Dictionary with metrics data
    """
    if not TENSORBOARD_AVAILABLE:
        return {}

    event_files = find_event_files(log_dir)
    if not event_files:
        print(f"No event files found in {log_dir}")
        return {}

    # Use the most recent event file
    event_file = event_files[-1]
    print(f"Reading: {event_file}")

    # Load the event file
    ea = EventAccumulator(event_file)
    ea.Reload()

    # Get available tags
    available_tags = ea.Tags()

    metrics = {}

    # Extract scalar metrics
    scalar_tags = available_tags.get('scalars', [])
    if tags:
        scalar_tags = [t for t in scalar_tags if t in tags]

    for tag in scalar_tags:
        events = ea.Scalars(tag)
        metrics[tag] = {
            'steps': [e.step for e in events],
            'values': [e.value for e in events],
            'wall_times': [e.wall_time for e in events]
        }

    return metrics


def print_summary(metrics: Dict, last_n: int = 10):
    """Print a summary of the metrics."""

    if not metrics:
        print("No metrics found")
        return

    print("\n" + "=" * 80)
    print("TENSORBOARD METRICS SUMMARY")
    print("=" * 80)

    for tag, data in metrics.items():
        steps = data['steps']
        values = data['values']

        if not values:
            continue

        print(f"\n{tag}:")
        print(f"  Total points: {len(values)}")
        print(f"  First value: {values[0]:.6f} (step {steps[0]})")
        print(f"  Last value: {values[-1]:.6f} (step {steps[-1]})")
        print(f"  Min value: {min(values):.6f}")
        print(f"  Max value: {max(values):.6f}")
        print(f"  Average: {sum(values) / len(values):.6f}")

        if len(values) > last_n:
            print(f"  Last {last_n} values:")
            for i in range(-last_n, 0):
                print(f"    Step {steps[i]:5d}: {values[i]:.6f}")


def print_detailed(metrics: Dict, tag: str = None):
    """Print detailed values for specific metrics."""

    if not metrics:
        print("No metrics found")
        return

    if tag and tag in metrics:
        # Print specific metric
        data = metrics[tag]
        print(f"\n{tag} (all values):")
        print("-" * 40)
        for step, value in zip(data['steps'], data['values']):
            print(f"Step {step:6d}: {value:.6f}")
    else:
        # Print all available tags
        print("\nAvailable metrics:")
        for i, tag in enumerate(metrics.keys(), 1):
            num_points = len(metrics[tag]['values'])
            print(f"  {i}. {tag} ({num_points} points)")


def export_to_json(metrics: Dict, output_file: str):
    """Export metrics to JSON file."""

    # Convert to serializable format
    export_data = {}
    for tag, data in metrics.items():
        export_data[tag] = {
            'steps': data['steps'],
            'values': data['values']
        }

    with open(output_file, 'w') as f:
        json.dump(export_data, f, indent=2)

    print(f"Metrics exported to {output_file}")


def export_to_csv(metrics: Dict, output_file: str):
    """Export metrics to CSV file."""

    if not PANDAS_AVAILABLE:
        print("pandas not installed. Install with: pip install pandas")
        return

    # Create a DataFrame
    all_data = []
    for tag, data in metrics.items():
        for step, value in zip(data['steps'], data['values']):
            all_data.append({
                'metric': tag,
                'step': step,
                'value': value
            })

    df = pd.DataFrame(all_data)
    df.to_csv(output_file, index=False)
    print(f"Metrics exported to {output_file}")


def plot_metrics(metrics: Dict, output_file: str = None):
    """Plot metrics and save to file."""

    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib not installed. Install with: pip install matplotlib")
        return

    # Filter for loss metrics
    loss_metrics = {k: v for k, v in metrics.items()
                    if 'loss' in k.lower()}

    if not loss_metrics:
        print("No loss metrics found to plot")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    for tag, data in loss_metrics.items():
        ax.plot(data['steps'], data['values'], label=tag, marker='.')

    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.set_title('Training Metrics')
    ax.legend()
    ax.grid(True, alpha=0.3)

    if output_file:
        plt.savefig(output_file, dpi=100, bbox_inches='tight')
        print(f"Plot saved to {output_file}")
    else:
        plt.show()


def compare_runs(log_dirs: List[str], metric: str = "loss"):
    """Compare metrics across multiple training runs."""

    print(f"\nComparing '{metric}' across {len(log_dirs)} runs:")
    print("-" * 60)

    for log_dir in log_dirs:
        metrics = read_tensorboard_logs(log_dir, [metric])

        if metric in metrics:
            values = metrics[metric]['values']
            steps = metrics[metric]['steps']

            if values:
                print(f"\n{os.path.basename(log_dir)}:")
                print(f"  Final value: {values[-1]:.6f} (step {steps[-1]})")
                print(f"  Min value: {min(values):.6f}")
                print(f"  Average: {sum(values) / len(values):.6f}")
        else:
            print(f"\n{os.path.basename(log_dir)}: No '{metric}' found")


def watch_live(log_dir: str, interval: int = 10):
    """Watch metrics in real-time (like tail -f)."""

    import time

    print(f"Watching {log_dir} (Ctrl+C to stop)...")
    print("-" * 60)

    last_step = {}

    try:
        while True:
            metrics = read_tensorboard_logs(log_dir)

            for tag in ['loss', 'eval_loss', 'learning_rate']:
                if tag in metrics and metrics[tag]['values']:
                    current_step = metrics[tag]['steps'][-1]
                    current_value = metrics[tag]['values'][-1]

                    if tag not in last_step or last_step[tag] != current_step:
                        print(f"[Step {current_step:5d}] {tag}: {current_value:.6f}")
                        last_step[tag] = current_step

            time.sleep(interval)

    except KeyboardInterrupt:
        print("\nStopped watching")


def main():
    parser = argparse.ArgumentParser(description="Read TensorBoard logs from command line")

    # Mode selection
    parser.add_argument("log_dir", type=str, nargs='?',
                        help="TensorBoard log directory")

    # Commands
    parser.add_argument("--summary", action="store_true",
                        help="Print summary statistics")
    parser.add_argument("--detailed", type=str, metavar="METRIC",
                        help="Print all values for specific metric")
    parser.add_argument("--list", action="store_true",
                        help="List all available metrics")
    parser.add_argument("--last", type=int, default=10,
                        help="Number of recent values to show (default: 10)")

    # Export options
    parser.add_argument("--export-json", type=str, metavar="FILE",
                        help="Export metrics to JSON file")
    parser.add_argument("--export-csv", type=str, metavar="FILE",
                        help="Export metrics to CSV file")
    parser.add_argument("--plot", type=str, nargs='?', const=True, metavar="FILE",
                        help="Plot metrics (optionally save to file)")

    # Advanced options
    parser.add_argument("--compare", type=str, nargs='+', metavar="DIR",
                        help="Compare metrics across multiple runs")
    parser.add_argument("--metric", type=str, default="loss",
                        help="Metric to compare (default: loss)")
    parser.add_argument("--watch", action="store_true",
                        help="Watch metrics in real-time")
    parser.add_argument("--interval", type=int, default=10,
                        help="Watch interval in seconds (default: 10)")

    args = parser.parse_args()

    # Handle compare mode
    if args.compare:
        compare_runs(args.compare, args.metric)
        return

    # Require log_dir for other operations
    if not args.log_dir:
        parser.error("log_dir is required (except for --compare)")

    # Read metrics
    metrics = read_tensorboard_logs(args.log_dir)

    if not metrics:
        print("No metrics found. Check the log directory path.")
        return

    # Handle different modes
    if args.watch:
        watch_live(args.log_dir, args.interval)
    elif args.list:
        print_detailed(metrics)
    elif args.detailed:
        print_detailed(metrics, args.detailed)
    elif args.export_json:
        export_to_json(metrics, args.export_json)
    elif args.export_csv:
        export_to_csv(metrics, args.export_csv)
    elif args.plot:
        output_file = args.plot if isinstance(args.plot, str) else None
        plot_metrics(metrics, output_file)
    else:
        # Default: show summary
        print_summary(metrics, args.last)


if __name__ == "__main__":
    main()
