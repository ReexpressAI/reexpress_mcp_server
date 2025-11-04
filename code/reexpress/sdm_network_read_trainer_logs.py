# Copyright Reexpress AI, Inc. All rights reserved.

"""
Simple command-line utility to read HuggingFace Trainer logs without TensorBoard.
Reads trainer_state.json files directly.
"""

import argparse
import json
import os
import glob
from typing import Dict, List, Optional
from datetime import datetime


def find_trainer_states(model_dir: str) -> List[str]:
    """Find all trainer_state.json files in model directory."""
    # Look in main dir and checkpoint subdirs
    states = []

    # Main directory
    main_state = os.path.join(model_dir, "trainer_state.json")
    if os.path.exists(main_state):
        states.append(main_state)

    # Checkpoint directories
    checkpoint_pattern = os.path.join(model_dir, "checkpoint-*", "trainer_state.json")
    states.extend(glob.glob(checkpoint_pattern))

    return sorted(states)


def read_trainer_state(filepath: str) -> Dict:
    """Read a trainer_state.json file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def extract_metrics(state: Dict) -> Dict:
    """Extract organized metrics from trainer state."""

    metrics = {
        'train_loss': [],
        'eval_loss': [],
        'learning_rate': [],
        'epoch': [],
        'grad_norm': []
    }

    for entry in state.get('log_history', []):
        step = entry.get('step', 0)

        # Training metrics
        if 'loss' in entry:
            metrics['train_loss'].append({
                'step': step,
                'value': entry['loss'],
                'epoch': entry.get('epoch', 0)
            })

        # Evaluation metrics
        if 'eval_loss' in entry:
            metrics['eval_loss'].append({
                'step': step,
                'value': entry['eval_loss'],
                'epoch': entry.get('epoch', 0)
            })

        # Learning rate
        if 'learning_rate' in entry:
            metrics['learning_rate'].append({
                'step': step,
                'value': entry['learning_rate'],
                'epoch': entry.get('epoch', 0)
            })

        # Gradient norm
        if 'grad_norm' in entry:
            metrics['grad_norm'].append({
                'step': step,
                'value': entry['grad_norm'],
                'epoch': entry.get('epoch', 0)
            })

    return metrics


def print_summary(state: Dict):
    """Print training summary."""

    print("\n" + "=" * 80)
    print("TRAINING SUMMARY")
    print("=" * 80)

    # Basic info
    print(f"\nBest metric: {state.get('best_metric', 'N/A')}")
    print(f"Best model checkpoint: {state.get('best_model_checkpoint', 'N/A')}")
    print(f"Total training steps: {state.get('global_step', 0)}")
    print(f"Max steps: {state.get('max_steps', 'N/A')}")
    print(f"Number of epochs: {state.get('epoch', 0)}")

    # Extract metrics
    metrics = extract_metrics(state)

    # Training loss
    if metrics['train_loss']:
        losses = [m['value'] for m in metrics['train_loss']]
        print(f"\nTraining Loss:")
        print(f"  Initial: {losses[0]:.6f}")
        print(f"  Final: {losses[-1]:.6f}")
        print(f"  Minimum: {min(losses):.6f}")
        print(f"  Average: {sum(losses) / len(losses):.6f}")

    # Evaluation loss
    if metrics['eval_loss']:
        eval_losses = [m['value'] for m in metrics['eval_loss']]
        eval_steps = [m['step'] for m in metrics['eval_loss']]
        best_eval_idx = eval_losses.index(min(eval_losses))

        print(f"\nEvaluation Loss:")
        print(f"  Initial: {eval_losses[0]:.6f} (step {eval_steps[0]})")
        print(f"  Final: {eval_losses[-1]:.6f} (step {eval_steps[-1]})")
        print(f"  Best: {eval_losses[best_eval_idx]:.6f} (step {eval_steps[best_eval_idx]})")
        print(f"  Number of evaluations: {len(eval_losses)}")

    # Learning rate
    if metrics['learning_rate']:
        lrs = [m['value'] for m in metrics['learning_rate']]
        print(f"\nLearning Rate:")
        print(f"  Initial: {lrs[0]:.2e}")
        print(f"  Final: {lrs[-1]:.2e}")
        print(f"  Max: {max(lrs):.2e}")


def print_recent(state: Dict, n: int = 10):
    """Print recent training steps."""

    print(f"\n" + "=" * 80)
    print(f"LAST {n} LOGGED ENTRIES")
    print("=" * 80)

    log_history = state.get('log_history', [])

    for entry in log_history[-n:]:
        step = entry.get('step', 0)
        epoch = entry.get('epoch', 0)

        print(f"\nStep {step} (Epoch {epoch:.2f}):")

        # Print key metrics
        if 'loss' in entry:
            print(f"  Training loss: {entry['loss']:.6f}")
        if 'eval_loss' in entry:
            print(f"  Eval loss: {entry['eval_loss']:.6f}")
        if 'learning_rate' in entry:
            print(f"  Learning rate: {entry['learning_rate']:.2e}")
        if 'grad_norm' in entry:
            print(f"  Gradient norm: {entry['grad_norm']:.4f}")


def print_epoch_summary(state: Dict):
    """Print per-epoch statistics."""

    print("\n" + "=" * 80)
    print("PER-EPOCH SUMMARY")
    print("=" * 80)

    metrics = extract_metrics(state)

    # Group by epoch
    epochs_data = {}

    for metric_type in ['train_loss', 'eval_loss']:
        for entry in metrics[metric_type]:
            epoch = int(entry['epoch'])
            if epoch not in epochs_data:
                epochs_data[epoch] = {'train_losses': [], 'eval_losses': []}

            if metric_type == 'train_loss':
                epochs_data[epoch]['train_losses'].append(entry['value'])
            else:
                epochs_data[epoch]['eval_losses'].append(entry['value'])

    # Print per epoch
    for epoch in sorted(epochs_data.keys()):
        data = epochs_data[epoch]
        print(f"\nEpoch {epoch}:")

        if data['train_losses']:
            avg_train = sum(data['train_losses']) / len(data['train_losses'])
            min_train = min(data['train_losses'])
            print(f"  Train loss - Avg: {avg_train:.6f}, Min: {min_train:.6f}")

        if data['eval_losses']:
            # Usually only one eval per epoch, but handle multiple
            for i, eval_loss in enumerate(data['eval_losses']):
                print(f"  Eval loss: {eval_loss:.6f}")


def compare_checkpoints(model_dir: str):
    """Compare metrics across all checkpoints."""

    states_files = find_trainer_states(model_dir)

    if not states_files:
        print(f"No trainer_state.json files found in {model_dir}")
        return

    print("\n" + "=" * 80)
    print("CHECKPOINT COMPARISON")
    print("=" * 80)

    checkpoints = []

    for state_file in states_files:
        state = read_trainer_state(state_file)
        checkpoint_dir = os.path.dirname(state_file)
        checkpoint_name = os.path.basename(checkpoint_dir)

        metrics = extract_metrics(state)

        # Get final metrics
        info = {
            'name': checkpoint_name if checkpoint_name else 'final',
            'step': state.get('global_step', 0),
            'epoch': state.get('epoch', 0)
        }

        if metrics['train_loss']:
            info['final_train_loss'] = metrics['train_loss'][-1]['value']

        if metrics['eval_loss']:
            info['final_eval_loss'] = metrics['eval_loss'][-1]['value']
            info['best_eval_loss'] = min(m['value'] for m in metrics['eval_loss'])

        checkpoints.append(info)

    # Sort by step
    checkpoints.sort(key=lambda x: x['step'])

    # Print comparison
    for ckpt in checkpoints:
        print(f"\n{ckpt['name']}:")
        print(f"  Step: {ckpt['step']}, Epoch: {ckpt['epoch']:.2f}")

        if 'final_train_loss' in ckpt:
            print(f"  Final train loss: {ckpt['final_train_loss']:.6f}")

        if 'final_eval_loss' in ckpt:
            print(f"  Final eval loss: {ckpt['final_eval_loss']:.6f}")

        if 'best_eval_loss' in ckpt:
            print(f"  Best eval loss: {ckpt['best_eval_loss']:.6f}")


def export_metrics(state: Dict, format: str, output_file: str):
    """Export metrics to file."""

    metrics = extract_metrics(state)

    if format == 'json':
        # Export as JSON
        export_data = {
            'summary': {
                'best_metric': state.get('best_metric'),
                'best_model_checkpoint': state.get('best_model_checkpoint'),
                'total_steps': state.get('global_step', 0),
                'epochs': state.get('epoch', 0)
            },
            'metrics': metrics
        }

        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2)

    elif format == 'csv':
        # Export as CSV (requires manual formatting)
        with open(output_file, 'w') as f:
            f.write("step,epoch,metric_type,value\n")

            for metric_type, entries in metrics.items():
                for entry in entries:
                    f.write(f"{entry['step']},{entry['epoch']},{metric_type},{entry['value']}\n")

    print(f"Metrics exported to {output_file}")


def watch_training(model_dir: str, interval: int = 10):
    """Watch training progress in real-time."""

    import time

    print(f"Watching {model_dir}/trainer_state.json")
    print("Press Ctrl+C to stop")
    print("-" * 60)

    state_file = os.path.join(model_dir, "trainer_state.json")
    last_step = 0

    try:
        while True:
            if os.path.exists(state_file):
                try:
                    state = read_trainer_state(state_file)
                    current_step = state.get('global_step', 0)

                    if current_step > last_step:
                        # Get latest entry
                        log_history = state.get('log_history', [])
                        if log_history:
                            latest = log_history[-1]

                            output = f"[Step {current_step:5d}]"

                            if 'loss' in latest:
                                output += f" Loss: {latest['loss']:.4f}"
                            if 'eval_loss' in latest:
                                output += f" | Eval: {latest['eval_loss']:.4f}"
                            if 'learning_rate' in latest:
                                output += f" | LR: {latest['learning_rate']:.2e}"

                            print(output)
                            last_step = current_step

                except (json.JSONDecodeError, KeyError):
                    # File might be being written
                    pass

            time.sleep(interval)

    except KeyboardInterrupt:
        print("\nStopped watching")


def main():
    parser = argparse.ArgumentParser(description="Read HuggingFace Trainer logs from command line")

    parser.add_argument("model_dir", type=str, help="Model output directory")

    # Display options
    parser.add_argument("--summary", action="store_true", default=True,
                        help="Show training summary (default)")
    parser.add_argument("--recent", type=int, metavar="N",
                        help="Show last N log entries")
    parser.add_argument("--epochs", action="store_true",
                        help="Show per-epoch summary")
    parser.add_argument("--compare", action="store_true",
                        help="Compare all checkpoints")

    # Export options
    parser.add_argument("--export-json", type=str, metavar="FILE",
                        help="Export metrics to JSON")
    parser.add_argument("--export-csv", type=str, metavar="FILE",
                        help="Export metrics to CSV")

    # Live monitoring
    parser.add_argument("--watch", action="store_true",
                        help="Watch training progress live")
    parser.add_argument("--interval", type=int, default=10,
                        help="Watch interval in seconds")

    args = parser.parse_args()

    # Find the trainer state file
    if args.compare:
        compare_checkpoints(args.model_dir)
    elif args.watch:
        watch_training(args.model_dir, args.interval)
    else:
        # Read the main trainer state
        state_file = os.path.join(args.model_dir, "trainer_state.json")

        if not os.path.exists(state_file):
            # Try to find in checkpoints
            states = find_trainer_states(args.model_dir)
            if states:
                state_file = states[-1]  # Use most recent
                print(f"Using: {state_file}")
            else:
                print(f"No trainer_state.json found in {args.model_dir}")
                return

        state = read_trainer_state(state_file)

        # Handle different display modes
        if args.recent:
            print_recent(state, args.recent)
        elif args.epochs:
            print_epoch_summary(state)
        elif args.export_json:
            export_metrics(state, 'json', args.export_json)
        elif args.export_csv:
            export_metrics(state, 'csv', args.export_csv)
        else:
            print_summary(state)


if __name__ == "__main__":
    main()
