# Copyright Reexpress AI, Inc. All rights reserved.

"""
GPU monitoring script to track memory usage during training.
Run this in a separate terminal while training.
"""

import argparse
import time
import subprocess
import json
from datetime import datetime
import os


def get_gpu_info():
    """Get current GPU usage information."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu',
             '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            check=True
        )

        gpus = []
        for line in result.stdout.strip().split('\n'):
            parts = line.split(', ')
            if len(parts) >= 6:
                gpus.append({
                    'index': int(parts[0]),
                    'name': parts[1],
                    'memory_used': int(parts[2]),
                    'memory_total': int(parts[3]),
                    'memory_percent': (int(parts[2]) / int(parts[3])) * 100,
                    'gpu_utilization': int(parts[4]),
                    'temperature': int(parts[5])
                })
        return gpus
    except Exception as e:
        print(f"Error getting GPU info: {e}")
        return []


def monitor_gpus(interval=5, log_file=None):
    """Monitor GPU usage continuously."""

    print("Starting GPU monitoring... Press Ctrl+C to stop")
    print("-" * 80)

    max_memory = {0: 0, 1: 0}  # Track max memory for each GPU

    try:
        while True:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            gpus = get_gpu_info()

            # Clear screen (optional - comment out if you want to see history)
            os.system('clear' if os.name == 'posix' else 'cls')

            print(f"Timestamp: {timestamp}")
            print("-" * 80)

            for gpu in gpus:
                idx = gpu['index']
                mem_used = gpu['memory_used']
                mem_total = gpu['memory_total']
                mem_percent = gpu['memory_percent']

                # Update max memory
                if idx in max_memory:
                    max_memory[idx] = max(max_memory[idx], mem_used)

                # Create memory bar
                bar_length = 50
                filled = int(bar_length * mem_percent / 100)
                bar = '█' * filled + '░' * (bar_length - filled)

                print(f"GPU {idx}: {gpu['name']}")
                print(f"  Memory: [{bar}] {mem_used:,}/{mem_total:,} MB ({mem_percent:.1f}%)")
                print(f"  Max Memory Used: {max_memory.get(idx, 0):,} MB")
                print(f"  GPU Utilization: {gpu['gpu_utilization']}%")
                print(f"  Temperature: {gpu['temperature']}°C")
                print()

            # Calculate total memory usage
            total_used = sum(gpu['memory_used'] for gpu in gpus)
            total_available = sum(gpu['memory_total'] for gpu in gpus)
            total_percent = (total_used / total_available * 100) if total_available > 0 else 0

            print("-" * 80)
            print(f"Total Memory: {total_used:,}/{total_available:,} MB ({total_percent:.1f}%)")
            print(f"Total Max Memory: {sum(max_memory.values()):,} MB")

            # Log to file if specified
            if log_file:
                log_entry = {
                    'timestamp': timestamp,
                    'gpus': gpus,
                    'total_used': total_used,
                    'total_available': total_available,
                    'max_memory': max_memory
                }

                with open(log_file, 'a') as f:
                    f.write(json.dumps(log_entry) + '\n')

            time.sleep(interval)

    except KeyboardInterrupt:
        print("\n" + "-" * 80)
        print("Monitoring stopped.")
        print(f"Peak memory usage:")
        for idx, max_mem in max_memory.items():
            print(f"  GPU {idx}: {max_mem:,} MB")


def analyze_log(log_file):
    """Analyze GPU usage from log file."""

    print(f"Analyzing log file: {log_file}")
    print("-" * 80)

    entries = []
    with open(log_file, 'r') as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))

    if not entries:
        print("No data found in log file")
        return

    # Find peak usage for each GPU
    gpu_peaks = {}
    gpu_avg_util = {}
    gpu_temps = {}

    for entry in entries:
        for gpu in entry['gpus']:
            idx = gpu['index']

            if idx not in gpu_peaks:
                gpu_peaks[idx] = 0
                gpu_avg_util[idx] = []
                gpu_temps[idx] = []

            gpu_peaks[idx] = max(gpu_peaks[idx], gpu['memory_used'])
            gpu_avg_util[idx].append(gpu['gpu_utilization'])
            gpu_temps[idx].append(gpu['temperature'])

    # Print statistics
    print("GPU Statistics:")
    for idx in sorted(gpu_peaks.keys()):
        print(f"\nGPU {idx}:")
        print(f"  Peak Memory: {gpu_peaks[idx]:,} MB")
        print(f"  Avg Utilization: {sum(gpu_avg_util[idx]) / len(gpu_avg_util[idx]):.1f}%")
        print(f"  Avg Temperature: {sum(gpu_temps[idx]) / len(gpu_temps[idx]):.1f}°C")
        print(f"  Max Temperature: {max(gpu_temps[idx])}°C")

    # Find time periods of high usage
    high_usage_periods = []
    threshold = 60000  # 60GB in MB

    for i, entry in enumerate(entries):
        total_used = entry['total_used']
        if total_used > threshold:
            high_usage_periods.append({
                'timestamp': entry['timestamp'],
                'usage': total_used
            })

    if high_usage_periods:
        print(f"\nHigh memory usage periods (>{threshold:,} MB):")
        for period in high_usage_periods[:10]:  # Show first 10
            print(f"  {period['timestamp']}: {period['usage']:,} MB")

    print("\nTraining duration:")
    start_time = datetime.strptime(entries[0]['timestamp'], "%Y-%m-%d %H:%M:%S")
    end_time = datetime.strptime(entries[-1]['timestamp'], "%Y-%m-%d %H:%M:%S")
    duration = end_time - start_time
    print(f"  Start: {entries[0]['timestamp']}")
    print(f"  End: {entries[-1]['timestamp']}")
    print(f"  Duration: {duration}")


def main():
    parser = argparse.ArgumentParser(description="Monitor GPU usage during training")

    parser.add_argument("--interval", type=int, default=5,
                        help="Monitoring interval in seconds")
    parser.add_argument("--log_file", type=str,
                        help="Log file to save GPU usage data")
    parser.add_argument("--analyze", type=str,
                        help="Analyze existing log file instead of monitoring")

    args = parser.parse_args()

    if args.analyze:
        analyze_log(args.analyze)
    else:
        monitor_gpus(interval=args.interval, log_file=args.log_file)


if __name__ == "__main__":
    main()
