"""
visualize.py - Standalone visualization script for Nova Micro fine-tuning evaluation results.

Reads eval_results.json and generates:
  1. results/accuracy_comparison.png  - Grouped bar chart (base vs fine-tuned)
  2. results/confusion_matrix.png     - Confusion matrix heatmap (fine-tuned model, optional)

Usage:
  python visualize.py
  python visualize.py --results results/eval_results.json --output-dir results/
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RESULTS_DIR = Path(__file__).parent.parent / 'results'
DEFAULT_RESULTS_FILE = RESULTS_DIR / 'eval_results.json'

LABELS = ['positive', 'negative', 'neutral']
LABEL_DISPLAY = ['Positive', 'Negative', 'Neutral']


# ---------------------------------------------------------------------------
# Chart 1: Accuracy comparison bar chart
# ---------------------------------------------------------------------------

def plot_accuracy_comparison(results: dict, output_path: Path) -> None:
    """Generate a grouped bar chart comparing overall and per-class accuracy.

    Args:
        results: Parsed eval_results.json dict.
        output_path: Destination PNG file path.
    """
    has_base = 'base_model' in results
    has_custom = 'custom_model' in results

    if not has_base and not has_custom:
        print("[WARN] No model results found in JSON. Skipping accuracy chart.")
        return

    # Build value arrays in order: Overall, Positive, Negative, Neutral
    category_labels = ['Overall'] + LABEL_DISPLAY
    category_keys = ['accuracy'] + LABELS  # keys used in JSON

    def extract_values(model_key: str) -> list[float]:
        data = results[model_key]
        values = []
        for key in category_keys:
            if key == 'accuracy':
                values.append(float(data.get('accuracy', 0.0)))
            else:
                values.append(float(data.get('per_class', {}).get(key, 0.0)))
        return values

    base_values = extract_values('base_model') if has_base else []
    custom_values = extract_values('custom_model') if has_custom else []

    x = np.arange(len(category_labels))
    fig, ax = plt.subplots(figsize=(11, 6))

    # Color palette
    COLOR_BASE = '#9CA3AF'    # gray-400
    COLOR_CUSTOM = '#2563EB'  # blue-600
    COLOR_BASE_TEXT = '#374151'
    COLOR_CUSTOM_TEXT = '#1e40af'

    if has_base and has_custom:
        width = 0.35
        bars_base = ax.bar(
            x - width / 2, base_values, width,
            label='Base Model (us.amazon.nova-micro-v1:0)',
            color=COLOR_BASE, edgecolor='white', linewidth=0.5,
        )
        bars_custom = ax.bar(
            x + width / 2, custom_values, width,
            label='Fine-tuned Model',
            color=COLOR_CUSTOM, edgecolor='white', linewidth=0.5,
        )
        _annotate_bars(ax, bars_base, color=COLOR_BASE_TEXT)
        _annotate_bars(ax, bars_custom, color=COLOR_CUSTOM_TEXT)

    elif has_base:
        width = 0.45
        bars_base = ax.bar(x, base_values, width, label='Base Model', color=COLOR_BASE, edgecolor='white')
        _annotate_bars(ax, bars_base, color=COLOR_BASE_TEXT)

    else:  # has_custom only
        width = 0.45
        bars_custom = ax.bar(x, custom_values, width, label='Fine-tuned Model', color=COLOR_CUSTOM, edgecolor='white')
        _annotate_bars(ax, bars_custom, color=COLOR_CUSTOM_TEXT)

    # Axes formatting
    ax.set_xlabel('Category', fontsize=12, labelpad=8)
    ax.set_ylabel('Accuracy', fontsize=12, labelpad=8)
    ax.set_title(
        'Nova Micro Fine-tuning: Sentiment Classification Accuracy',
        fontsize=13, fontweight='bold', pad=14,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(category_labels, fontsize=11)
    ax.set_ylim(0, 1.18)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:.1f}'))
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(axis='y', linestyle='--', alpha=0.35, color='#d1d5db')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Improvement delta annotation (only when both models present)
    if has_base and has_custom:
        delta = custom_values[0] - base_values[0]
        sign = '+' if delta >= 0 else ''
        delta_text = f'Overall improvement: {sign}{delta:.2f}'
        ax.text(
            0.01, 0.97, delta_text,
            transform=ax.transAxes, ha='left', va='top',
            fontsize=10, color='#059669', fontweight='bold',
        )

    # Test count footnote
    test_count = results.get('test_count', '?')
    ax.text(
        0.99, 0.01, f'n = {test_count} test samples',
        transform=ax.transAxes, ha='right', va='bottom',
        fontsize=9, color='#6b7280',
    )

    plt.tight_layout()
    os.makedirs(output_path.parent, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[VIZ] Saved accuracy comparison chart: {output_path}")


def _annotate_bars(ax: plt.Axes, bars, color: str = '#374151') -> None:
    """Add value labels on top of each bar."""
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f'{height:.2f}',
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 5),
            textcoords='offset points',
            ha='center', va='bottom',
            fontsize=9, color=color, fontweight='medium',
        )


# ---------------------------------------------------------------------------
# Chart 2: Confusion matrix heatmap
# ---------------------------------------------------------------------------

def plot_confusion_matrix(results: dict, output_path: Path) -> None:
    """Generate a confusion matrix heatmap for the fine-tuned model.

    Requires 'predictions' list inside results['custom_model'].
    Skips gracefully if prediction detail is unavailable.

    Args:
        results: Parsed eval_results.json dict (must include predictions).
        output_path: Destination PNG file path.
    """
    if 'custom_model' not in results:
        print("[VIZ] No custom_model entry found; skipping confusion matrix.")
        return

    predictions = results['custom_model'].get('predictions')
    if not predictions:
        print("[VIZ] 'predictions' list not present in custom_model results; skipping confusion matrix.")
        print("      Run evaluate.py with --custom-model-arn to generate prediction detail.")
        return

    try:
        import seaborn as sns
    except ImportError:
        print("[WARN] seaborn not installed; skipping confusion matrix. Install with: pip install seaborn")
        return

    # Build confusion matrix (rows=true, cols=predicted)
    label_index = {lbl: i for i, lbl in enumerate(LABELS)}
    n = len(LABELS)
    matrix = np.zeros((n, n), dtype=int)

    for pred in predictions:
        true_idx = label_index.get(pred.get('true_label'))
        pred_idx = label_index.get(pred.get('predicted_label'))
        if true_idx is not None and pred_idx is not None:
            matrix[true_idx][pred_idx] += 1

    # Compute per-row accuracy for annotation
    row_totals = matrix.sum(axis=1, keepdims=True)
    matrix_pct = np.where(row_totals > 0, matrix / row_totals, 0.0)

    # Build combined annotation (count + percentage)
    annot = np.empty_like(matrix, dtype=object)
    for i in range(n):
        for j in range(n):
            annot[i][j] = f"{matrix[i][j]}\n({matrix_pct[i][j]:.0%})"

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        matrix,
        annot=annot,
        fmt='',
        cmap='Blues',
        xticklabels=LABEL_DISPLAY,
        yticklabels=LABEL_DISPLAY,
        ax=ax,
        linewidths=0.8,
        linecolor='#e5e7eb',
        cbar_kws={'shrink': 0.8},
    )
    ax.set_xlabel('Predicted Label', fontsize=11, labelpad=8)
    ax.set_ylabel('True Label', fontsize=11, labelpad=8)
    ax.set_title('Fine-tuned Model: Confusion Matrix', fontsize=12, fontweight='bold', pad=12)
    ax.tick_params(axis='both', labelsize=10)

    # Overall accuracy annotation
    total_correct = int(np.trace(matrix))
    total_samples = int(matrix.sum())
    overall_acc = total_correct / total_samples if total_samples > 0 else 0.0
    ax.text(
        0.99, -0.12, f'Overall accuracy: {overall_acc:.2%}  ({total_correct}/{total_samples})',
        transform=ax.transAxes, ha='right', va='top',
        fontsize=9, color='#374151',
    )

    plt.tight_layout()
    os.makedirs(output_path.parent, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[VIZ] Saved confusion matrix: {output_path}")


# ---------------------------------------------------------------------------
# Helper: pretty-print result summary to stdout
# ---------------------------------------------------------------------------

def print_summary(results: dict) -> None:
    """Print a human-readable accuracy summary table to stdout."""
    test_count = results.get('test_count', '?')
    print(f"\n{'='*52}")
    print(f"  Evaluation Summary  (n={test_count})")
    print(f"{'='*52}")
    print(f"{'Category':<14} {'Base':>10} {'Fine-tuned':>12}")
    print(f"{'-'*52}")

    def fmt(val) -> str:
        return f'{val:.4f}' if val is not None else '  N/A  '

    def get_val(model_key, metric):
        if model_key not in results:
            return None
        data = results[model_key]
        if metric == 'accuracy':
            return data.get('accuracy')
        return data.get('per_class', {}).get(metric)

    rows = [
        ('Overall', 'accuracy'),
        ('Positive', 'positive'),
        ('Negative', 'negative'),
        ('Neutral', 'neutral'),
    ]
    for display_name, key in rows:
        base_val = get_val('base_model', key)
        custom_val = get_val('custom_model', key)
        delta_str = ''
        if base_val is not None and custom_val is not None:
            delta = custom_val - base_val
            sign = '+' if delta >= 0 else ''
            delta_str = f'  ({sign}{delta:.4f})'
        print(f"  {display_name:<12} {fmt(base_val):>10} {fmt(custom_val):>12}{delta_str}")

    print(f"{'='*52}\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Generate accuracy comparison and confusion matrix charts from eval_results.json.'
    )
    parser.add_argument(
        '--results',
        default=str(DEFAULT_RESULTS_FILE),
        metavar='JSON_PATH',
        help=f'Path to eval_results.json (default: {DEFAULT_RESULTS_FILE})',
    )
    parser.add_argument(
        '--output-dir',
        default=str(RESULTS_DIR),
        metavar='DIR',
        help=f'Directory to write chart PNGs (default: {RESULTS_DIR})',
    )
    parser.add_argument(
        '--no-confusion-matrix',
        action='store_true',
        help='Skip confusion matrix generation even if predictions are available.',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    results_path = Path(args.results)
    output_dir = Path(args.output_dir)

    if not results_path.exists():
        print(f"[ERROR] Results file not found: {results_path}")
        print("  Run evaluate.py first to generate results.")
        raise SystemExit(1)

    print(f"[LOAD] Reading results from: {results_path}")
    with open(results_path, 'r', encoding='utf-8') as f:
        results = json.load(f)

    print_summary(results)

    os.makedirs(output_dir, exist_ok=True)

    # Chart 1: Accuracy comparison
    plot_accuracy_comparison(results, output_dir / 'accuracy_comparison.png')

    # Chart 2: Confusion matrix (optional)
    if not args.no_confusion_matrix:
        plot_confusion_matrix(results, output_dir / 'confusion_matrix.png')

    print("\n[DONE] Visualization complete.")


if __name__ == '__main__':
    main()
