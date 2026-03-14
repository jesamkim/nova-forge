"""
evaluate.py - Before/After evaluation script for Nova Micro fine-tuning.

Compares sentiment classification accuracy between:
  - Base model: us.amazon.nova-micro-v1:0
  - Custom model: Provisioned Throughput ARN (optional)

Usage:
  python evaluate.py --base-only
  python evaluate.py --custom-model-arn <ARN>
  python evaluate.py --results-only results/eval_results.json
"""

import argparse
import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path

import boto3
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

AWS_PROFILE = 'profile2'
AWS_REGION = 'us-east-1'
BASE_MODEL_ID = 'us.amazon.nova-micro-v1:0'

DATA_PATH = Path(__file__).parent.parent / 'data' / 'test.jsonl'
RESULTS_DIR = Path(__file__).parent.parent / 'results'
RESULTS_FILE = RESULTS_DIR / 'eval_results.json'

LABELS = ['positive', 'negative', 'neutral']
RATE_LIMIT_SLEEP = 0.5  # seconds between API calls


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_test_data(path: Path) -> list[dict]:
    """Load test examples from JSONL file.

    Each line must have the format:
      {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "positive|negative|neutral"}]}

    Returns a list of {"prompt": str, "label": str} dicts.
    """
    examples = []
    with open(path, 'r', encoding='utf-8') as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                messages = record['messages']
                # Find user prompt and assistant label
                prompt = next(m['content'] for m in messages if m['role'] == 'user')
                label = next(m['content'] for m in messages if m['role'] == 'assistant')
                label = label.strip().lower()
                if label not in LABELS:
                    print(f"[WARN] Line {line_no}: unexpected label '{label}', skipping.")
                    continue
                examples.append({'prompt': prompt, 'label': label})
            except (KeyError, StopIteration, json.JSONDecodeError) as e:
                print(f"[WARN] Line {line_no}: failed to parse ({e}), skipping.")
    return examples


# ---------------------------------------------------------------------------
# Bedrock API
# ---------------------------------------------------------------------------

def build_bedrock_client():
    """Create a Bedrock Runtime client using profile2."""
    session = boto3.Session(profile_name=AWS_PROFILE)
    return session.client('bedrock-runtime', region_name=AWS_REGION)


def extract_label(response_text: str) -> str:
    """Extract the first matching sentiment label from model response text.

    Returns one of 'positive', 'negative', 'neutral', or 'unknown'.
    """
    text = response_text.lower()
    for label in LABELS:
        if re.search(r'\b' + label + r'\b', text):
            return label
    return 'unknown'


def call_converse(client, model_id: str, prompt: str) -> str:
    """Send a single prompt via the Bedrock Converse API and return the response text."""
    response = client.converse(
        modelId=model_id,
        messages=[
            {
                'role': 'user',
                'content': [{'text': prompt}],
            }
        ],
        inferenceConfig={
            'maxTokens': 16,
            'temperature': 0.0,
        },
    )
    # Extract text from the response content blocks
    output_message = response['output']['message']
    parts = [block['text'] for block in output_message['content'] if 'text' in block]
    return ' '.join(parts).strip()


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(client, model_id: str, examples: list[dict], label: str) -> dict:
    """Run model inference on all examples and compute accuracy metrics.

    Args:
        client: Bedrock Runtime client.
        model_id: Model ID or Provisioned Throughput ARN.
        examples: List of {"prompt": str, "label": str} dicts.
        label: Human-readable model label for progress logging.

    Returns:
        dict with keys: accuracy, per_class, predictions
    """
    total = len(examples)
    correct = 0
    per_class_correct = defaultdict(int)
    per_class_total = defaultdict(int)
    predictions = []

    print(f"\n[{label}] Evaluating {total} examples with model: {model_id}")

    for idx, example in enumerate(examples):
        prompt = example['prompt']
        true_label = example['label']
        per_class_total[true_label] += 1

        try:
            response_text = call_converse(client, model_id, prompt)
            predicted_label = extract_label(response_text)
        except Exception as e:
            print(f"  [ERROR] Example {idx + 1}: API call failed ({e})")
            predicted_label = 'unknown'
            response_text = ''

        is_correct = predicted_label == true_label
        if is_correct:
            correct += 1
            per_class_correct[true_label] += 1

        predictions.append({
            'prompt': prompt,
            'true_label': true_label,
            'predicted_label': predicted_label,
            'response_text': response_text,
            'correct': is_correct,
        })

        if (idx + 1) % 10 == 0:
            running_acc = correct / (idx + 1)
            print(f"  Progress: {idx + 1}/{total} | Running accuracy: {running_acc:.3f}")

        # Rate limiting
        time.sleep(RATE_LIMIT_SLEEP)

    # Compute overall accuracy
    accuracy = correct / total if total > 0 else 0.0

    # Compute per-class accuracy
    per_class = {}
    for lbl in LABELS:
        if per_class_total[lbl] > 0:
            per_class[lbl] = per_class_correct[lbl] / per_class_total[lbl]
        else:
            per_class[lbl] = 0.0

    print(f"  [DONE] Overall accuracy: {accuracy:.4f}")
    for lbl in LABELS:
        print(f"    {lbl}: {per_class[lbl]:.4f} ({per_class_correct[lbl]}/{per_class_total[lbl]})")

    return {
        'accuracy': round(accuracy, 4),
        'per_class': {k: round(v, 4) for k, v in per_class.items()},
        'predictions': predictions,
    }


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_accuracy_comparison(results: dict, output_path: Path) -> None:
    """Generate grouped bar chart comparing base vs custom model accuracy.

    Saves to output_path as PNG.
    """
    categories = ['Overall', 'Positive', 'Negative', 'Neutral']
    category_keys = ['accuracy', 'positive', 'negative', 'neutral']

    has_base = 'base_model' in results
    has_custom = 'custom_model' in results

    base_values = []
    custom_values = []

    for i, key in enumerate(category_keys):
        if has_base:
            if key == 'accuracy':
                base_values.append(results['base_model']['accuracy'])
            else:
                base_values.append(results['base_model']['per_class'].get(key, 0.0))
        if has_custom:
            if key == 'accuracy':
                custom_values.append(results['custom_model']['accuracy'])
            else:
                custom_values.append(results['custom_model']['per_class'].get(key, 0.0))

    x = np.arange(len(categories))
    fig, ax = plt.subplots(figsize=(10, 6))

    if has_base and has_custom:
        width = 0.35
        bars_base = ax.bar(x - width / 2, base_values, width, label='Base Model', color='#888888')
        bars_custom = ax.bar(x + width / 2, custom_values, width, label='Fine-tuned Model', color='#2563EB')

        # Annotate bar values
        for bar in bars_base:
            height = bar.get_height()
            ax.annotate(
                f'{height:.2f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 4),
                textcoords='offset points',
                ha='center', va='bottom', fontsize=9, color='#444444',
            )
        for bar in bars_custom:
            height = bar.get_height()
            ax.annotate(
                f'{height:.2f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 4),
                textcoords='offset points',
                ha='center', va='bottom', fontsize=9, color='#1e40af',
            )
    elif has_base:
        width = 0.4
        bars_base = ax.bar(x, base_values, width, label='Base Model', color='#888888')
        for bar in bars_base:
            height = bar.get_height()
            ax.annotate(
                f'{height:.2f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 4),
                textcoords='offset points',
                ha='center', va='bottom', fontsize=9,
            )
    elif has_custom:
        width = 0.4
        bars_custom = ax.bar(x, custom_values, width, label='Fine-tuned Model', color='#2563EB')
        for bar in bars_custom:
            height = bar.get_height()
            ax.annotate(
                f'{height:.2f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 4),
                textcoords='offset points',
                ha='center', va='bottom', fontsize=9,
            )

    ax.set_xlabel('Category', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Nova Micro Fine-tuning: Sentiment Classification Accuracy', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.1f}'))
    ax.legend(fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.4)

    # Add test count annotation
    test_count = results.get('test_count', '?')
    ax.text(
        0.99, 0.02, f'n={test_count} test samples',
        transform=ax.transAxes, ha='right', va='bottom',
        fontsize=9, color='#666666',
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[VIZ] Saved accuracy comparison chart to: {output_path}")


def plot_confusion_matrix(results: dict, output_path: Path) -> None:
    """Generate confusion matrix heatmap for the custom (fine-tuned) model.

    Only generated when custom model predictions are available.
    Saves to output_path as PNG.
    """
    if 'custom_model' not in results:
        print("[VIZ] No custom model results; skipping confusion matrix.")
        return

    predictions = results['custom_model'].get('predictions', [])
    if not predictions:
        print("[VIZ] No prediction detail available; skipping confusion matrix.")
        return

    import seaborn as sns

    # Build confusion matrix array (rows=true, cols=predicted)
    label_index = {lbl: i for i, lbl in enumerate(LABELS)}
    matrix = np.zeros((len(LABELS), len(LABELS)), dtype=int)

    for pred in predictions:
        true_idx = label_index.get(pred['true_label'])
        pred_idx = label_index.get(pred['predicted_label'])
        if true_idx is not None and pred_idx is not None:
            matrix[true_idx][pred_idx] += 1

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=[lbl.capitalize() for lbl in LABELS],
        yticklabels=[lbl.capitalize() for lbl in LABELS],
        ax=ax,
        linewidths=0.5,
    )
    ax.set_xlabel('Predicted Label', fontsize=11)
    ax.set_ylabel('True Label', fontsize=11)
    ax.set_title('Fine-tuned Model: Confusion Matrix', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[VIZ] Saved confusion matrix to: {output_path}")


# ---------------------------------------------------------------------------
# Result persistence
# ---------------------------------------------------------------------------

def save_results(results: dict, path: Path) -> None:
    """Save evaluation results to JSON, excluding raw predictions from the top-level file."""
    # Store a slimmed version (no per-example predictions) for the summary file
    summary = {'test_count': results.get('test_count', 0)}

    for model_key in ('base_model', 'custom_model'):
        if model_key in results:
            model_data = results[model_key]
            summary[model_key] = {
                'accuracy': model_data['accuracy'],
                'per_class': model_data['per_class'],
            }

    os.makedirs(path.parent, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[SAVE] Results saved to: {path}")


def load_results(path: Path) -> dict:
    """Load previously saved evaluation results from JSON."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Evaluate Nova Micro base and fine-tuned models on Korean sentiment classification.'
    )

    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        '--base-only',
        action='store_true',
        help='Evaluate base model only (no custom model required).',
    )
    mode_group.add_argument(
        '--custom-model-arn',
        metavar='ARN',
        help='Provisioned Throughput ARN for the fine-tuned custom model. Evaluates both base and custom.',
    )
    mode_group.add_argument(
        '--results-only',
        metavar='JSON_PATH',
        help='Skip inference; load existing results JSON and regenerate visualizations.',
    )

    parser.add_argument(
        '--data-path',
        default=str(DATA_PATH),
        help=f'Path to test.jsonl (default: {DATA_PATH})',
    )
    parser.add_argument(
        '--output-dir',
        default=str(RESULTS_DIR),
        help=f'Directory to write results and charts (default: {RESULTS_DIR})',
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # --results-only: skip inference, just visualize
    if args.results_only:
        results_path = Path(args.results_only)
        print(f"[MODE] results-only: loading {results_path}")
        results = load_results(results_path)
        plot_accuracy_comparison(results, output_dir / 'accuracy_comparison.png')
        plot_confusion_matrix(results, output_dir / 'confusion_matrix.png')
        return

    # Load test data
    data_path = Path(args.data_path)
    print(f"[DATA] Loading test data from: {data_path}")
    examples = load_test_data(data_path)
    print(f"[DATA] Loaded {len(examples)} examples.")

    if not examples:
        print("[ERROR] No examples loaded. Aborting.")
        return

    # Build Bedrock client
    client = build_bedrock_client()

    results: dict = {'test_count': len(examples)}

    # Evaluate base model
    print("\n[STEP 1] Evaluating base model...")
    base_result = evaluate_model(client, BASE_MODEL_ID, examples, label='Base Model')
    results['base_model'] = base_result

    # Evaluate custom model (if ARN provided)
    if args.custom_model_arn:
        print("\n[STEP 2] Evaluating fine-tuned custom model...")
        custom_result = evaluate_model(client, args.custom_model_arn, examples, label='Custom Model')
        results['custom_model'] = custom_result

        # Print improvement summary
        base_acc = results['base_model']['accuracy']
        custom_acc = results['custom_model']['accuracy']
        delta = custom_acc - base_acc
        print(f"\n[SUMMARY] Base: {base_acc:.4f} | Fine-tuned: {custom_acc:.4f} | Delta: {delta:+.4f}")

    # Save results
    results_file = output_dir / 'eval_results.json'
    save_results(results, results_file)

    # Generate visualizations
    print("\n[VIZ] Generating visualizations...")
    plot_accuracy_comparison(results, output_dir / 'accuracy_comparison.png')
    plot_confusion_matrix(results, output_dir / 'confusion_matrix.png')

    print("\n[DONE] Evaluation complete.")


if __name__ == '__main__':
    main()
