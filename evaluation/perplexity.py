import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import numpy as np
from tqdm import tqdm
import json
import glob
import os
import re
import pandas as pd
import argparse

if torch.cuda.is_available():
    device = torch.device('cuda')
    print("Using CUDA")
elif torch.backends.mps.is_available():
    device = torch.device('cpu')
    print("MPS available but using CPU due to transformer compatibility issues")
else:
    device = torch.device('cpu')
    print("Using CPU")

print(f"Device: {device}\n")


def calculate_perplexity(text, model, tokenizer, max_length=1024):
    # Encode the text
    encodings = tokenizer(text, return_tensors='pt', truncation=True, max_length=max_length)
    input_ids = encodings.input_ids.to(device)

    # Calculate loss
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss

    # Perplexity is exp(loss)
    perplexity = torch.exp(loss).item()

    return perplexity


def calculate_perplexities_for_dataset(data, model, tokenizer):
    # Initialize storage for perplexities
    input_ppls = []
    prediction_ppls = []
    actual_ppls = []

    # Track issues
    empty_inputs = 0
    empty_predictions = 0
    empty_actuals = 0
    invalid_input_ppls = 0
    invalid_prediction_ppls = 0
    invalid_actual_ppls = 0

    # Calculate perplexity for each sample
    for idx, sample in enumerate(tqdm(data, desc="Processing samples", leave=False)):
        # Calculate for input
        if sample['input'].strip():
            input_ppl = calculate_perplexity(sample['input'], model, tokenizer)
            if np.isfinite(input_ppl):
                input_ppls.append(input_ppl)
            else:
                invalid_input_ppls += 1
        else:
            empty_inputs += 1

        # Calculate for prediction
        if sample['prediction'].strip():
            pred_ppl = calculate_perplexity(sample['prediction'], model, tokenizer)
            if np.isfinite(pred_ppl):
                prediction_ppls.append(pred_ppl)
            else:
                invalid_prediction_ppls += 1
        else:
            empty_predictions += 1

        # Calculate for actual
        if sample['actual'].strip():
            actual_ppl = calculate_perplexity(sample['actual'], model, tokenizer)
            if np.isfinite(actual_ppl):
                actual_ppls.append(actual_ppl)
            else:
                invalid_actual_ppls += 1
        else:
            empty_actuals += 1

    # Print warnings if issues found
    if empty_predictions > 0:
        print(f"    ⚠️  WARNING: Found {empty_predictions} empty predictions!")
    if invalid_prediction_ppls > 0:
        print(f"    ⚠️  WARNING: Found {invalid_prediction_ppls} invalid (inf/nan) prediction perplexities!")

    # Calculate statistics (handle empty lists)
    def safe_stats(ppls_list):
        if len(ppls_list) > 0:
            return {
                'average': np.mean(ppls_list),
                'std': np.std(ppls_list),
                'min': np.min(ppls_list),
                'max': np.max(ppls_list),
                'count': len(ppls_list),
                'individual': ppls_list
            }
        else:
            return {
                'average': np.nan,
                'std': np.nan,
                'min': np.nan,
                'max': np.nan,
                'count': 0,
                'individual': []
            }

    results = {
        'input': safe_stats(input_ppls),
        'prediction': safe_stats(prediction_ppls),
        'actual': safe_stats(actual_ppls)
    }

    return results


def extract_checkpoint_number(filename):
    """Extract checkpoint number from filename."""
    match = re.search(r'checkpoint-(\d+)', filename)
    if match:
        return int(match.group(1))
    return None


def extract_experiment_name(filename):
    # Pattern: full_samples_gutenberg-100k_EXPERIMENT_checkpoint-*.json
    # or: full_samples_gutenberg-100k_EXPERIMENT_final.json
    basename = os.path.basename(filename)

    # Try to match checkpoint files
    match = re.search(r'full_samples_gutenberg-100k_([^_]+)_checkpoint-\d+\.json', basename)
    if match:
        return match.group(1)

    # Try to match final files
    match = re.search(r'full_samples_gutenberg-100k_([^_]+)_final\.json', basename)
    if match:
        return match.group(1)

    return None


def process_all_experiments(base_pattern='full_samples_*_checkpoint-*.json', model_name='gpt2'):
    """
    Process all experiment types and combine results.
    """
    # Find all files
    all_files = glob.glob(base_pattern)

    if not all_files:
        print(f"No files found matching pattern: {base_pattern}")
        return None

    # Group files by experiment type
    experiments = {}
    for file in all_files:
        exp_name = extract_experiment_name(file)
        checkpoint = extract_checkpoint_number(file)

        if exp_name and checkpoint is not None:
            if exp_name not in experiments:
                experiments[exp_name] = []
            experiments[exp_name].append((checkpoint, file))

    if not experiments:
        print("No valid experiment files found!")
        return None

    print(f"Found {len(experiments)} experiment types:")
    for exp_name, files in experiments.items():
        print(f"  - {exp_name}: {len(files)} checkpoints")
    print()

    # Load model once
    print(f"Loading {model_name}...")
    model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
    model.eval()
    print("Model loaded!\n")

    # Process each experiment
    all_experiment_results = {}

    for exp_name in sorted(experiments.keys()):
        files = experiments[exp_name]

        print(f"\n{'=' * 80}")
        print(f"PROCESSING EXPERIMENT: {exp_name}")
        print('=' * 80)

        # Sort files by checkpoint
        files.sort(key=lambda x: x[0])

        exp_results = {}

        for checkpoint, file in files:
            print(f"\n  Checkpoint {checkpoint}: {os.path.basename(file)}")

            # Load dataset
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            print(f"    Loaded {len(data)} samples")

            # Calculate perplexities
            results = calculate_perplexities_for_dataset(data, model, tokenizer)

            # Store results
            exp_results[checkpoint] = {
                'filename': file,
                'num_samples': len(data),
                'results': results
            }

            # Print summary
            for field in ['input', 'prediction', 'actual']:
                count = results[field]['count']
                if count > 0:
                    print(f"    {field}: Avg={results[field]['average']:.2f}, "
                          f"Std={results[field]['std']:.2f}, Count={count}")
                else:
                    print(f"    {field}: No valid samples")

        all_experiment_results[exp_name] = exp_results

    return all_experiment_results


def save_combined_results(all_experiment_results, output_file='combined_perplexity_results.csv'):
    rows = []

    for exp_name, exp_results in all_experiment_results.items():
        for checkpoint in sorted(exp_results.keys()):
            results = exp_results[checkpoint]['results']

            row = {
                'Experiment': exp_name,
                'Checkpoint': checkpoint,
                'Num_Samples': exp_results[checkpoint]['num_samples'],
                'Input_Avg': results['input']['average'],
                'Input_Std': results['input']['std'],
                'Input_Count': results['input']['count'],
                'Prediction_Avg': results['prediction']['average'],
                'Prediction_Std': results['prediction']['std'],
                'Prediction_Count': results['prediction']['count'],
                'Actual_Avg': results['actual']['average'],
                'Actual_Std': results['actual']['std'],
                'Actual_Count': results['actual']['count']
            }
            rows.append(row)

    # Create DataFrame
    df = pd.DataFrame(rows)

    # Save to CSV
    df.to_csv(output_file, index=False, float_format='%.2f')
    print(f"\n{'=' * 80}")
    print(f"SAVED COMBINED RESULTS TO: {output_file}")
    print('=' * 80)

    # Print summary
    print("\nSummary by Experiment:")
    print("-" * 80)
    for exp_name in sorted(df['Experiment'].unique()):
        exp_df = df[df['Experiment'] == exp_name]
        print(f"\n{exp_name}:")
        print(f"  Checkpoints: {exp_df['Checkpoint'].min()} - {exp_df['Checkpoint'].max()}")
        print(f"  Total samples per checkpoint: {exp_df['Num_Samples'].iloc[0]}")

        # Average across all checkpoints (excluding NaN)
        print(f"  Average perplexity (across all checkpoints):")
        if exp_df['Input_Avg'].notna().any():
            print(f"    Input: {exp_df['Input_Avg'].mean():.2f} (std: {exp_df['Input_Std'].mean():.2f})")
        if exp_df['Prediction_Avg'].notna().any():
            print(f"    Prediction: {exp_df['Prediction_Avg'].mean():.2f} (std: {exp_df['Prediction_Std'].mean():.2f})")
        if exp_df['Actual_Avg'].notna().any():
            print(f"    Actual: {exp_df['Actual_Avg'].mean():.2f} (std: {exp_df['Actual_Std'].mean():.2f})")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Calculate perplexity for GPT-2 generated text across multiple experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all files in full-samples directory (default)
  python script.py

  # Specify custom pattern
  python script.py -p "full-samples/full_samples_gutenberg-100k_*_checkpoint-*.json"

  # Different directory
  python script.py -p "my-samples/full_samples_*_checkpoint-*.json"

  # Specify output file
  python script.py -o my_results.csv
        """
    )

    parser.add_argument(
        '-p', '--pattern',
        type=str,
        default='full-samples/full_samples_gutenberg-100k_*_checkpoint-*.json',
        help="File pattern to match (default: full-samples/full_samples_gutenberg-100k_*_checkpoint-*.json)"
    )

    parser.add_argument(
        '-o', '--output',
        type=str,
        default='combined_perplexity_results.csv',
        help="Output CSV filename (default: combined_perplexity_results.csv)"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("GPT-2 PERPLEXITY CALCULATOR FOR MULTIPLE EXPERIMENTS")
    print("=" * 80)
    print(f"Pattern: {args.pattern}")
    print(f"Output: {args.output}")
    print()

    # Process all experiments
    all_experiment_results = process_all_experiments(base_pattern=args.pattern, model_name='gpt2')

    if all_experiment_results:
        # Save combined results
        df = save_combined_results(all_experiment_results, output_file=args.output)

        # Print final summary table
        print("\n" + "=" * 80)
        print("FINAL RESULTS TABLE (First 20 rows)")
        print("=" * 80)
        print(df.head(20).to_string(index=False))
        if len(df) > 20:
            print(f"\n... ({len(df) - 20} more rows)")
        print("=" * 80)
        print(f"\nFull results saved to: {args.output}")
    else:
        print("\nNo datasets were processed. Please check your file pattern and try again.")