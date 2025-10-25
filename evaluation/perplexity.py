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


def calculate_perplexity(text, model, tokenizer, device='cpu', max_length=1024):
    """
    Calculate perplexity for a single text sample.
    """
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


def calculate_perplexities_for_dataset(data, model, tokenizer, device):
    """
    Calculate average perplexity for input, prediction, and actual fields.
    """
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
    for idx, sample in enumerate(tqdm(data, desc="Processing samples")):
        # Calculate for input
        if sample['input'].strip():
            input_ppl = calculate_perplexity(sample['input'], model, tokenizer, device)
            if np.isfinite(input_ppl):
                input_ppls.append(input_ppl)
            else:
                invalid_input_ppls += 1
        else:
            empty_inputs += 1

        # Calculate for prediction
        if sample['prediction'].strip():
            pred_ppl = calculate_perplexity(sample['prediction'], model, tokenizer, device)
            if np.isfinite(pred_ppl):
                prediction_ppls.append(pred_ppl)
            else:
                invalid_prediction_ppls += 1
        else:
            empty_predictions += 1

        # Calculate for actual
        if sample['actual'].strip():
            actual_ppl = calculate_perplexity(sample['actual'], model, tokenizer, device)
            if np.isfinite(actual_ppl):
                actual_ppls.append(actual_ppl)
            else:
                invalid_actual_ppls += 1
        else:
            empty_actuals += 1

    # Print warnings if issues found
    if empty_predictions > 0:
        print(f"\n⚠️  WARNING: Found {empty_predictions} empty predictions!")
    if invalid_prediction_ppls > 0:
        print(f"⚠️  WARNING: Found {invalid_prediction_ppls} invalid (inf/nan) prediction perplexities!")
    if empty_inputs > 0:
        print(f"⚠️  WARNING: Found {empty_inputs} empty inputs!")
    if invalid_input_ppls > 0:
        print(f"⚠️  WARNING: Found {invalid_input_ppls} invalid (inf/nan) input perplexities!")
    if empty_actuals > 0:
        print(f"⚠️  WARNING: Found {empty_actuals} empty actuals!")
    if invalid_actual_ppls > 0:
        print(f"⚠️  WARNING: Found {invalid_actual_ppls} invalid (inf/nan) actual perplexities!")

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


def process_all_datasets(pattern='*checkpoint-*.json', model_name='gpt2', device=None):
    """
    Process all dataset files matching the pattern.
    """
    # Set device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")

    # Find all matching files
    files = glob.glob(pattern)

    if not files:
        print(f"No files found matching pattern: {pattern}")
        print("Please make sure your dataset files are in the current directory.")
        return None

    # Sort files by checkpoint number
    files_with_checkpoints = []
    for file in files:
        checkpoint = extract_checkpoint_number(file)
        if checkpoint is not None:
            files_with_checkpoints.append((checkpoint, file))

    files_with_checkpoints.sort(key=lambda x: x[0])

    print(f"Found {len(files_with_checkpoints)} dataset files:")
    for checkpoint, file in files_with_checkpoints:
        print(f"  - Checkpoint {checkpoint}: {file}")
    print()

    # Load model and tokenizer once
    print(f"Loading {model_name}...")
    model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
    model.eval()
    print("Model loaded!\n")

    # Process each file
    all_results = {}

    for checkpoint, file in files_with_checkpoints:
        print(f"\n{'=' * 70}")
        print(f"Processing Checkpoint {checkpoint}: {file}")
        print('=' * 70)

        # Load dataset
        with open(file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        print(f"Loaded {len(data)} samples")

        # Calculate perplexities
        results = calculate_perplexities_for_dataset(data, model, tokenizer, device)

        # Store results
        all_results[checkpoint] = {
            'filename': file,
            'num_samples': len(data),
            'results': results
        }

        # Print summary for this checkpoint
        print(f"\nResults for Checkpoint {checkpoint}:")
        for field in ['input', 'prediction', 'actual']:
            count = results[field]['count']
            if count > 0:
                print(f"  {field.upper()}: Avg={results[field]['average']:.2f}, "
                      f"Std={results[field]['std']:.2f}, "
                      f"Min={results[field]['min']:.2f}, "
                      f"Max={results[field]['max']:.2f}, "
                      f"Valid_Samples={count}")
            else:
                print(f"  {field.upper()}: No valid samples (all empty or invalid)")

    return all_results


def create_summary_table(all_results):
    """Create a summary table comparing all checkpoints."""
    rows = []

    for checkpoint in sorted(all_results.keys()):
        results = all_results[checkpoint]['results']

        row = {
            'Checkpoint': checkpoint,
            'Num_Samples': all_results[checkpoint]['num_samples'],
            'Input_Avg': results['input']['average'],
            'Input_Std': results['input']['std'],
            'Prediction_Avg': results['prediction']['average'],
            'Prediction_Std': results['prediction']['std'],
            'Actual_Avg': results['actual']['average'],
            'Actual_Std': results['actual']['std']
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    return df


def save_results(all_results, output_dir='perplexity_results'):
    """Save all results to files."""
    os.makedirs(output_dir, exist_ok=True)

    # Save detailed results for each checkpoint
    for checkpoint, data in all_results.items():
        output_file = os.path.join(output_dir, f'checkpoint_{checkpoint}_detailed.json')

        # Convert numpy types to Python types for JSON serialization
        output = {
            'checkpoint': checkpoint,
            'filename': data['filename'],
            'num_samples': data['num_samples'],
            'results': {
                field: {
                    'average': float(data['results'][field]['average']) if np.isfinite(
                        data['results'][field]['average']) else None,
                    'std': float(data['results'][field]['std']) if np.isfinite(data['results'][field]['std']) else None,
                    'min': float(data['results'][field]['min']) if np.isfinite(data['results'][field]['min']) else None,
                    'max': float(data['results'][field]['max']) if np.isfinite(data['results'][field]['max']) else None,
                    'count': data['results'][field]['count'],
                    'individual': [float(x) for x in data['results'][field]['individual']]
                }
                for field in ['input', 'prediction', 'actual']
            }
        }

        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)

    # Create comprehensive JSON with all results
    all_results_json = {
        'summary': [],
        'detailed': {}
    }

    # Add summary for each checkpoint
    for checkpoint in sorted(all_results.keys()):
        results = all_results[checkpoint]['results']

        summary_entry = {
            'checkpoint': checkpoint,
            'filename': all_results[checkpoint]['filename'],
            'num_samples': all_results[checkpoint]['num_samples'],
            'input': {
                'average': float(results['input']['average']) if np.isfinite(results['input']['average']) else None,
                'std': float(results['input']['std']) if np.isfinite(results['input']['std']) else None,
                'min': float(results['input']['min']) if np.isfinite(results['input']['min']) else None,
                'max': float(results['input']['max']) if np.isfinite(results['input']['max']) else None,
                'count': results['input']['count']
            },
            'prediction': {
                'average': float(results['prediction']['average']) if np.isfinite(
                    results['prediction']['average']) else None,
                'std': float(results['prediction']['std']) if np.isfinite(results['prediction']['std']) else None,
                'min': float(results['prediction']['min']) if np.isfinite(results['prediction']['min']) else None,
                'max': float(results['prediction']['max']) if np.isfinite(results['prediction']['max']) else None,
                'count': results['prediction']['count']
            },
            'actual': {
                'average': float(results['actual']['average']) if np.isfinite(results['actual']['average']) else None,
                'std': float(results['actual']['std']) if np.isfinite(results['actual']['std']) else None,
                'min': float(results['actual']['min']) if np.isfinite(results['actual']['min']) else None,
                'max': float(results['actual']['max']) if np.isfinite(results['actual']['max']) else None,
                'count': results['actual']['count']
            }
        }
        all_results_json['summary'].append(summary_entry)

        # Add detailed individual perplexities
        all_results_json['detailed'][str(checkpoint)] = {
            'input': [float(x) for x in results['input']['individual']],
            'prediction': [float(x) for x in results['prediction']['individual']],
            'actual': [float(x) for x in results['actual']['individual']]
        }

    # Save comprehensive JSON
    json_file = os.path.join(output_dir, 'all_results.json')
    with open(json_file, 'w') as f:
        json.dump(all_results_json, f, indent=2)

    # Create and save summary table
    summary_df = create_summary_table(all_results)

    # Save as CSV
    csv_file = os.path.join(output_dir, 'summary_all_checkpoints.csv')
    summary_df.to_csv(csv_file, index=False, float_format='%.2f')

    # Save as formatted text
    txt_file = os.path.join(output_dir, 'summary_all_checkpoints.txt')
    with open(txt_file, 'w') as f:
        f.write("=" * 100 + "\n")
        f.write("PERPLEXITY SUMMARY - ALL CHECKPOINTS\n")
        f.write("=" * 100 + "\n\n")
        f.write(summary_df.to_string(index=False))
        f.write("\n\n" + "=" * 100 + "\n")

    print(f"\n\nResults saved to '{output_dir}/' directory:")
    print(f"  - All results (JSON): all_results.json")
    print(f"  - Detailed results: checkpoint_X_detailed.json")
    print(f"  - Summary CSV: summary_all_checkpoints.csv")
    print(f"  - Summary text: summary_all_checkpoints.txt")

    return summary_df


if __name__ == "__main__":
    # Process all checkpoint datasets
    # Adjust the pattern if your files have a different naming convention

    parser = argparse.ArgumentParser(
        description='Test fine-tuned GPT-2 model for token reversal',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '-p', '--path',
        type=str,
        required=True,
        help="Path to test data file (one example per line)"
    )

    parser.add_argument('-t', '--type',
                        type=str,
                        required=True,
                        help="Type of perturbation (wordHop, partialReverse, localShuffle, etc.)")

    args = parser.parse_args()

    print("=" * 70)
    print("GPT-2 PERPLEXITY CALCULATOR FOR MULTIPLE CHECKPOINTS")
    print("=" * 70)
    print()

    pattern = f'*_{args.type}_checkpoint-*.json'

    # Process all datasets
    all_results = process_all_datasets(pattern=pattern, model_name='gpt2')

    if all_results:
        # Save results
        summary_df = save_results(all_results, output_dir=f'perplexity_results_{args.type}')

        # Print final summary
        print("\n\n" + "=" * 100)
        print("FINAL SUMMARY - ALL CHECKPOINTS")
        print("=" * 100)
        print(summary_df.to_string(index=False))
        print("=" * 100)
    else:
        print("\nNo datasets were processed. Please check your file pattern and try again.")