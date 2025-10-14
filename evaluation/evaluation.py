import sys
import os
import glob

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from exact_match import exact_match
from bleu import bleu_score

import json
import argparse
from pathlib import Path
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from tqdm import tqdm
from utils.reverse import partial_reverse_batch
from utils.HOP import wordhop_batch
from utils.shuffle import local_shuffle_batch

metrics = {
    'exact_match': exact_match,
    'BLEU': bleu_score
}

functions = {
    "partialReverse": partial_reverse_batch,
    "localShuffle": local_shuffle_batch,
    "wordHop": wordhop_batch
}


def get_device():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    return device


DEVICE = get_device()


def load_test_data(dataset_path):
    if not Path(dataset_path).exists():
        raise FileNotFoundError(f"Test data file not found: {dataset_path}")

    lines = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                lines.append(line)

    return lines


def save_dataset(data, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(data)} examples to {output_file}")


def load_sentences_from_file(input_file):
    sentences = []

    if not Path(input_file).exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and len(line.split()) >= 3:  # Must have at least 3 tokens
                sentences.append(line)

    if not sentences:
        raise ValueError(f"No valid sentences found in {input_file}")

    print(f"Loaded {len(sentences)} sentences from {input_file}")
    return sentences


def generate_test_data(input_file, type_of_perturbation):
    print("Generating training data...")
    sentences = load_sentences_from_file(input_file)
    training_data = functions[type_of_perturbation](sentences, 512)
    return training_data


def split_into_chunks(text, tokenizer, max_chunk_size=800, overlap=100):
    """Split text into overlapping chunks that fit within model limits."""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []

    if len(tokens) <= max_chunk_size:
        return [{'text': text, 'start': 0, 'end': len(tokens)}]

    start = 0
    while start < len(tokens):
        end = min(start + max_chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        chunks.append({'text': chunk_text, 'start': start, 'end': end})

        if end >= len(tokens):
            break
        start = end - overlap

    return chunks


def merge_chunks(chunks, overlap, tokenizer):
    if len(chunks) <= 1:
        return chunks[0] if chunks else ""

    result = chunks[0]
    for i in range(1, len(chunks)):
        # Simple merge: add next chunk with space
        result = result.rstrip() + " " + chunks[i].lstrip()

    return result


def process_single_chunk(input_text, tokenizer, model, max_position_embeddings):
    prompt = f"Fix this text: {input_text}\nCorrected:"

    prompt_encoding = tokenizer(prompt, return_tensors="pt")
    input_ids = prompt_encoding['input_ids'].to(DEVICE)
    attention_mask = prompt_encoding['attention_mask'].to(DEVICE)

    input_tokens = tokenizer.encode(input_text)
    input_length = prompt_encoding['input_ids'].shape[1]

    max_new_tokens = min(len(input_tokens) + 5, max_position_embeddings - input_length)
    min_new_tokens = max(1, len(input_tokens) - 5)

    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            temperature=0.3,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated = tokenizer.decode(output[0], skip_special_tokens=True)

    if "Corrected:" in generated:
        corrected = generated.split("Corrected:")[1].strip()
    else:
        corrected = generated

    return corrected


def process_long_text(input_corrupted, tokenizer, model, max_position_embeddings):
    # Check if we need chunking
    input_tokens = tokenizer.encode(input_corrupted)
    prompt_template = "Fix this text: {}\nCorrected:"

    # Estimate prompt overhead
    prompt_overhead = len(tokenizer.encode(prompt_template.format("")))
    max_input_size = max_position_embeddings - prompt_overhead - 50  # Leave room for generation

    if len(input_tokens) <= max_input_size:
        # Process normally
        return process_single_chunk(input_corrupted, tokenizer, model, max_position_embeddings)

    # Split into chunks
    print(f"\n  Text too long ({len(input_tokens)} tokens), splitting into chunks...")
    chunks = split_into_chunks(input_corrupted, tokenizer, max_chunk_size=max_input_size, overlap=100)

    corrected_chunks = []
    for i, chunk in enumerate(chunks):
        print(f"  Processing chunk {i + 1}/{len(chunks)}...")
        corrected = process_single_chunk(chunk['text'], tokenizer, model, max_position_embeddings)
        corrected_chunks.append(corrected)

    # Merge chunks
    merged = merge_chunks(corrected_chunks, overlap=100, tokenizer=tokenizer)
    print(f"  Merged {len(chunks)} chunks into final output")

    return merged


def test_model(model_path, test_examples, metric):
    # Validate model path
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found at: {model_path}")

    print(f"\nLoading model from: {model_path}")

    # Load tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained('mission-impossible-lms/partial-reverse-gpt2')
    model = GPT2LMHeadModel.from_pretrained(model_path)
    model.config.pad_token_id = tokenizer.eos_token_id

    # Move model to the appropriate device
    model = model.to(DEVICE)
    model.eval()  # Set to evaluation mode

    # Get model's max position embeddings
    max_position_embeddings = model.config.max_position_embeddings
    print(f"Model max position embeddings: {max_position_embeddings}")

    total_count = len(test_examples)
    prediction = []
    actual = []

    with tqdm(test_examples, total=total_count) as pbar:
        for input_corrupted, test_input in pbar:

            if not input_corrupted:
                continue

            try:
                # Process with chunking if needed
                corrected = process_long_text(
                    input_corrupted,
                    tokenizer,
                    model,
                    max_position_embeddings
                )

                prediction.append(corrected)
                actual.append(test_input)

            except Exception as e:
                print(f"\nError processing example: {e}")
                print(f"Input length: {len(tokenizer.encode(input_corrupted))} tokens")
                prediction.append("")
                actual.append(test_input)
                continue

            accuracy = metrics[metric](prediction, actual)
            pbar.set_description(f"Accuracy: {accuracy:.4f}")

    final_score = metrics[metric](prediction, actual)
    print(f"\nmodel: {model_path}")
    print(f"{metric}: {final_score}")
    return final_score


def get_checkpoints_sorted(path):
    # Ensure path exists
    if not os.path.isdir(path):
        raise ValueError(f"Path does not exist or is not a directory: {path}")

    # Find all dirs matching the pattern checkpoint*
    checkpoint_dirs = [d for d in glob.glob(os.path.join(path, "checkpoint*")) if os.path.isdir(d)]

    # Sort by creation time (Oldest first)
    checkpoint_dirs.sort(key=lambda d: os.path.getctime(d), reverse=False)

    return checkpoint_dirs


def save_results(results, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved results to {output_file}")


def main(model_path, dataset_path, metric, type_of_perturbation):
    test_data_path = f"./test_data_{dataset_path.split('/')[-1].split('.')[0]}_{type_of_perturbation}.json"

    # Generate training data from input file
    print(f"Reading sentences from {dataset_path}...")
    test_examples = None

    if not Path(test_data_path).exists():
        print(f"{Path(test_data_path).resolve()} not found.\nGenerating training data from {dataset_path}...")
        test_examples = generate_test_data(
            input_file=dataset_path,
            type_of_perturbation=type_of_perturbation)
        save_dataset(test_examples, test_data_path)
    else:
        print(f"Loading training data from {Path(test_data_path).resolve()}...")
        with open(test_data_path, 'r', encoding='utf-8') as f:
            test_examples = json.load(f)

    results = {}

    # Evaluate all checkpoints
    checkpoints = get_checkpoints_sorted(model_path)
    if checkpoints:
        for checkpoint_dir in checkpoints:
            print(f"\n{'=' * 80}")
            print(f"Evaluating checkpoint: {os.path.basename(checkpoint_dir)}")
            print(f"{'=' * 80}")
            checkpoint = os.path.basename(checkpoint_dir)
            results[checkpoint] = test_model(checkpoint_dir, test_examples, metric)

    results['final'] = test_model(model_path, test_examples, metric)

    # Save results
    output_file = f"./results_{dataset_path.split('/')[-1].split('.')[0]}_{type_of_perturbation}.json"
    save_results(results, output_file)

    print(f"\n{'=' * 80}")
    print("EVALUATION SUMMARY")
    print(f"{'=' * 80}")
    for key, value in results.items():
        print(f"{key}: {value:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Test fine-tuned GPT-2 model for token reversal',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '-m', '--model',
        type=str,
        required=True,
        help="Path to fine-tuned model directory"
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

    parser.add_argument("--metric",
                        type=str,
                        default="exact_match",
                        help="Metric to use for evaluation. Default: exact_match. Options: exact_match, BLEU")

    args = parser.parse_args()

    main(args.model, args.path, args.metric, args.type)
