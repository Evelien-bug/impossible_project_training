import sys
import os


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from exact_match import exact_match
from bleu import bleu_score

import argparse
from pathlib import Path
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from tqdm import tqdm
from utils.reverse import partial_reverse_in_batch

metrics = {
    'exact_match': exact_match,
    'BLEU': bleu_score
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


def create_test_example(text, marker='🅁'):
    import random

    tokens = text.split()
    if len(tokens) < 3:
        return None, None

    # Random split point
    split_idx = random.randint(1, len(tokens) - 2)

    before = tokens[:split_idx]
    after = tokens[split_idx:]

    # Create corrupted version
    corrupted = ' '.join(before) + marker + ' ' + ' '.join(reversed(after))
    original = text

    return corrupted, original


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


def test_model(model_path, test_examples, metric):
    # Validate model path
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found at: {model_path}")

    print(f"\nLoading model from: {model_path}")

    # Load tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained('mission-impossible-lms/partial-reverse-gpt2')
    model = GPT2LMHeadModel.from_pretrained(model_path)

    # Move model to the appropriate device
    model = model.to(DEVICE)
    model.eval()  # Set to evaluation mode

    print("\n" + "=" * 80)
    print("TESTING MODEL")
    print(f"Using device: {DEVICE}")
    print("=" * 80 + "\n")

    total_count = len(test_examples)
    prediction = []
    actual = []
    inputs_corrupted = partial_reverse_in_batch(test_examples, batch_size=512)
    for input_corrupted, test_input in tqdm(inputs_corrupted):

        if not input_corrupted:
            continue

        prompt = f"Fix this text: {input_corrupted}\nCorrected:"

        # Tokenize input
        input_tokens = tokenizer.encode(input_corrupted)
        prompt_encoding = tokenizer(prompt, return_tensors="pt")

        # Move input to device
        input_ids = prompt_encoding['input_ids'].to(DEVICE)
        attention_mask = prompt_encoding['attention_mask'].to(DEVICE)

        # Calculate max_new_tokens strictly
        max_new_tokens = len(input_tokens) + 5
        min_new_tokens = max(1, len(input_tokens) - 5)

        # Generate
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

        # Decode output
        generated = tokenizer.decode(output[0], skip_special_tokens=True)

        # Extract just the corrected part
        if "Corrected:" in generated:
            corrected = generated.split("Corrected:")[1].strip()
        else:
            corrected = generated
        prediction.append(corrected)
        actual.append(test_input)
        if test_input.strip() != corrected.strip():
            print(f"Perturbed:\t{input_corrupted}")
            print(f"Prediction:\t{corrected}")
            print(f"Actual:\t\t{test_input}")
            print(f"{metrics[metric]}: {metrics[metric](prediction, actual)}")
            print("-" * 40)

    print(f"{metrics[metric]}: {metrics[metric](prediction, actual)}")


def main(model_path, dataset_path, metric):
    print("\n" + "=" * 80)
    print("GPT-2 TOKEN REVERSAL - TESTING")
    print("=" * 80)

    # Load test data
    data = load_test_data(dataset_path)

    # Test model
    test_model(model_path, data, metric)


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

    parser.add_argument("--metric",
                        type=str,
                        default="exact_match",
                        help="Metric to use for evaluation. Default: exact_match. Options: exact_match, BELU")
    args = parser.parse_args()

    main(args.model, args.path, args.metric)
