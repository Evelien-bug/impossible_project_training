from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import argparse
from pathlib import Path


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

def test_model(model_path, test_examples):
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

    for i, test_input in enumerate(test_examples, 1):
        # IMPORTANT: Use the same prompt format as training
        prompt = f"Fix this text: {test_input}\nCorrected:"

        # Tokenize input
        input_tokens = tokenizer.encode(test_input)
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

        # Display results
        print(f"Test {i}/{total_count}")
        print("-" * 80)
        print(f"Input:  {test_input}")
        print(f"Output: {corrected}")
        print()

    print("-" * 80)
    print(f"✅ Testing complete! Processed {total_count} examples")
    print("-" * 80 + "\n")


def create_test_example(text, marker='🅁'):
    import random

    tokens = text.split()
    if len(tokens) < 3:
        return None

    # Random split point
    split_idx = random.randint(1, len(tokens) - 2)

    before = tokens[:split_idx]
    after = tokens[split_idx:]

    # Create corrupted version
    corrupted = ' '.join(before) + marker + ' ' + ' '.join(reversed(after))
    original = text

    return corrupted, original


def load_test_data(dataset_path, mode='auto', marker='🅁'):
    if not Path(dataset_path).exists():
        raise FileNotFoundError(f"Test data file not found: {dataset_path}")

    lines = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                lines.append(line)

    if not lines:
        raise ValueError(f"No test data found in: {dataset_path}")

    # Auto-detect mode
    if mode == 'auto':
        # Check if first line has the marker
        if marker in lines[0]:
            mode = 'corrupted'
        else:
            mode = 'original'

    if mode == 'corrupted':
        # Data already has 🅁 marker, use as-is
        print(f"✓ Loaded {len(lines)} corrupted test examples from {dataset_path}")
        return lines

    elif mode == 'original':
        # Generate corrupted versions from original sentences
        test_data = []
        for line in lines:
            example = create_test_example(line, marker)
            if example:
                test_data.append(example)

        print(f"✓ Loaded {len(lines)} original sentences")
        print(f"✓ Generated {len(test_data)} corrupted test examples")
        return [corrupted for corrupted, _ in test_data]

    else:
        raise ValueError(f"Invalid mode: {mode}. Use 'auto', 'corrupted', or 'original'")


def main(model_path, dataset_path, mode='auto'):
    print("\n" + "=" * 80)
    print("GPT-2 TOKEN REVERSAL - TESTING")
    print("=" * 80)

    # Load test data
    test_samples = load_test_data(dataset_path, mode=mode)

    # Test model
    test_model(model_path, test_samples)


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
    parser.add_argument(
        '--mode',
        type=str,
        default='auto',
        choices=['auto', 'corrupted', 'original'],
        help="Data mode: 'auto' (detect), 'corrupted' (has 🅁), 'original' (will add 🅁)"
    )

    args = parser.parse_args()

    main(args.model, args.path, args.mode)