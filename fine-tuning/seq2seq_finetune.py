import random
import json
from pathlib import Path
import torch
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
)
import yaml
from datasets import Dataset
import argparse


def load_configs(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


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


def create_reversal_example(text, marker='🅁'):
    tokens = text.split()
    if len(tokens) < 3:
        return None

    split_idx = random.randint(1, len(tokens) - 2)
    before = tokens[:split_idx]
    after = tokens[split_idx:]

    corrupted = ' '.join(before) + marker + ' ' + ' '.join(reversed(after))
    original = text

    return corrupted, original


def load_sentences_from_file(input_file):
    sentences = []

    if not Path(input_file).exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and len(line.split()) >= 3:
                sentences.append(line)

    if not sentences:
        raise ValueError(f"No valid sentences found in {input_file}")

    print(f"Loaded {len(sentences)} sentences from {input_file}")
    return sentences


def generate_training_data(input_file, marker='🅁'):
    training_data = []
    sentences = load_sentences_from_file(input_file)

    for sentence in sentences:
        example = create_reversal_example(sentence, marker)
        if example:
            training_data.append(example)

    return training_data


def save_dataset(data, output_file='training_data_seq2seq.json'):
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(data)} examples to {output_file}")


def prepare_seq2seq_dataset(training_data, tokenizer, train_split=0.9, max_length=128):
    # Split into train and eval
    split_idx = int(len(training_data) * train_split)
    train_data = training_data[:split_idx]
    eval_data = training_data[split_idx:]

    def process_data(data):
        """Process data in seq2seq style."""
        input_ids_list = []
        attention_mask_list = []
        labels_list = []

        for corrupted, correct in data:
            # SEQ2SEQ APPROACH:
            # Input: corrupted text (with 🅁 marker)
            # Output: corrected text
            # NO INSTRUCTION PROMPT
            # NO MASKING

            # Tokenize input (corrupted)
            input_encoded = tokenizer(
                corrupted,
                truncation=True,
                max_length=max_length,
                padding='max_length',
                return_tensors=None
            )

            # Tokenize output (correct) - this becomes labels
            output_encoded = tokenizer(
                correct,
                truncation=True,
                max_length=max_length,
                padding='max_length',
                return_tensors=None
            )

            # CRITICAL DIFFERENCE: No masking! All tokens get loss
            # In masked approach: labels = [-100] * prompt_length + tokens
            # In seq2seq: labels = all_output_tokens
            labels = output_encoded['input_ids']

            # Replace padding token id with -100 (standard practice)
            labels = [
                token_id if token_id != tokenizer.pad_token_id else -100
                for token_id in labels
            ]

            input_ids_list.append(input_encoded['input_ids'])
            attention_mask_list.append(input_encoded['attention_mask'])
            labels_list.append(labels)

        return {
            'input_ids': input_ids_list,
            'attention_mask': attention_mask_list,
            'labels': labels_list
        }

    print("Processing training data with seq2seq approach (traditional method)...")
    print("NOTE: This method is LESS efficient than masked labels!")
    train_processed = process_data(train_data)
    eval_processed = process_data(eval_data)

    # Create datasets
    train_dataset = Dataset.from_dict(train_processed)
    eval_dataset = Dataset.from_dict(eval_processed)

    # Set format for PyTorch
    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    eval_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    print(f"✓ Train samples: {len(train_dataset)}")
    print(f"✓ Eval samples: {len(eval_dataset)}")

    return train_dataset, eval_dataset


def train_model(train_dataset, eval_dataset, config, model_name, output_dir):
    # Load model and tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(model_name)

    # Move model to device
    model = model.to(DEVICE)

    # Print model size
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Get training arguments from config
    training_config = config.get('training_arguments', {})
    training_args = TrainingArguments(**training_config)

    # Initialize trainer
    # Note: No data_collator needed - labels already prepared
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # Train
    print("Starting training (seq2seq method)...\n")
    trainer.train()

    # Save final model
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")

    return model, tokenizer


def main(config, input_file, model_name):
    MARKER = '🅁'
    OUTPUT_DIR = config.get('training_arguments', {}).get('output_dir', './gpt2-seq2seq')
    training_data = generate_training_data(input_file=input_file, marker=MARKER)

    print()

    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    train_dataset, eval_dataset = prepare_seq2seq_dataset(
        training_data,
        tokenizer,
        max_length=128
    )
    print()

    train_model(
        train_dataset,
        eval_dataset,
        config,
        model_name=model_name,
        output_dir=OUTPUT_DIR
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--model', type=str, required=True,
                        help="Model name or path (e.g., 'gpt2', 'gpt2-medium')")
    parser.add_argument('-p', '--path', type=str, required=True,
                        help="Path to input sentences file")
    parser.add_argument('-c', '--config', type=str, required=True,
                        help="Path to YAML configuration file")
    parser.add_argument('-t', '--type', type=str, required=True,
                        help="Type of perturbation (wordHop, partialReverse, localShuffle, etc.)")

    args = parser.parse_args()
    config = load_configs(args.config)

    main(config=config, input_file=args.path, model_name=args.model, type_of_perturbation=args.type)