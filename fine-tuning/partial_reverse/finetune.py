import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
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
from tqdm import tqdm
from utils.reverse import partial_reverse


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


def generate_training_data(input_file, marker='🅁'):
    training_data = []

    sentences = load_sentences_from_file(input_file)

    for sentence in sentences:
        example = partial_reverse(sentence)
        if example:
            training_data.append((example, sentences))

    return training_data


def save_dataset(data, output_file='training_data.json'):
    """Save dataset to JSON file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(data)} examples to {output_file}")


def prepare_dataset(training_data, tokenizer, train_split=0.9, max_length=128):

    # Split into train and eval
    split_idx = int(len(training_data) * train_split)
    train_data = training_data[:split_idx]
    eval_data = training_data[split_idx:]

    def process_data(data):
        """Process data with masked labels."""
        input_ids_list = []
        attention_mask_list = []
        labels_list = []

        for corrupted, correct in tqdm(data):
            # INSTRUCTION FORMAT - tells GPT-2 what to do
            full_text = f"Fix this text: {corrupted}\nCorrected: {correct}<|endoftext|>"

            # Tokenize full text
            encoded = tokenizer(
                full_text,
                truncation=True,
                max_length=max_length,
                padding='max_length',
                return_tensors=None
            )

            # Find where "Corrected:" starts (where we want to start computing loss)
            prompt_text = f"Fix this text: {corrupted}\nCorrected:"
            prompt_encoded = tokenizer(
                prompt_text,
                truncation=True,
                max_length=max_length,
                add_special_tokens=True,
                return_tensors=None
            )
            prompt_length = len(prompt_encoded['input_ids'])

            # CRITICAL: Create labels with masking
            # -100 = ignored by loss function (prompt part)
            # actual token IDs = used for loss computation (output part)
            labels = [-100] * prompt_length + encoded['input_ids'][prompt_length:]

            # Ensure correct length
            labels = labels[:max_length]
            if len(labels) < max_length:
                labels = labels + [-100] * (max_length - len(labels))

            input_ids_list.append(encoded['input_ids'])
            attention_mask_list.append(encoded['attention_mask'])
            labels_list.append(labels)

        return {
            'input_ids': input_ids_list,
            'attention_mask': attention_mask_list,
            'labels': labels_list
        }

    print("Processing training data with masked labels...")
    train_processed = process_data(train_data)
    eval_processed = process_data(eval_data)

    # Create datasets
    train_dataset = Dataset.from_dict(train_processed)
    eval_dataset = Dataset.from_dict(eval_processed)

    # Set format for PyTorch - IMPORTANT: include 'labels' column
    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    eval_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    return train_dataset, eval_dataset


def train_model(
        train_dataset,
        eval_dataset,
        config,
        model_name,
        output_dir='./gpt2-reversal',
):
    # Load model and tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(model_name)

    # Move model to device (MPS/CUDA/CPU)
    model = model.to(DEVICE)

    # NO data_collator - we handle labels ourselves in prepare_dataset
    # This is important: DataCollatorForLanguageModeling would interfere with our custom masked labels

    training_config = config.get('training_arguments', {})
    training_args = TrainingArguments(**training_config)

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # Train
    print("Starting training...")
    trainer.train()

    # Save final model
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")

    return model, tokenizer


def main(config, input_file='input_sentences.txt', model_name='gpt2'):
    MARKER = '🅁'
    OUTPUT_DIR = config.get('training_arguments', {}).get('output_dir', './gpt2-reversal')

    # Step 1: Generate training data from input file
    print(f"Reading sentences from {input_file}...")
    training_data = generate_training_data(
        input_file=input_file,
        marker=MARKER)

    # Optionally save the dataset
    # save_dataset(training_data, 'training_data.json')

    # Step 2: Prepare tokenizer and datasets with masked labels
    print("\nPreparing datasets...")
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    train_dataset, eval_dataset = prepare_dataset(
        training_data,
        tokenizer,
        max_length=128
    )
    print(f"Train samples: {len(train_dataset)}")
    print(f"Eval samples: {len(eval_dataset)}")

    # Step 3: Train model
    train_model(
        train_dataset,
        eval_dataset,
        config,
        model_name=model_name,
        output_dir=OUTPUT_DIR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--model', type=str, required=True,
                        help="Model name or path (e.g., 'gpt2', 'gpt2-medium')")
    parser.add_argument('-p', '--path', type=str, required=True,
                        help="Path to input sentences file")
    parser.add_argument('-c', '--config', type=str, required=True,
                        help="Path to YAML configuration file")
    args = parser.parse_args()
    config = load_configs(args.config)

    main(config=config, input_file=args.path, model_name=args.model)