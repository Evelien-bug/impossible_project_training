import sys
import os


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pathlib import Path
import torch
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
)
import json
import yaml
from datasets import Dataset
import argparse
from tqdm import tqdm
from utils.reverse import partial_reverse_batch
from utils.HOP import wordhop_batch
from utils.shuffle import local_shuffle_batch

functions = {
    "partialReverse": partial_reverse_batch,
    "localShuffle": local_shuffle_batch,
    "wordHop": wordhop_batch
}


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


def generate_training_data(input_file, type_of_perturbation):
    print("Generating training data...")
    sentences = load_sentences_from_file(input_file)
    training_data = functions[type_of_perturbation](sentences)
    return training_data

def save_dataset(data, output_file):
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

            # Create labels with masking
            # -100 = ignored by loss function (prompt part)
            # actual token IDs = used for loss computation (output part)
            # Find the actual position of "Corrected:" in the tokenized output
            corrected_token_ids = tokenizer.encode("Corrected:", add_special_tokens=False)
            position = None
            for i in range(len(encoded['input_ids']) - len(corrected_token_ids)):
                if encoded['input_ids'][i:i + len(corrected_token_ids)] == corrected_token_ids:
                    position = i + len(corrected_token_ids)
                    break

            if position is None:
                # Fallback to your original method
                position = prompt_length

            # Create labels
            labels = [-100] * position + encoded['input_ids'][position:]
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


def main(config, input_file, model_name, type_of_perturbation):
    OUTPUT_DIR = config.get('training_arguments', {}).get('output_dir', None)
    if not OUTPUT_DIR:
        raise ValueError("Output directory must be specified in training_arguments.output_dir")
    training_data_path = f"./training_data_{input_file.split('/')[-1].split('.')[0]}_{type_of_perturbation}.json"

    # Generate training data from input file
    print(f"Reading sentences from {input_file}...")
    training_data = None
    if not Path(training_data_path).exists():
        print(f"{Path(training_data_path).resolve()} not found.\n Generating training data from {input_file}...")
        training_data = generate_training_data(
            input_file=input_file,
            type_of_perturbation=type_of_perturbation)
        save_dataset(training_data, training_data_path)
    else:
        print(f"Loading training data from {Path(training_data_path).resolve()}...")
        with open(training_data_path, 'r', encoding='utf-8') as f:
            training_data = json.load(f)

    # Prepare tokenizer and datasets with masked labels
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

    # Train model
    train_model(
        train_dataset,
        eval_dataset,
        config,
        model_name=model_name,
        output_dir=OUTPUT_DIR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-p', '--path', type=str, required=True,
                        help="Path to input sentences file")
    parser.add_argument('-c', '--config', type=str, required=True,
                        help="Path to YAML configuration file")
    parser.add_argument('-t', '--type', type=str, required=True,
                        help="Type of perturbation (wordHop, partialReverse, localShuffle, etc.)")

    args = parser.parse_args()
    config = load_configs(args.config)
    if args.type == 'wordHop':
        model = 'mission-impossible-lms/word-hop-gpt2'
    elif args.type == 'partialReverse':
        model = 'mission-impossible-lms/partial-reverse-gpt2'
    elif args.type == 'localShuffle':
        model = 'mission-impossible-lms/local-shuffle-w3-gpt2'
    else:
        raise ValueError("Invalid perturbation type. Choose from 'wordHop', 'partialReverse', or 'localShuffle'.")
    main(config=config, input_file=args.path, model_name=model, type_of_perturbation=args.type)