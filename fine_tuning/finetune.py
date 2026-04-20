import json
import argparse
from pathlib import Path
import sys

import torch
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset
from tqdm import tqdm

# Add parent directory to path to allow imports from utils
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.utils import load_sentences_from_file, save_dataset, load_configs, get_device
from utils.reverse import partial_reverse_batch
from utils.hop import wordhop_batch
from utils.shuffle import local_shuffle_batch, local_shuffle_batch_with_window, full_shuffle_batch

functions = {
    "partialReverse": partial_reverse_batch,
    "localShuffle": local_shuffle_batch,
    "localShuffle3": lambda texts: local_shuffle_batch_with_window(texts, window_size=3),
    "localShuffle5": lambda texts: local_shuffle_batch_with_window(texts, window_size=5),
    "fullShuffle": lambda texts: full_shuffle_batch(texts, seed=57),
    "wordHop": wordhop_batch
}

DEVICE = get_device()


def generate_training_data(input_file, type_of_perturbation):
    print("Generating training data...")
    sentences = load_sentences_from_file(input_file)
    training_data = functions[type_of_perturbation](sentences)
    return training_data


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
            corrected_token_ids = tokenizer.encode("Corrected:", add_special_tokens=False)
            position = None
            for i in range(len(encoded['input_ids']) - len(corrected_token_ids)):
                if encoded['input_ids'][i:i + len(corrected_token_ids)] == corrected_token_ids:
                    position = i + len(corrected_token_ids)
                    break

            if position is None:
                position = prompt_length

            # Create labels
            labels = [-100] * position + encoded['input_ids'][position:]
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

    # Set format for PyTorch
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
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model = model.to(DEVICE)

    training_config = config.get('training_arguments', {})
    training_args = TrainingArguments(**training_config)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    print("Starting training...")
    trainer.train()

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")

    return model, tokenizer


def main(config, input_file, model_name, type_of_perturbation):
    OUTPUT_DIR = config.get('training_arguments', {}).get('output_dir', None)
    if not OUTPUT_DIR:
        raise ValueError("Output directory must be specified in training_arguments.output_dir")
    training_data_path = f"./training_data_{input_file.split('/')[-1].split('.')[0]}_{type_of_perturbation}.json"

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
                        help="Type of perturbation (wordHop, partialReverse, localShuffle3, localShuffle5, fullShuffle etc.)")

    args = parser.parse_args()
    config = load_configs(args.config)
    model = 'gpt2-medium'
