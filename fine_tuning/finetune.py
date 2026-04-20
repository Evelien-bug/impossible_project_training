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

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.utils import load_configs, get_device

DEVICE = get_device()


def prepare_dataset(training_data, tokenizer, train_split=0.9, max_length=128):
    split_idx = int(len(training_data) * train_split)
    train_data = training_data[:split_idx]
    eval_data = training_data[split_idx:]

    def process_data(data):
        input_ids_list = []
        attention_mask_list = []
        labels_list = []

        for corrupted, correct in tqdm(data):
            full_text = f"Fix this text: {corrupted}\nCorrected: {correct}<|endoftext|>"

            encoded = tokenizer(
                full_text,
                truncation=True,
                max_length=max_length,
                padding='max_length',
                return_tensors=None
            )

            prompt_text = f"Fix this text: {corrupted}\nCorrected:"
            prompt_encoded = tokenizer(
                prompt_text,
                truncation=True,
                max_length=max_length,
                add_special_tokens=True,
                return_tensors=None
            )
            prompt_length = len(prompt_encoded['input_ids'])

            corrected_token_ids = tokenizer.encode("Corrected:", add_special_tokens=False)
            position = None
            for i in range(len(encoded['input_ids']) - len(corrected_token_ids)):
                if encoded['input_ids'][i:i + len(corrected_token_ids)] == corrected_token_ids:
                    position = i + len(corrected_token_ids)
                    break

            if position is None:
                position = prompt_length

            labels = [-100] * position + encoded['input_ids'][position:]
            labels = labels[:max_length]
            if len(labels) < max_length:
                labels = labels + [-100] * (max_length - len(labels)]

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

    train_dataset = Dataset.from_dict(train_processed)
    eval_dataset = Dataset.from_dict(eval_processed)

    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    eval_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    return train_dataset, eval_dataset


def train_model(
        train_dataset,
        eval_dataset,
        config,
        model_name,
        output_dir,
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


def main(config, input_file, model_name):
    OUTPUT_DIR = config.get('training_arguments', {}).get('output_dir', None)
    if not OUTPUT_DIR:
        raise ValueError("Output directory must be specified in training_arguments.output_dir")

    print(f"Loading training data from {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
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
                        help="Path to pre-generated perturbed dataset")
    parser.add_argument('-c', '--config', type=str, required=True,
                        help="Path to YAML configuration file")

    args = parser.parse_args()
    config = load_configs(args.config)
    model = 'gpt2-medium'

    main(config=config, input_file=args.path, model_name=model)
