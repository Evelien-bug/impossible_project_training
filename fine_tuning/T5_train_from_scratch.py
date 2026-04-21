import json
import argparse
from pathlib import Path
import sys

import torch
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    T5Config,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)
from datasets import Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.utils import load_configs, get_device

DEVICE = get_device()


def prepare_dataset(training_data, tokenizer, train_split=0.9,
                    max_input_length=128, max_target_length=128):
    split_idx = int(len(training_data) * train_split)
    train_data = training_data[:split_idx]
    eval_data = training_data[split_idx:]

    def process_data(data):
        input_ids_list = []
        attention_mask_list = []
        labels_list = []

        for corrupted, correct in tqdm(data):
            # Encode the perturbed sentence as encoder input.
            # No prompt template needed — T5's architecture handles conditioning.
            input_encoded = tokenizer(
                corrupted,
                truncation=True,
                max_length=max_input_length,
                padding='max_length',
                return_tensors=None,
            )

            # Encode the original sentence as decoder target.
            target_encoded = tokenizer(
                correct,
                truncation=True,
                max_length=max_target_length,
                padding='max_length',
                return_tensors=None,
            )

            # Replace pad token ids in labels with -100 so they're ignored by loss.
            labels = [
                token_id if token_id != tokenizer.pad_token_id else -100
                for token_id in target_encoded['input_ids']
            ]

            input_ids_list.append(input_encoded['input_ids'])
            attention_mask_list.append(input_encoded['attention_mask'])
            labels_list.append(labels)

        return {
            'input_ids': input_ids_list,
            'attention_mask': attention_mask_list,
            'labels': labels_list,
        }

    print("Processing training data for T5 seq2seq...")
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
    tokenizer = T5Tokenizer.from_pretrained(model_name)

    # From-scratch initialization: load only the config, not the weights.
    config_model = T5Config.from_pretrained(model_name)
    model = T5ForConditionalGeneration(config_model)
    model = model.to(DEVICE)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    training_config = config.get('training_arguments', {})
    training_args = TrainingArguments(**training_config)

    # Data collator handles dynamic padding and proper label shifting for T5.
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
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
    tokenizer = T5Tokenizer.from_pretrained(model_name)

    train_dataset, eval_dataset = prepare_dataset(
        training_data,
        tokenizer,
        max_input_length=128,
        max_target_length=128,
    )
    print(f"Train samples: {len(train_dataset)}")
    print(f"Eval samples: {len(eval_dataset)}")

    train_model(
        train_dataset,
        eval_dataset,
        config,
        model_name=model_name,
        output_dir=OUTPUT_DIR,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-p', '--path', type=str, required=True,
                        help="Path to pre-generated perturbed dataset")
    parser.add_argument('-c', '--config', type=str, required=True,
                        help="Path to YAML configuration file")

    args = parser.parse_args()
    config = load_configs(args.config)
    model = 't5-base'

    main(config=config, input_file=args.path, model_name=model)
