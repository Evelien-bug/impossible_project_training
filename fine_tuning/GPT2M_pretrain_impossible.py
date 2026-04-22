import json
import argparse
from pathlib import Path
import sys

import torch
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    GPT2Config,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from datasets import Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.utils import load_configs, get_device

DEVICE = get_device()


def prepare_dataset(perturbed_sentences, tokenizer, train_split=0.9, max_length=512):
    """
    Tokenize perturbed sentences for standard causal language modeling.
    Stage 1: no pairs, no prompts, no masking — just perturbed text.
    """
    split_idx = int(len(perturbed_sentences) * train_split)
    train_data = perturbed_sentences[:split_idx]
    eval_data = perturbed_sentences[split_idx:]

    def process_data(sentences):
        input_ids_list = []
        attention_mask_list = []

        for text in tqdm(sentences):
            encoded = tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                padding='max_length',
                return_tensors=None,
            )
            input_ids_list.append(encoded['input_ids'])
            attention_mask_list.append(encoded['attention_mask'])

        return {
            'input_ids': input_ids_list,
            'attention_mask': attention_mask_list,
        }

    print("Tokenizing perturbed sentences for stage 1 pretraining...")
    train_processed = process_data(train_data)
    eval_processed = process_data(eval_data)

    train_dataset = Dataset.from_dict(train_processed)
    eval_dataset = Dataset.from_dict(eval_processed)

    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    eval_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])

    return train_dataset, eval_dataset


def extract_perturbed_sentences(pair_data):
    """
    Extract only the perturbed side of (perturbed, original) pairs.
    Stage 1 pretraining only needs the impossible language, not the reconstruction target.
    """
    return [perturbed for perturbed, _original in pair_data]


def train_model(train_dataset, eval_dataset, config, model_name, output_dir):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # From-scratch: load config only, not pretrained weights.
    config_model = GPT2Config.from_pretrained(model_name)
    model = GPT2LMHeadModel(config_model)
    model = model.to(DEVICE)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    training_config = config.get('training_arguments', {})
    training_args = TrainingArguments(**training_config)

    # For causal LM, the collator shifts input_ids to create labels automatically.
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM, not masked LM.
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    print("Starting stage 1 pretraining on impossible language...")
    trainer.train()

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Stage 1 model saved to {output_dir}")

    return model, tokenizer


def main(config, input_file, model_name, max_samples=None):
    OUTPUT_DIR = config.get('training_arguments', {}).get('output_dir', None)
    if not OUTPUT_DIR:
        raise ValueError("Output directory must be specified in training_arguments.output_dir")

    print(f"Loading data from {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        pair_data = json.load(f)

    if max_samples is not None:
        pair_data = pair_data[:max_samples]
        print(f"Limiting to first {max_samples} samples for pilot run.")

    # For stage 1, we only need the perturbed sentences, not the originals.
    perturbed_sentences = extract_perturbed_sentences(pair_data)
    print(f"Extracted {len(perturbed_sentences)} perturbed sentences for pretraining.")

    print("\nPreparing datasets...")
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    train_dataset, eval_dataset = prepare_dataset(
        perturbed_sentences,
        tokenizer,
        max_length=512,
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
                        help="Path to perturbed dataset (will extract perturbed side only)")
    parser.add_argument('-c', '--config', type=str, required=True,
                        help="Path to YAML configuration file")
    parser.add_argument('--max-samples', type=int, default=None,
                        help="Limit dataset to first N samples (for pilot runs)")

    args = parser.parse_args()
    config = load_configs(args.config)
    model = 'gpt2-medium'

    main(
        config=config,
        input_file=args.path,
        model_name=model,
        max_samples=args.max_samples,
    )
