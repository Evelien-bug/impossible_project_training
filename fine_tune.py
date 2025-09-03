import argparse
import json
import yaml

import matplotlib.pyplot as plt
import torch
from datasets import Dataset, DatasetDict, load_from_disk
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TrainingArguments, Trainer, AddedToken, AutoModelForSeq2SeqLM
from peft import LoraConfig, get_peft_model

import utils

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    torch.cuda.set_device(0)
    torch.cuda.empty_cache()
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

def load_dataset(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    train_dataset = Dataset.from_list(data["train"])
    valid_dataset = Dataset.from_list(data["validation"])

    dataset = DatasetDict({
        "train": train_dataset,
        "validation": valid_dataset
    })

    return dataset


def get_gpt2_tokenizer_with_markers():
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    new_tokens = [
        AddedToken(utils.MARKER_HOP_SING, lstrip=True, rstrip=False),
        AddedToken(utils.MARKER_HOP_PLUR, lstrip=True, rstrip=False)
    ]
    tokenizer.add_tokens(new_tokens)
    return tokenizer


def preprocess(tokenizer, device):
    def tokenize(examples):
        inputs = tokenizer(
            examples['perturbed_text'],
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )
        labels = tokenizer(
            examples['original_text'],
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )['input_ids']

        inputs['labels'] = labels

        inputs = {k: v.to(device) for k, v in inputs.items()}
        return inputs

    return tokenize
class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_perplexities = []
        self.eval_perplexities = []

    def compute_perplexity(self, dataset):
        total_loss = 0
        total_length = 0

        for batch in dataset:
            with torch.no_grad():
                outputs = self.model(input_ids=batch['input_ids'].to(self.model.device),
                                     labels=batch['labels'].to(self.model.device))
                total_loss += outputs.loss.item() * batch['input_ids'].size(1)
                total_length += batch['input_ids'].size(1)

        return torch.exp(torch.tensor(total_loss / total_length)).item()

    def on_epoch_end(self, args, state, control, **kwargs):
        train_perplexity = self.compute_perplexity(self.train_dataset)
        eval_perplexity = self.compute_perplexity(self.eval_dataset)

        self.train_perplexities.append(train_perplexity)
        self.eval_perplexities.append(eval_perplexity)

        print(f"Epoch {state.epoch}:")
        print(f"Train Perplexity: {train_perplexity:.2f}")
        print(f"Eval Perplexity: {eval_perplexity:.2f}")

        plt.figure(figsize=(10, 6))
        plt.plot(self.train_perplexities, label='Train', marker='o')
        plt.plot(self.eval_perplexities, label='Validation', marker='s')
        plt.title('Training Progress')
        plt.xlabel('Epoch')
        plt.ylabel('Perplexity')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'perplexity_plot_epoch_{state.epoch}.png')
        plt.close()

        return super().on_epoch_end(args, state, control, **kwargs)

def load_configs(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-p', '--path', type=str, required=True,
                        help="Path to file")
    parser.add_argument('-c', '--config', type=str, required=True,
                        help="Path to YAML configuration file")
    args = parser.parse_args()
    config = load_configs(args.config)

    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model.to(device)

    dataset = load_from_disk(args.path)
    lora_config = config.get('lora_config', {})

    model = get_peft_model(model, LoraConfig(**lora_config))

    training_config = config.get('training_arguments', {})
    training_args = TrainingArguments(**training_config)

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation']
    )

    trainer.train()
    model = model.merge_and_unload()
    model.save_pretrained('./fine_tuned_model')