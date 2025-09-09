import argparse
import json
import yaml

import matplotlib.pyplot as plt
import torch
from datasets import Dataset, DatasetDict, load_from_disk
from transformers import  GPT2LMHeadModel, GPT2Config, TrainingArguments, Trainer, AddedToken

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



def create_gpt_from_scratch(config, vocab_size):
    model_config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=config.get('n_positions', 1024),  # Context length
        n_embd=config.get('n_embd', 768),  # Embedding dimension
        n_layer=config.get('n_layer', 12),  # Number of transformer layers
        n_head=config.get('n_head', 12),  # Number of attention heads
        n_inner=config.get('n_inner', 3072),  # Feed-forward dimension
        activation_function=config.get('activation_function', 'gelu_new'),
        resid_pdrop=config.get('resid_pdrop', 0.1),
        embd_pdrop=config.get('embd_pdrop', 0.1),
        attn_pdrop=config.get('attn_pdrop', 0.1),
        layer_norm_epsilon=config.get('layer_norm_epsilon', 1e-5),
        initializer_range=config.get('initializer_range', 0.02),
        summary_type=config.get('summary_type', 'cls_index'),
        summary_use_proj=config.get('summary_use_proj', True),
        summary_activation=config.get('summary_activation', None),
        summary_proj_to_labels=config.get('summary_proj_to_labels', True),
        summary_first_dropout=config.get('summary_first_dropout', 0.1),
        use_cache=config.get('use_cache', True),
        bos_token_id=config.get('bos_token_id', 50256),
        eos_token_id=config.get('eos_token_id', 50256),
    )

    print(f"Creating model from scratch with configuration:")
    print(f"  - Vocab size: {vocab_size}")
    print(f"  - Embedding dimension: {model_config.n_embd}")
    print(f"  - Number of layers: {model_config.n_layer}")
    print(f"  - Number of attention heads: {model_config.n_head}")
    print(f"  - Context length: {model_config.n_positions}")

    model = GPT2LMHeadModel(config=model_config)
    model.apply(model._init_weights)

    return model

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
    parser.add_argument('-t', '--type', type=str, required=True,
                        help="Type of perturbation")

    args = parser.parse_args()
    configs = load_configs(args.config)
    perturb_type = args.type
    tokenizer = None

    if perturb_type == 'hop':
        tokenizer = utils.gpt2_hop_tokenizer
    elif perturb_type == 'reverse':
        tokenizer = utils.gpt2_rev_tokenizer
    elif perturb_type == 'shuffle':
        tokenizer = utils.gpt2_original_tokenizer

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    vocab_size = len(tokenizer)

    model = create_gpt_from_scratch(configs.get('model_config', {}), vocab_size)
    model.to(device)

    dataset = load_from_disk(args.path)

    training_args = TrainingArguments(**configs.get('training_arguments', {}))

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation']
    )

    trainer.train()
    model = model.merge_and_unload()
    model.save_pretrained('./fine_tuned_model')