import argparse
import json

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-p', '--path', type=str, required=True,
                        help="Path to file")
    # parser.add_argument('-t', '--type', type=str, required=True,
    #                     help="Type of perturbation")
    args = parser.parse_args()

    # data = load_dataset(args.path)
    # perturb_type = args.types

    # tokenizer = None

    # if perturb_type == 'hop':
    #     tokenizer = utils.gpt2_hop_tokenizer
    # elif perturb_type == 'reverse':
    #     tokenizer = utils.gpt2_rev_tokenizer
    # elif perturb_type == 'shuffle':
    #     tokenizer = utils.gpt2_original_tokenizer
    #
    # if tokenizer.pad_token is None:
    #     tokenizer.pad_token = tokenizer.eos_token

    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model.to(device)

    # tokenize = preprocess(tokenizer, device)
    #
    # dataset = data.map(
    #     tokenize,
    #     batched=True,
    #     batch_size=2000,
    #     remove_columns=['perturbed_text', 'original_text'],
    #     num_proc=None,
    #     load_from_cache_file=True,
    #     desc="Tokenizing dataset"
    # )
    dataset = load_from_disk(args.path)

    config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=["c_attn", "c_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        fan_in_fan_out=True
    )

    model = get_peft_model(model, config)

    training_args = TrainingArguments(
        output_dir="./gpt2-lora-translation",
        per_device_train_batch_size=24,
        per_device_eval_batch_size=24,
        gradient_accumulation_steps=2,
        save_strategy="steps",
        logging_strategy="steps",
        learning_rate=3e-4,
        num_train_epochs=3,
        save_steps=300,
        logging_steps=75,
        report_to="none",
        remove_unused_columns=False,
        label_smoothing_factor=0.1,
        dataloader_pin_memory=True,
        dataloader_num_workers=2,
        fp16=True,
        warmup_steps=100,
        weight_decay=0.01,
        max_grad_norm=1.0,
        optim="adamw_hf",
        group_by_length=True,
    )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation']
    )

    trainer.train()
    model = model.merge_and_unload()
    model.save_pretrained('./fine_tuned_model')
    # tokenizer.save_pretrained('./fine_tuned_model')