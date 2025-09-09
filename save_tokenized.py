import argparse
import json
import os

from datasets import Dataset, DatasetDict
import utils

def preprocess(tokenizer, max_length=128):
    def tokenize(examples):
        input_ids_list = []
        label_ids_list = []

        for src, tgt in zip(examples["perturbed_text"], examples["original_text"]):
            # Build "source + EOS + target"
            combined = src + tokenizer.eos_token + tgt
            tokenized = tokenizer(
                combined,
                padding="max_length",
                truncation=True,
                max_length=max_length
            )

            input_ids = tokenized["input_ids"]

            labels = input_ids.copy()

            src_ids = tokenizer(src + tokenizer.eos_token, max_length=max_length, truncation=True)["input_ids"]
            labels[:len(src_ids)] = [-100] * len(src_ids)

            input_ids_list.append(input_ids)
            label_ids_list.append(labels)

        return {
            "input_ids": input_ids_list,
            "attention_mask": [tokenizer.get_attention_mask(ids, max_length=max_length) for ids in input_ids_list]
                if hasattr(tokenizer, "get_attention_mask") else None,
            "labels": label_ids_list
        }

    return tokenize



def load_dataset(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    train_dataset = Dataset.from_list(data["train"])
    valid_dataset = Dataset.from_list(data["validate"])

    dataset = DatasetDict({
        "train": train_dataset,
        "validation": valid_dataset
    })

    return dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-p', '--path', type=str, required=True,
                        help="Path to file")
    parser.add_argument('-t', '--type', type=str, required=True,
                        help="Type of perturbation")
    args = parser.parse_args()

    data = load_dataset(args.path)
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

    tokenize = preprocess(tokenizer)

    dataset = data.map(
        tokenize,
        batched=True,
        batch_size=2000,
        remove_columns=['perturbed_text', 'original_text'],
        num_proc=8,
        load_from_cache_file=True,  # Use cache if available
        desc="Tokenizing dataset"
    )

    dataset.save_to_disk(f"{os.path.dirname(args.path)}/tokenized_dataset")