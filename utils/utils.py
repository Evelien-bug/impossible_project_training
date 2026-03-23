from transformers import AutoTokenizer, AddedToken
import json
from pathlib import Path
import torch
import yaml

MARKER_HOP_SING = "🅂"
MARKER_HOP_PLUR = "🄿"
MARKER_REV = "🅁"


def get_gpt2_tokenizer_with_markers(marker_list):
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    if len(marker_list) == 0:
        return tokenizer

    new_tokens = []
    for marker in marker_list:
        new_tokens.append(AddedToken(marker, lstrip=True, rstrip=False))
    tokenizer.add_tokens(new_tokens)
    return tokenizer


gpt2_original_tokenizer = get_gpt2_tokenizer_with_markers([])

gpt2_hop_tokenizer = get_gpt2_tokenizer_with_markers(
    [MARKER_HOP_SING, MARKER_HOP_PLUR])
marker_sg_token = gpt2_hop_tokenizer.get_added_vocab()[
    MARKER_HOP_SING]
marker_pl_token = gpt2_hop_tokenizer.get_added_vocab()[
    MARKER_HOP_PLUR]

gpt2_rev_tokenizer = get_gpt2_tokenizer_with_markers(
    [MARKER_REV])
marker_rev_token = gpt2_rev_tokenizer.get_added_vocab()[
    MARKER_REV]

MARKER_TOKEN_IDS = [marker_sg_token, marker_pl_token, marker_rev_token]


def get_device():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    return device


def load_configs(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def save_dataset(data, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(data)} examples to {output_file}")

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
