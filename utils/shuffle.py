import argparse
import os
from pathlib import Path
from typing import List
from numpy.random import default_rng
from tqdm import tqdm
from transformers import GPT2Tokenizer
from utils import save_dataset, load_sentences_from_file

tokenizer = None


def _init_tokenizer():
    global tokenizer
    if tokenizer is None:
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token


_init_tokenizer()


def noshuffle(text: str) -> str:
    text = text.strip()
    return text


def local_shuffle(text: str, window_size: int = 3, seed: int = None) -> str:
    text = ' ' + text + ' '

    # Get token IDs (like in the paper)
    tokens = tokenizer.encode(text)

    if len(tokens) < window_size:
        return text

    # Shuffle tokens in batches of size window
    shuffled_tokens = []
    for i in range(0, len(tokens), window_size):
        batch = tokens[i:i + window_size].copy()

        # Use numpy RNG (same as paper)
        if seed is not None:
            default_rng(seed).shuffle(batch)
        else:
            default_rng().shuffle(batch)

        shuffled_tokens += batch

    # Decode back to text
    return tokenizer.decode(shuffled_tokens).strip()


def full_shuffle(text: str, seed: int = None) -> str:
    text = text.strip()

    # Get token IDs
    tokens = tokenizer.encode(text)

    if len(tokens) == 0:
        return text

    # Shuffle all tokens
    shuffled_tokens = tokens.copy()
    if seed is not None:
        default_rng(seed).shuffle(shuffled_tokens)
    else:
        default_rng().shuffle(shuffled_tokens)

    # Decode back to text
    return tokenizer.decode(shuffled_tokens)

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


def local_shuffle_batch(texts: List[str]) -> List[tuple]:
    training_data = []
    for sentence in tqdm(texts):
        training_data.append((local_shuffle(sentence), sentence))
    return list(zip(training_data, texts))

def pre_process(input_file, training_data_path):
    training_data = []
    sentences = load_sentences_from_file(input_file)
    for sentence in tqdm(sentences):
        training_data.append((local_shuffle(sentence), sentence))
    save_dataset(training_data, training_data_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, required=True, )

    args = parser.parse_args()

    pre_process(args.input, f"training_data_{os.path.basename(args.input).split('.')[0]}_localShuffle.json")
