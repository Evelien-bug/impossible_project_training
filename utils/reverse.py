from transformers import GPT2Tokenizer
import random
from typing import List
from tqdm import tqdm
import argparse
import os
from utils import save_dataset, load_sentences_from_file

tokenizer = None


def _init_tokenizer():
    global tokenizer
    if tokenizer is None:
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token


_init_tokenizer()

REVERSE_MARKER = '🅁'


def noreverse(text: str) -> str:
    text = text.strip()
    token_ids = tokenizer.encode(text)
    tokens = [tokenizer.decode([tid]) for tid in token_ids]

    if len(tokens) == 0:
        return text

    # Insert 🅁 at a random position
    insert_pos = random.randint(1, len(tokens))
    tokens.insert(insert_pos, ' ' + REVERSE_MARKER)

    return ''.join(tokens)


def partial_reverse(text: str) -> str:
    random.seed(42)
    text = text.strip()
    token_ids = tokenizer.encode(text)
    tokens = [tokenizer.decode([tid]) for tid in token_ids]

    if len(tokens) < 3:
        return None

    # Insert 🅁 at a random position
    insert_pos = random.randint(0, len(tokens) - 2)

    # Split tokens
    before = tokens[:insert_pos]
    after = tokens[insert_pos:]

    # Reverse the tokens after 🅁
    after_reversed = after[::-1]

    result = before + [REVERSE_MARKER] + after_reversed

    return ' '.join(result).replace('  ', ' ')


def full_reverse(text: str) -> str:
    text = text.strip()
    token_ids = tokenizer.encode(text)
    tokens = [tokenizer.decode([tid]) for tid in token_ids]

    if len(tokens) == 0:
        return text

    # Insert 🅁 at a random position
    insert_pos = random.randint(0, len(tokens))
    tokens.insert(insert_pos, ' ' + REVERSE_MARKER)

    # Reverse all tokens
    tokens_reversed = tokens[::-1]

    return ' '.join(tokens_reversed).replace('  ', ' ')


def partial_reverse_batch(texts: List[str]) -> List[tuple]:
    training_data = []
    for sentence in tqdm(texts):
        training_data.append((partial_reverse(sentence), sentence))
    return list(zip(training_data, texts))

def pre_process(input_file, training_data_path):
    training_data = []
    sentences = load_sentences_from_file(input_file)
    for sentence in tqdm(sentences):
        training_data.append((partial_reverse(sentence), sentence))
    save_dataset(training_data, training_data_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, required=True, )

    args = parser.parse_args()

    pre_process(args.input, f"training_data_{os.path.basename(args.input).split('.')[0]}_partialReverse.json")
