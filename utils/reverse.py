from transformers import GPT2Tokenizer
import random
from multiprocessing import Pool, cpu_count
from typing import List
from tqdm import tqdm

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
    tokens.insert(insert_pos, ' '+REVERSE_MARKER)

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
    tokens.insert(insert_pos, ' '+ REVERSE_MARKER)

    # Reverse all tokens
    tokens_reversed = tokens[::-1]

    return ' '.join(tokens_reversed).replace('  ', ' ')

def _process_chunk_partial_reverse_batched(args):
    chunk, batch_size = args
    results = []
    for i in range(0, len(chunk), batch_size):
        batch = chunk[i:i + batch_size]
        results.extend(_partial_reverse_batch(batch))
    return results

def _chunk_list(lst: List, n_chunks: int) -> List[List]:
    chunk_size = len(lst) // n_chunks + (1 if len(lst) % n_chunks else 0)
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

def _partial_reverse_batch(texts: List[str]) -> List[str]:
    random.seed(42)
    texts = [t.strip() for t in texts]

    encoded = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    input_ids = encoded['input_ids']
    attention_mask = encoded['attention_mask']

    results = []
    for i in range(len(texts)):
        mask = attention_mask[i].bool()
        token_ids = input_ids[i][mask].tolist()
        tokens = [tokenizer.decode([tid]) for tid in token_ids]

        if len(tokens) < 3:
            results.append(None)
            continue

        insert_pos = random.randint(0, len(tokens) - 2)
        before = tokens[:insert_pos]
        after = tokens[insert_pos:]
        after_reversed = after[::-1]

        result = before + [REVERSE_MARKER] + after_reversed
        results.append(' '.join(result).replace('  ', ' '))

    return results

def partial_reverse_fast(texts: List[str], batch_size: int = 128, n_workers: int = None) -> List[str]:
    if n_workers is None:
        n_workers = max(1, cpu_count() - 1)

    if len(texts) < 500:
        return _partial_reverse_batch(texts)

    chunks = _chunk_list(texts, n_workers)
    chunk_args = [(chunk, batch_size) for chunk in chunks]

    with Pool(n_workers) as pool:
        results_nested = pool.map(_process_chunk_partial_reverse_batched, chunk_args)

    results = []
    for chunk_results in results_nested:
        results.extend(chunk_results)

    return results

def partial_reverse_batch(texts: List[str], batch_size: int = 128, n_workers: int = None) -> List[tuple]:

    results = []
    for text in tqdm(texts):
        results.append((partial_reverse(text), text))
    return results
    # originals = [t.strip() for t in texts]
    # corrupted = partial_reverse_fast(texts, batch_size, n_workers)
    # return list(zip(corrupted, originals))


# Example usage
if __name__ == "__main__":
    # Example sentences
    text1 = "He cleans his very messy bookshelf."
    text2 = "They clean his very messy books."
    text3 = "She walks to the store and buys some milk."

    print("=" * 60)
    print("Example 1:")
    print("Original:       ", text1)
    print("NOREVERSE:      ", noreverse(text1))
    print("PARTIALREVERSE: ", partial_reverse(text1))
    print("FULLREVERSE:    ", full_reverse(text1))

    print("\n" + "=" * 60)
    print("Example 2:")
    print("Original:       ", text2)
    print("NOREVERSE:      ", noreverse(text2))
    print("PARTIALREVERSE: ", partial_reverse(text2))
    print("FULLREVERSE:    ", full_reverse(text2))

    print("\n" + "=" * 60)
    print("Example 3:")
    print("Original:       ", text3)
    print("NOREVERSE:      ", noreverse(text3))
    print("PARTIALREVERSE: ", partial_reverse(text3))
    print("FULLREVERSE:    ", full_reverse(text3))

    # Multi-sentence example
    print("\n" + "=" * 60)
    print("Multi-sentence example:")
    text4 = "He walks quickly. She runs faster."
    print("Original:       ", text4)
    print("NOREVERSE:      ", noreverse(text4))
    print("PARTIALREVERSE: ", partial_reverse(text4))
    print("FULLREVERSE:    ", full_reverse(text4))