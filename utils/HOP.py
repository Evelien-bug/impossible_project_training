import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import argparse
from transformers import GPT2Tokenizer, AddedToken
import spacy
from typing import List
from multiprocessing import Pool, cpu_count
from pathlib import Path
import json
from tqdm import tqdm


def tokenizer_with_markers():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    marker_list = [SINGULAR_MARKER, PLURAL_MARKER]
    if len(marker_list) == 0:
        return tokenizer

    new_tokens = []
    for marker in marker_list:
        new_tokens.append(AddedToken(marker, lstrip=True, rstrip=False))
    tokenizer.add_tokens(new_tokens)
    return tokenizer
nlp = spacy.load("en_core_web_trf")
SINGULAR_MARKER = '🅂'
PLURAL_MARKER = '🄿'

tokenizer = tokenizer_with_markers()


def _chunk_list(lst: List, n_chunks: int) -> List[List]:
    chunk_size = len(lst) // n_chunks + (1 if len(lst) % n_chunks else 0)
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def nohop(text: str) -> str:
    doc = nlp(text)

    # Build mapping of character positions to verb info
    verb_info = {}
    for token in doc:
        if is_3rd_person_present_verb(token):
            verb_info[token.idx] = {
                'end': token.idx + len(token.text),
                'lemma': token.lemma_,
                'marker': SINGULAR_MARKER if is_singular_verb(token) else PLURAL_MARKER
            }

    # Tokenize with GPT-2
    token_ids = tokenizer.encode(text)
    tokens = [tokenizer.decode([tid]) for tid in token_ids]

    # Track position in original text
    result = []
    char_pos = 0

    for token in tokens:
        # Find position of this token in original text
        token_clean = token.lstrip()
        token_start = text.find(token_clean, char_pos)

        if token_start == -1:
            result.append(token)
            continue

        token_end = token_start + len(token_clean)

        # Check if this token overlaps with a verb
        verb_found = None
        for v_start, v_data in verb_info.items():
            if v_start >= token_start and v_start < token_end:
                verb_found = v_data
                break

        if verb_found:
            # Replace with lemma and add marker
            result.append(token.replace(token_clean, verb_found['lemma']))
            result.append(' ' + verb_found['marker'])
        else:
            result.append(token)

        char_pos = token_end

    return ''.join(result)


def tokenhop(text: str) -> str:
    doc = nlp(text)

    # Build mapping of character positions to verb info
    verb_positions = {}
    for token in doc:
        if is_3rd_person_present_verb(token):
            verb_positions[token.idx] = {
                'end': token.idx + len(token.text),
                'lemma': token.lemma_,
                'marker': SINGULAR_MARKER if is_singular_verb(token) else PLURAL_MARKER
            }

    # Tokenize with GPT-2
    token_ids = tokenizer.encode(text)
    tokens = [tokenizer.decode([tid]) for tid in token_ids]

    # Track position and find verbs
    result = []
    pending_markers = {}  # {token_index: marker}
    char_pos = 0

    for i, token in enumerate(tokens):
        # Insert any pending markers
        if i in pending_markers:
            result.append(' ' + pending_markers[i])

        token_clean = token.lstrip()
        token_start = text.find(token_clean, char_pos)

        if token_start == -1:
            result.append(token)
            continue

        token_end = token_start + len(token_clean)

        # Check if this token contains a verb
        verb_found = None
        for v_start, v_data in verb_positions.items():
            if v_start >= token_start and v_start < token_end:
                verb_found = v_data
                break

        if verb_found:
            result.append(token.replace(token_clean, verb_found['lemma']))
            # Schedule marker 4 tokens later
            insert_pos = i + 4
            pending_markers[insert_pos] = verb_found['marker']
        else:
            result.append(token)

        char_pos = token_end

    # Add remaining markers at the end
    for idx in sorted(pending_markers.keys()):
        if idx >= len(tokens):
            result.append(' ' + pending_markers[idx])

    return ''.join(result)


def wordhop(text: str) -> str:
    doc = nlp(text)
    tokens = list(doc)
    result = []
    pending_markers = []  # List of (target_word_count, marker) tuples
    word_count = 0

    for i, token in enumerate(tokens):
        # Check if we should insert any pending markers at this word count
        markers_to_insert = [m for wc, m in pending_markers if wc == word_count]
        for marker in markers_to_insert:
            result.append(marker)
            result.append(' ')
        # Remove inserted markers
        pending_markers = [(wc, m) for wc, m in pending_markers if wc != word_count]

        if is_3rd_person_present_verb(token):
            # Use spaCy's lemma
            result.append(token.lemma_)
            # Schedule marker to be inserted 4 words after this verb
            marker = SINGULAR_MARKER if is_singular_verb(token) else PLURAL_MARKER
            target_wc = word_count + 4 + (1 if not token.is_punct else 0)
            pending_markers.append((target_wc, ' ' + marker))
        else:
            result.append(token.text)

        # Increment word count (skip punctuation)
        if not token.is_punct:
            word_count += 1

        # Preserve spacing
        if token.whitespace_:
            result.append(' ')

    # Add any remaining markers at the end
    for _, marker in pending_markers:
        result.append(' ')
        result.append(marker)

    return ''.join(result).strip().replace('  ', ' ')


def _wordhop_batch(texts: List[str]) -> List[str]:

    docs = list(nlp.pipe(texts))
    results = []

    for text, doc in zip(texts, docs):
        tokens = list(doc)
        result = []
        pending_markers = []
        word_count = 0

        for i, token in enumerate(tokens):
            markers_to_insert = [m for wc, m in pending_markers if wc == word_count]
            for marker in markers_to_insert:
                result.append(marker)
                result.append(' ')
            pending_markers = [(wc, m) for wc, m in pending_markers if wc != word_count]

            if is_3rd_person_present_verb(token):
                result.append(token.lemma_)
                marker = SINGULAR_MARKER if is_singular_verb(token) else PLURAL_MARKER
                target_wc = word_count + 4 + (1 if not token.is_punct else 0)
                pending_markers.append((target_wc, ' ' + marker))
            else:
                result.append(token.text)

            if not token.is_punct:
                word_count += 1

            if token.whitespace_:
                result.append(' ')

        for _, marker in pending_markers:
            result.append(' ')
            result.append(marker)

        results.append(''.join(result).strip().replace('  ', ' '))

    return results

def _process_chunk_wordhop_batched(args):
    chunk, batch_size = args
    results = []
    for i in range(0, len(chunk), batch_size):
        batch = chunk[i:i + batch_size]
        results.extend(_wordhop_batch(batch))
    return results

def wordhop_fast(texts: List[str], batch_size: int = 32, n_workers: int = None) -> List[str]:

    if n_workers is None:
        n_workers = max(1, cpu_count() - 1)
    print(f"Using {n_workers} workers")
    if len(texts) < 100:
        return _wordhop_batch(texts)

    chunks = _chunk_list(texts, n_workers)
    chunk_args = [(chunk, batch_size) for chunk in chunks]

    with Pool(n_workers) as pool:
        results_nested = pool.map(_process_chunk_wordhop_batched, chunk_args)

    results = []
    for chunk_results in results_nested:
        results.extend(chunk_results)

    return results

def wordhop_batch(texts: List[str], batch_size: int = 32, n_workers: int = None) -> List[tuple]:
    originals = [t.strip() for t in texts]
    corrupted = wordhop_fast(texts, batch_size, n_workers)
    return list(zip(corrupted, originals))

def is_3rd_person_present_verb(token) -> bool:
    # Check for present tense verbs
    if token.tag_ in ['VBZ', 'VBP']:
        return True
    if token.pos_ == 'VERB':
        morph = token.morph.to_dict()
        if morph.get('Tense') == 'Pres' and morph.get('VerbForm') == 'Fin':
            return True
    return False


def is_singular_verb(token) -> bool:
    if token.tag_ == 'VBZ':
        return True
    morph = token.morph.to_dict()
    if morph.get('Number') == 'Sing' and morph.get('Person') == '3':
        return True
    return False

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


def generate_training_data(input_file):
    print("Generating training data...")
    sentences = load_sentences_from_file(input_file)
    training_data = wordhop_batch(sentences, 512)
    return training_data

def save_dataset(data, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(data)} examples to {output_file}")


def pre_process(input_file, training_data_path):
    training_data = []
    sentences = load_sentences_from_file(input_file)
    for sentence in tqdm(sentences):
        training_data.append((wordhop(sentence), sentence))
    save_dataset(training_data, training_data_path)

if __name__ == "__main__":
    # Example sentences
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, required=True,)

    args = parser.parse_args()

    pre_process(args.input, 'wordhop_1k_test.json')
    # text1 = "He cleans his very messy bookshelf."
    # text2 = "They clean their very messy bookshelf."
    # text3 = "She walks to the store and buys some milk."
    #
    # print("=" * 60)
    # print("Example 1:")
    # print("Original: ", text1)
    # print("NOHOP:    ", nohop(text1))
    # print("TOKENHOP: ", tokenhop(text1))
    # print("WORDHOP:  ", wordhop(text1))
    #
    # print("\n" + "=" * 60)
    # print("Example 2:")
    # print("Original: ", text2)
    # print("NOHOP:    ", nohop(text2))
    # print("TOKENHOP: ", tokenhop(text2))
    # print("WORDHOP:  ", wordhop(text2))
    #
    # print("\n" + "=" * 60)
    # print("Example 3:")
    # print("Original: ", text3)
    # print("NOHOP:    ", nohop(text3))
    # print("TOKENHOP: ", tokenhop(text3))
    # print("WORDHOP:  ", wordhop(text3))
    #
    # # Multi-sentence example
    # print("\n" + "=" * 60)
    # print("Multi-sentence example:")
    # text4 = "He walks quickly. She runs faster. They play together."
    # print("Original: ", text4)
    # print("NOHOP:    ", nohop(text4))
    # print("TOKENHOP: ", tokenhop(text4))
    # print("WORDHOP:  ", wordhop(text4))