import os
import sys

from utils import save_dataset, load_sentences_from_file

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import argparse
from transformers import GPT2Tokenizer, AddedToken
import spacy
from typing import List
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
            result.append(token.replace(token_clean, ' '+verb_found['lemma']))
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
            result.append(token.replace(token_clean, ' '+verb_found['lemma']))
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
            result.append(' ' + token.lemma_)
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

def wordhop_batch(texts: List[str]) -> List[tuple]:
    training_data = []
    for sentence in tqdm(texts):
        training_data.append((wordhop(sentence), sentence))
    return list(zip(training_data, texts))

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

def generate_training_data(input_file):
    print("Generating training data...")
    sentences = load_sentences_from_file(input_file)
    training_data = wordhop_batch(sentences, 512)
    return training_data


def pre_process(input_file, training_data_path):
    training_data = []
    sentences = load_sentences_from_file(input_file)
    for sentence in tqdm(sentences):
        training_data.append((wordhop(sentence), sentence))
    save_dataset(training_data, training_data_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, required=True, )

    args = parser.parse_args()

    pre_process(args.input, f"training_data_{os.path.basename(args.input).split('.')[0]}_wordHop.json")
