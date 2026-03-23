import pytest

from utils.hop import nohop, tokenhop, wordhop, SINGULAR_MARKER, PLURAL_MARKER

# Test sentences
text_singular = "He cleans his very messy bookshelf."
text_plural = "They clean their very messy bookshelf."
text_multiple_verbs = "She walks to the store and buys some milk."

# --- Tests for nohop ---

def test_nohop_singular():
    expected = f"He clean {SINGULAR_MARKER} his very messy bookshelf."
    # The tokenizer might add spaces differently, so we normalize by splitting and rejoining
    normalized_expected = " ".join(expected.split())
    normalized_actual = " ".join(nohop(text_singular).split())
    assert normalized_actual == normalized_expected

def test_nohop_plural():
    expected = f"They clean {PLURAL_MARKER} their very messy bookshelf."
    normalized_expected = " ".join(expected.split())
    normalized_actual = " ".join(nohop(text_plural).split())
    assert normalized_actual == normalized_expected

def test_nohop_multiple_verbs():
    expected = f"She walk {SINGULAR_MARKER} to the store and buy {SINGULAR_MARKER} some milk."
    normalized_expected = " ".join(expected.split())
    normalized_actual = " ".join(nohop(text_multiple_verbs).split())
    assert normalized_actual == normalized_expected

# --- Tests for tokenhop ---
# Tokenhop is sensitive to the specific tokenization, so these tests are based on GPT-2's behavior.
def test_tokenhop_singular():
    # "cleans" is one token. Marker is inserted 4 tokens later.
    # GPT-2 tokens: 'He', ' cleans', ' his', ' very', ' messy', ' bookshelf', '.'
    # Expected: 'He', ' clean', ' his', ' very', ' messy', ' 🅂', ' bookshelf', '.'
    expected = f"He clean his very messy {SINGULAR_MARKER} bookshelf."
    normalized_expected = " ".join(expected.split())
    normalized_actual = " ".join(tokenhop(text_singular).split())
    assert normalized_actual == normalized_expected

def test_tokenhop_plural():
    # "clean" is one token.
    # GPT-2 tokens: 'They', ' clean', ' their', ' very', ' messy', ' bookshelf', '.'
    # Expected: 'They', ' clean', ' their', ' very', ' messy', ' 🄿', ' bookshelf', '.'
    expected = f"They clean their very messy {PLURAL_MARKER} bookshelf."
    normalized_expected = " ".join(expected.split())
    normalized_actual = " ".join(tokenhop(text_plural).split())
    assert normalized_actual == normalized_expected

# --- Tests for wordhop ---
def test_wordhop_singular():
    # Marker is inserted 4 words later.
    # words: "He", "cleans", "his", "very", "messy", "bookshelf"
    # The marker for "cleans" (word 2) should appear before "bookshelf" (word 6)
    expected = f"He clean his very messy bookshelf {SINGULAR_MARKER} ."
    normalized_expected = " ".join(expected.split())
    normalized_actual = " ".join(wordhop(text_singular).split())
    assert normalized_actual == normalized_expected

def test_wordhop_plural():
    # Marker for "clean" (word 2) should appear before "bookshelf" (word 6)
    expected = f"They clean their very messy bookshelf {PLURAL_MARKER} ."
    normalized_expected = " ".join(expected.split())
    normalized_actual = " ".join(wordhop(text_plural).split())
    assert normalized_actual == normalized_expected

def test_wordhop_multiple_verbs():
    # "walks" (word 2) -> marker before word 6 ("and")
    # "buys" (word 7) -> marker before word 11 ("milk")
    expected = f"She walk to the store and {SINGULAR_MARKER} buy some milk. {SINGULAR_MARKER}"
    normalized_expected = " ".join(expected.split())
    normalized_actual = " ".join(wordhop(text_multiple_verbs).split())
    assert normalized_actual == normalized_expected
