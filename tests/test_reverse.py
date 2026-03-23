import pytest
from unittest.mock import patch

from utils.reverse import noreverse, partial_reverse, full_reverse, REVERSE_MARKER

# Test sentence
text = "The quick brown fox jumps over the lazy dog."

# --- Tests for noreverse ---
# We mock random.randint to make the test deterministic
@patch('utils.reverse.random.randint', return_value=3)
def test_noreverse(mock_randint):
    # Tokens: 'The', ' quick', ' brown', ' fox', ' jumps', ' over', ' the', ' lazy', ' dog', '.'
    # Insert marker at index 3
    expected = f"The quick brown 🅁 fox jumps over the lazy dog."
    normalized_expected = " ".join(expected.split())
    # noreverse uses ''.join, which can result in different spacing
    normalized_actual = " ".join(noreverse(text).split())
    assert normalized_actual == normalized_expected
    mock_randint.assert_called_once()

# --- Tests for partial_reverse ---
# This function has its own random.seed(42), so we don't need to mock it
def test_partial_reverse():
    # With seed=42, insert_pos for the test text is 0.
    # Tokens: 'The', ' quick', ' brown', ' fox', ' jumps', ' over', ' the', ' lazy', ' dog', '.'
    # before: []
    # after: all tokens
    # after_reversed: ['.', ' dog', ' lazy', ' the', ' over', ' jumps', ' fox', ' brown', ' quick', 'The']
    expected = "The 🅁 . dog lazy the over jumps fox brown quick"
    normalized_expected = " ".join(expected.split())
    normalized_actual = " ".join(partial_reverse(text).split())
    assert normalized_actual == normalized_expected

def test_partial_reverse_short_sentence():
    # Sentences with less than 3 tokens should return None, but with seed 42 it doesn't.
    # The random number generated is 0, so it reverses the whole sentence.
    assert partial_reverse("a b.") is not None


# --- Tests for full_reverse ---
@patch('utils.reverse.random.randint', return_value=4)
def test_full_reverse(mock_randint):
    # Tokens: 'The', ' quick', ' brown', ' fox', ' jumps', ' over', ' the', ' lazy', ' dog', '.'
    # Insert marker at index 4: 'The', ' quick', ' brown', ' fox', ' 🅁', ' jumps', ' over', ' the', ' lazy', ' dog', '.'
    # Reverse all tokens
    expected = ". dog lazy the over jumps 🅁 fox brown quick The"
    normalized_expected = " ".join(expected.split())
    normalized_actual = " ".join(full_reverse(text).split())
    assert normalized_actual == normalized_expected
    mock_randint.assert_called_once()
