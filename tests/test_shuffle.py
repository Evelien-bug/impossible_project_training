import pytest
import sys
from pathlib import Path

# Add the utils directory to the path to import the modules
utils_path = Path(__file__).parent.parent / "utils"
sys.path.append(str(utils_path))

from shuffle import noshuffle, local_shuffle, full_shuffle

# Test sentence
text = "The quick brown fox jumps over the lazy dog."

# --- Tests for noshuffle ---
def test_noshuffle():
    # noshuffle should simply strip whitespace
    text_with_space = "  some text.  "
    assert noshuffle(text_with_space) == "some text."
    assert noshuffle(text) == text

# --- Tests for local_shuffle ---
# We use a seed to make the shuffling deterministic
def test_local_shuffle_window3():
    # The shuffling is based on token IDs, so we need to know what the shuffled text looks like.
    # We can get this by running the function once and capturing the output.
    expected = "brown quick The over jumps fox dog lazy the ."
    assert local_shuffle(text, window_size=3, seed=42) == expected

def test_local_shuffle_window5():
    expected = "jumps brown fox quick The. lazy dog the over"
    assert local_shuffle(text, window_size=5, seed=42) == expected

def test_local_shuffle_small_window():
    # If the number of tokens is less than the window size, it should not shuffle
    short_text = "a b c"
    assert local_shuffle(short_text, window_size=5, seed=42).strip() == "a b c"

# --- Tests for full_shuffle ---
def test_full_shuffle():
    # We use a seed to make the shuffling deterministic
    expected = "over theThe lazy fox brown jumps. quick dog"
    assert full_shuffle(text, seed=42).strip() == expected

def test_full_shuffle_empty_string():
    assert full_shuffle("", seed=42) == ""
