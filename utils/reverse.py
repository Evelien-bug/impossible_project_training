from transformers import GPT2Tokenizer
import random

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

REVERSE_MARKER = '🅁'

def noreverse(text: str) -> str:
    token_ids = tokenizer.encode(text)
    tokens = [tokenizer.decode([tid]) for tid in token_ids]

    if len(tokens) == 0:
        return text

    # Insert 🅁 at a random position
    insert_pos = random.randint(1, len(tokens))
    tokens.insert(insert_pos, REVERSE_MARKER)

    return ''.join(tokens)


def partial_reverse(text: str) -> str:
    token_ids = tokenizer.encode(text)
    tokens = [tokenizer.decode([tid]) for tid in token_ids]

    if len(tokens) == 0:
        return text

    # Insert 🅁 at a random position
    insert_pos = random.randint(1, len(tokens))

    # Split tokens
    before = tokens[:insert_pos]
    after = tokens[insert_pos:]

    # Reverse the tokens after 🅁
    after_reversed = after[::-1]

    result = before + [REVERSE_MARKER] + after_reversed

    return ''.join(result)


def full_reverse(text: str) -> str:
    token_ids = tokenizer.encode(text)
    tokens = [tokenizer.decode([tid]) for tid in token_ids]

    if len(tokens) == 0:
        return text

    # Insert 🅁 at a random position
    insert_pos = random.randint(0, len(tokens))
    tokens.insert(insert_pos, REVERSE_MARKER)

    # Reverse all tokens
    tokens_reversed = tokens[::-1]

    return ''.join(tokens_reversed)


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