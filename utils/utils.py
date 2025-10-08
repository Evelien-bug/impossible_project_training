
from collections import deque
from string import punctuation
from transformers import AutoTokenizer, AddedToken
from numpy.random import default_rng



MARKER_HOP_SING = "🅂"
MARKER_HOP_PLUR = "🄿"
MARKER_REV = "🅁"
BOS_TOKEN = "<BOS_TOKEN>"
PART_TOKENS = set(["n't", "'ll", "'s", "'re", "'ve", "'m"])
PUNCT_TOKENS = set(punctuation)



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

gpt2_det_tokenizer = get_gpt2_tokenizer_with_markers(
    [BOS_TOKEN])
bos_token_id = gpt2_det_tokenizer.get_added_vocab()[BOS_TOKEN]


MARKER_TOKEN_IDS = [marker_sg_token, marker_pl_token, marker_rev_token]



def merge_part_tokens(words):
    result = []
    for s in words:
        if result and s in PART_TOKENS and len(result) > 0:
            result[-1] += s
        else:
            result.append(s)
    return result


def __affect_hop_word(word):
    return ("AUX" in word['upos']) and ('VBZ' in word['xpos'] or 'VBP' in word['xpos'])


def __perturb_hop_words(sent, num_hops, marker_sg, marker_pl):
    return __perturb_hop_words_complete_hops(
        sent, num_hops, marker_sg, marker_pl)

def check_word_hops_completed(sent, num_hops=4, marker=MARKER_HOP_SING):
    _, hops_completed = __perturb_hop_words_complete_hops(
        sent, num_hops, marker, marker)
    return hops_completed


def __perturb_hop_words_complete_hops(sent, num_hops, marker_sg, marker_pl):

    word_annotations = sent["word_annotations"].copy()
    word_annotations.reverse()

    hop_completed = []
    new_sent = []
    for word in word_annotations:

        # Identify 3.pres verbs
        if __affect_hop_word(word):

            new_sent.append(
                word["lemma"] if word["lemma"] is not None else word["text"])

            insert_index = len(new_sent)-1
            skipped_words = 0
            while skipped_words < num_hops and insert_index > 0:

                if (not any([c.isalnum() for c in
                             "".join(new_sent[:insert_index])])):
                    break

                if (new_sent[insert_index] not in PART_TOKENS) and \
                        (not set(new_sent[insert_index]).issubset(PUNCT_TOKENS)):
                    skipped_words += 1
                insert_index -= 1

            if any([c.isalnum() for c in
                    "".join(new_sent[:insert_index])]):
                while insert_index != 0 and (new_sent[insert_index] in PART_TOKENS
                                             or set(new_sent[insert_index]).issubset(PUNCT_TOKENS)):
                    insert_index -= 1

            if insert_index != 0 and new_sent[insert_index-1] in PART_TOKENS:
                insert_index -= 1

            hop_completed.append(skipped_words == num_hops)

            if "VBZ" in word['xpos']:
                new_sent.insert(insert_index, marker_sg)
            elif "VBP" in word['xpos']:
                new_sent.insert(insert_index, marker_pl)
            else:
                raise Exception(
                    "Number not in verb features\n" + sent["sent_text"])

        else:
            new_sent.append(word["text"])

    new_sent.reverse()
    sent_string = " ".join(merge_part_tokens(new_sent))
    return sent_string, sent['sent_text']


def __perturb_hop_tokens(sent, num_hops):

    word_annotations = sent["word_annotations"].copy()
    word_annotations.reverse()

    new_sent = deque()
    tokens = []
    for word in word_annotations:

        if __affect_hop_word(word):

            lemma = word["lemma"] if word["lemma"] is not None else word["text"]

            if len(new_sent) > 0 and new_sent[0] in PART_TOKENS:
                lemma = lemma + new_sent[0]
                new_sent.popleft()

            if len(new_sent) > 0:
                sent_string = " ".join(merge_part_tokens(new_sent))
                tokens = gpt2_hop_tokenizer.encode(
                    " " + sent_string) + tokens

            if "VBZ" in word['xpos']:
                tokens.insert(num_hops, marker_sg_token)
            elif "VBP" in word['xpos']:
                tokens.insert(num_hops, marker_pl_token)
            else:
                raise Exception(
                    "Number not in verb features\n" + sent["sent_text"])

            new_sent = deque()
            new_sent.append(lemma)

        else:
            new_sent.appendleft(word["text"])

    if len(new_sent) > 0:
        sent_string = " ".join(merge_part_tokens(new_sent))
        return sent_string, sent["word_annotations"]


def __perturb_reverse(sent, rng, reverse, full):

    tokens = gpt2_rev_tokenizer.encode(sent["sent_text"])

    i = rng.choice(len(tokens)+1)
    tokens.insert(i, marker_rev_token)

    tokens_before = tokens[:i+1]
    tokens_after = tokens[i+1:]
    if reverse:
        tokens_after.reverse()
    new_tokens = tokens_before + tokens_after
    if full:
        assert not reverse
        new_tokens.reverse()

    return gpt2_rev_tokenizer.decode(new_tokens), sent["sent_text"]

def __perturb_shuffle_deterministic(sent, seed, shuffle):
    tokens = gpt2_original_tokenizer.encode(sent["sent_text"])
    if shuffle:
        default_rng(seed).shuffle(tokens)
    return tokens


def __perturb_shuffle_nondeterministic(sent, rng):
    tokens = gpt2_original_tokenizer.encode(sent["sent_text"])
    rng.shuffle(tokens)
    return tokens


def __perturb_shuffle_local(sent, seed, window=5):
    tokens = gpt2_original_tokenizer.encode(sent["sent_text"])

    shuffled_tokens = []
    for i in range(0, len(tokens), window):
        batch = tokens[i:i+window].copy()
        default_rng(seed).shuffle(batch)
        shuffled_tokens += batch

    return gpt2_rev_tokenizer.decode(shuffled_tokens) , sent["sent_text"]


def __perturb_shuffle_even_odd(sent):
    tokens = gpt2_original_tokenizer.encode(sent["sent_text"])
    even = [tok for i, tok in enumerate(tokens) if i % 2 == 0]
    odd = [tok for i, tok in enumerate(tokens) if i % 2 != 0]
    return even + odd


def affect_hop(sent):
    return any([__affect_hop_word(word) for word in sent['word_annotations']]) \
        and sent["constituency_parse"] is not None


def affect_reverse(sent):
    return True


def affect_shuffle(sent):
    return True


def affect_none(sent):
    return False

def filter_hop(sent):
    assert (affect_hop(sent))
    return check_word_hops_completed(sent, 4)


def filter_reverse(sent):
    return True


def filter_shuffle(sent):
    tokens = gpt2_original_tokenizer.encode(sent["sent_text"])
    return len(tokens) > 1 and len(tokens) <= 350

def perturb_hop_words4(sent):
    return __perturb_hop_words(sent, 4, MARKER_HOP_SING, MARKER_HOP_PLUR)


def perturb_hop_tokens4(sent):
    return __perturb_hop_tokens(sent, 4)


def perturb_hop_control(sent):
    return __perturb_hop_tokens(sent, 0)


def perturb_reverse(sent, rng, reverse=True, full=False):
    return __perturb_reverse(sent, rng, reverse, full)


def perturb_shuffle_deterministic(sent, seed=None, shuffle=True):
    return __perturb_shuffle_deterministic(sent, seed, shuffle)


def perturb_shuffle_nondeterministic(sent, rng):
    return __perturb_shuffle_nondeterministic(sent, rng)


def perturb_shuffle_local(sent, seed, window):
    return __perturb_shuffle_local(sent, seed, window)


def perturb_shuffle_even_odd(sent):
    return __perturb_shuffle_even_odd(sent)
