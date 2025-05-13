# Inspired by https://huggingface.co/learn/llm-course/chapter6/5

import os
import pickle
from main import load_text

# Suppress TensorFlow Info messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = (
    "3"  # 0: all messages, 1: filter out INFO messages, 2: filter out WARNING messages, 3: filter out ERROR messages
)
from tensorflow.keras.layers import StringLookup


def get_vocab(text):
    """
    Tokenizes the input text into words.
    """

    ids_from_chars = StringLookup(vocabulary=list(sorted(set(text))), mask_token=None)
    vocabulary = ids_from_chars.get_vocabulary()

    vocab_size = len(vocabulary)
    print("Vocabulary:", vocabulary)
    print("Vocabulary Size:", vocab_size)

    return vocabulary


def get_word_frequencies(all_words):
    """
    Returns a dictionary of word frequencies in the text.
    """
    word_frequencies = {}
    for word in all_words:
        if word in word_frequencies:
            word_frequencies[word] += 1
        else:
            word_frequencies[word] = 1
    return word_frequencies


def compute_pair_freqs(splits, word_frequencies):
    pair_freqs = {}
    for word, freq in word_frequencies.items():
        split = splits[word]
        if len(split) == 1:
            continue
        for i in range(len(split) - 1):
            pair = (split[i], split[i + 1])
            if pair not in pair_freqs:
                pair_freqs[pair] = 0
            pair_freqs[pair] += freq
    return pair_freqs


def get_most_frequent_pairs(pair_freqs):
    best_pair = ""
    max_freq = None

    for pair, freq in pair_freqs.items():
        if max_freq is None or max_freq < freq:
            best_pair = pair
            max_freq = freq

    return best_pair, max_freq


def merge_pair(a, b, splits):
    for word, split in splits.items():
        if len(split) == 1:
            continue

        i = 0
        while i < len(split) - 1:
            if split[i] == a and split[i + 1] == b:
                split = split[:i] + [a + b] + split[i + 2 :]
            else:
                i += 1
        splits[word] = split
    return splits


def main():
    text, all_words, _, _ = load_text("shubhammaindola/harry-potter-books")
    vocab = get_vocab(text)

    word_frequencies = get_word_frequencies(all_words)
    most_frequent_words = sorted(
        word_frequencies.items(), key=lambda x: x[1], reverse=True
    )[:500]
    print("Most frequent words:", most_frequent_words)
    print("Number of unique words:", len(word_frequencies))
    print()

    splits = {word: [c for c in word] for word in word_frequencies.keys()}
    print("Initial splits:", list(splits.items())[:50])
    print()

    num_iterations = 1000
    for _ in range(num_iterations):
        pair_freqs = compute_pair_freqs(splits, word_frequencies)

        best_pair, max_freq = get_most_frequent_pairs(pair_freqs)
        vocab.append("".join(best_pair))

        splits = merge_pair(best_pair[0], best_pair[1], splits)

    print("Final vocabulary:", vocab)
    print("Final vocabulary size:", len(vocab))

    # Save splits and vocab to files
    with open("vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)

    # Save splits
    with open("splits.pkl", "wb") as f:
        pickle.dump(splits, f)


if __name__ == "__main__":
    main()
