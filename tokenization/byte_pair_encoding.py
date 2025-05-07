import os
from main import load_text

# Suppress TensorFlow Info messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = (
    "3"  # 0: all messages, 1: filter out INFO messages, 2: filter out WARNING messages, 3: filter out ERROR messages
)
from tensorflow.keras.layers import StringLookup
from tensorflow.strings import unicode_split
from tensorflow.data import Dataset


def split_input_target(sequence):
    input_text = sequence[:-1]
    target_text = sequence[1:]
    return input_text, target_text


def tokenize_text(text):
    """
    Tokenizes the input text into words.
    """

    ids_from_chars = StringLookup(vocabulary=list(sorted(set(text))), mask_token=None)
    chars_from_ids = StringLookup(
        vocabulary=ids_from_chars.get_vocabulary(), invert=True, mask_token=None
    )

    vocab_size = len(ids_from_chars.get_vocabulary())
    print("Vocabulary:", ids_from_chars.get_vocabulary())
    print("Vocabulary Size:", vocab_size)

    return (
        vocab_size,
        ids_from_chars,
        chars_from_ids,
    )


def main():
    text, unique_words, _ = load_text("shubhammaindola/harry-potter-books")
    unique_words = list(unique_words)
    print("Unique words:", unique_words[:1000])
    (
        vocab_size,
        ids_from_chars,
        chars_from_ids,
    ) = tokenize_text(text)

    # word_embeddings = model.get_layer("word_embedding").get_weights()[0]
    # # Save word embeddings to file
    # np.save("word_embeddings_bpe.npy", word_embeddings)


if __name__ == "__main__":
    main()
