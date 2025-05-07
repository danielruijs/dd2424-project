import os
from main import load_text

# Suppress TensorFlow Info messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = (
    "3"  # 0: all messages, 1: filter out INFO messages, 2: filter out WARNING messages, 3: filter out ERROR messages
)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import skipgrams
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, Input, Dot, Dense, Reshape
from nltk.tokenize import RegexpTokenizer
import numpy as np
import pickle


def tokenize_text(text):
    """
    Tokenizes the input text into words.
    """
    # nltk.download("punkt_tab")
    # tokens = nltk.word_tokenize(text)
    tokenizer = RegexpTokenizer(r"\b\w+'\w+|\w+|\S")
    tokens = tokenizer.tokenize(text)
    print("Number of tokens:", len(tokens))

    tokenizer = Tokenizer(lower=False, filters="")
    tokenizer.fit_on_texts(tokens)
    words2idx = tokenizer.word_index
    idx2words = tokenizer.index_word
    vocab_size = len(words2idx) + 1
    print("Vocabulary size:", vocab_size)
    print("idx2words:", list(idx2words.values())[:200])
    print("words2idx:", list(words2idx.items())[:200])
    sequences = tokenizer.texts_to_sequences([tokens])[0]

    with open("tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)

    skip_grams, label = skipgrams(
        sequences,
        vocabulary_size=vocab_size,
        window_size=5,
        negative_samples=1.0,
    )
    print("First 10 skip-grams:", skip_grams[:10])
    print("First 10 labels:", label[:10])
    print("Number of skip-grams:", len(skip_grams))

    return vocab_size, skip_grams, label


def word2vec_model(vocab_size, embedding_dim=100):
    target_input = Input(shape=(1,), name="target_input")
    context_input = Input(shape=(1,), name="context_input")

    embedding = Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        name="word_embedding",
    )

    target_embedding = embedding(target_input)
    context_embedding = embedding(context_input)

    dot_product = Dot(axes=-1)([target_embedding, context_embedding])
    dot_product = Reshape((1,))(dot_product)

    output = Dense(1, activation="sigmoid")(dot_product)

    word2vec_model = Model(inputs=[target_input, context_input], outputs=output)
    word2vec_model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
    )
    word2vec_model.summary()
    return word2vec_model


def main():
    text, _, _, _ = load_text("shubhammaindola/harry-potter-books")
    vocab_size, skip_grams, label = tokenize_text(text)
    model = word2vec_model(vocab_size)
    targets, contexts = zip(*skip_grams)
    targets = np.array(targets, dtype=np.int32)
    contexts = np.array(contexts, dtype=np.int32)
    labels = np.array(label, dtype=np.int32)

    model.fit(
        x=[targets, contexts],
        y=labels,
        batch_size=1024,
        epochs=10,
    )

    word_embeddings = model.get_layer("word_embedding").get_weights()[0]
    # Save word embeddings to file
    np.save("word_embeddings.npy", word_embeddings)


if __name__ == "__main__":
    main()
