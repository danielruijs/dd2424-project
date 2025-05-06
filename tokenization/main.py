import kagglehub
import os
import datetime

# Suppress TensorFlow Info messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = (
    "3"  # 0: all messages, 1: filter out INFO messages, 2: filter out WARNING messages, 3: filter out ERROR messages
)
import tensorflow as tf
from tensorflow import keras
from keras import mixed_precision
import argparse
from models import BaseRNN, LSTM, LSTM2, OneStep
from transformer_model import Transformer, TransformerOneStep
import re
import pickle
import numpy as np
from nltk.tokenize import RegexpTokenizer

BATCH_SIZE = 32
BUFFER_SIZE = 10000
EPOCHS = 20
SEQ_LENGTH = 300
LR = 0.001
HIDDEN_UNITS = 1024
# For transformer
NUM_LAYERS = 1
DFF = 1024
NUM_HEADS = 4

mixed_precision.set_global_policy("float32")


def load_text(kaggle_url, verbose=True):
    # Download latest version
    path = kagglehub.dataset_download(kaggle_url)

    if verbose:
        print("Harry Potter books downloaded to: ", path)

    # List all text files
    file_paths = [
        os.path.join(path, fname)
        for fname in os.listdir(path)
        if fname.endswith(".txt")
    ]

    # Read all text files and concatenate them
    texts = []
    for path in file_paths:
        with open(path, "r", encoding="utf-8") as f:
            texts.append(f.read())

    # Concatenate all texts into a single string
    text = "\n".join(texts)

    if verbose:
        print("Harry Potter books loaded. Total characters: ", len(text))

    pattern = r"\b[a-zA-Z\-'’]+\b"
    all_words = re.findall(pattern, text)
    unique_words = set(all_words)
    if verbose:
        print("Total words: ", len(all_words))
        print("Unique words: ", len(unique_words))

    ngrams = {
        1: set(tf.strings.ngrams(tf.constant([all_words]), 1).numpy()[0]),
        2: set(tf.strings.ngrams(tf.constant([all_words]), 2).numpy()[0]),
        3: set(tf.strings.ngrams(tf.constant([all_words]), 3).numpy()[0]),
        4: set(tf.strings.ngrams(tf.constant([all_words]), 4).numpy()[0]),
    }

    return text, unique_words, ngrams


def percentage_ngrams(sample, ngrams):
    """
    Calculate the percentage of correctly spelt ngrams in the sample.
    """
    percentages = {}
    for ngram_size in ngrams.keys():
        pattern = r"\b[a-zA-Z\-'’]+\b"
        words_in_sample = re.findall(pattern, sample)
        ngrams_in_sample = tf.strings.ngrams(
            tf.constant([words_in_sample]), ngram_size
        ).numpy()[0]
        correctly_spelt_ngrams = [
            ngram for ngram in ngrams_in_sample if ngram in ngrams[ngram_size]
        ]
        percentages[ngram_size] = 100 * (
            len(correctly_spelt_ngrams) / len(ngrams_in_sample)
        )
    return percentages


def split_input_target(sequence):
    input_text = sequence[:-1]
    target_text = sequence[1:]
    return input_text, target_text


def create_dataset(
    text,
    torch_tokenizer,
    train_split=0.8,
    val_split=0.1,
    batch_size=32,
    seq_length=100,
    verbose=True,
):
    """
    Splits the text into training, validation, and test datasets.
    train_split: proportion of data to use for training.
    val_split: proportion of data to use for validation.
    The remaining data will be used for testing.
    """
    total_len = len(text)
    train_end = int(total_len * train_split)
    val_end = int(total_len * (train_split + val_split))

    train_text = text[:train_end]
    val_text = text[train_end:val_end]
    test_text = text[val_end:]

    if verbose:
        print(f"Train length: {len(train_text)}")
        print(f"Val length: {len(val_text)}")
        print(f"Test length: {len(test_text)}")

    def text_to_dataset(text):
        regex_tokenizer = RegexpTokenizer(r"\b\w+'\w+|\w+|\S")
        tokens = regex_tokenizer.tokenize(text)
        sequences = torch_tokenizer.texts_to_sequences([tokens])[0]
        ids_dataset = tf.data.Dataset.from_tensor_slices(sequences)
        sequences = ids_dataset.batch(seq_length + 1, drop_remainder=True)
        dataset = sequences.map(split_input_target)
        return dataset

    train_dataset = text_to_dataset(train_text)
    val_dataset = text_to_dataset(val_text)
    test_dataset = text_to_dataset(test_text)

    train_dataset = (
        train_dataset.shuffle(BUFFER_SIZE)
        .batch(batch_size, drop_remainder=True)
        .prefetch(tf.data.experimental.AUTOTUNE)
    )
    val_dataset = val_dataset.batch(batch_size, drop_remainder=True).prefetch(
        tf.data.experimental.AUTOTUNE
    )
    test_dataset = test_dataset.batch(batch_size, drop_remainder=True).prefetch(
        tf.data.experimental.AUTOTUNE
    )

    return (
        train_dataset,
        val_dataset,
        test_dataset,
    )


def create_model(
    model_name,
    word_embeddings,
    train_dataset,
    learning_rate,
    hidden_units,
    num_layers,
    dff,
    num_heads,
    verbose=True,
):
    if model_name == "rnn":
        model = BaseRNN(hidden_units, word_embeddings)
    elif model_name == "lstm":
        model = LSTM(hidden_units, word_embeddings)
    elif model_name == "lstm2":
        model = LSTM2(hidden_units, word_embeddings)
    elif model_name == "transformer":
        model = Transformer(word_embeddings, num_layers, dff, num_heads)

    loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=loss)

    if verbose:
        for input_example_batch, target_example_batch in train_dataset.take(1):
            example_batch_predictions = model(input_example_batch)
            print(
                example_batch_predictions.shape,
                "# (batch_size, sequence_length, vocab_size)",
            )

        model.summary()
    return model


class GenerateTextCallback(tf.keras.callbacks.Callback):
    def __init__(
        self,
        one_step_model,
        log_dir,
        ngrams,
        tokenizer,
        start_string=".",
        num_generate=300,
    ):
        super().__init__()
        # Store the OneStep model instance (important: pass the instance, not the class)
        self.one_step_model = one_step_model
        self.start_string = start_string
        self.num_generate = num_generate
        self.log_dir = log_dir
        self.ngrams = ngrams
        self.tokenizer = tokenizer
        self.idx2word = tokenizer.index_word

    def on_epoch_end(self, epoch, logs=None):
        print(f"\n\n--- Generating text after epoch {epoch + 1} ---")
        states = None
        # Use the start_string provided during initialization
        next_token = self.tokenizer.texts_to_sequences([self.start_string])[0]
        next_token = tf.constant(next_token)
        result = [next_token]

        for _ in range(self.num_generate):
            next_token, states = self.one_step_model.generate_one_step(
                next_token, states=states
            )
            result.append(next_token)

        # Convert token IDs back to words using index_word dictionary
        token_ids = [tensor.numpy()[0] for tensor in result]
        words = []
        for token_id in token_ids:
            words.append(self.idx2word[int(token_id)])

        # Join words to form the generated text
        result_text = " ".join(words)
        print(result_text, "\n\n" + "_" * 80)
        file_writer = tf.summary.create_file_writer(self.log_dir)
        with file_writer.as_default():
            tf.summary.text("Generated Text", result_text, step=epoch)

        percentage_ngrams_final = percentage_ngrams(result_text, self.ngrams)
        with file_writer.as_default():
            for ngram_size, pct in percentage_ngrams_final.items():
                tf.summary.scalar(
                    f"{ngram_size}-gram %",
                    pct,
                    step=epoch,
                )


class GenerateTextCallbackTransformer(tf.keras.callbacks.Callback):
    def __init__(
        self,
        one_step_model,
        log_dir,
        ngrams,
        seq_length,
        tokenizer,
        start_string=".",
        num_generate=300,
    ):
        super().__init__()
        # Store the OneStep model instance (important: pass the instance, not the class)
        self.one_step_model = one_step_model
        self.start_string = start_string
        self.num_generate = num_generate
        self.log_dir = log_dir
        self.ngrams = ngrams
        self.seq_length = seq_length
        self.tokenizer = tokenizer
        self.idx2word = tokenizer.index_word

    def on_epoch_end(self, epoch, logs=None):
        print(f"\n\n--- Generating text after epoch {epoch + 1} ---")
        # Use the start_string provided during initialization
        next_token = self.tokenizer.texts_to_sequences([self.start_string])[0]
        next_token = tf.expand_dims(tf.constant(next_token), 1)
        result = next_token

        for _ in range(self.num_generate):
            if len(result) > self.seq_length:
                input_seq = result[-self.seq_length :]
            else:
                input_seq = result
            next_token = self.one_step_model.generate_one_step(input_seq)
            result = tf.concat([result, tf.expand_dims(next_token, 1)], axis=1)

        # Convert token IDs back to words using index_word dictionary
        token_ids = result.numpy()[0]
        words = []
        for token_id in token_ids:
            words.append(self.idx2word[int(token_id)])

        # Join words to form the generated text
        result_text = " ".join(words)
        print(result_text, "\n\n" + "_" * 80)
        file_writer = tf.summary.create_file_writer(self.log_dir)
        with file_writer.as_default():
            tf.summary.text("Generated Text", result_text, step=epoch)

        percentage_ngrams_final = percentage_ngrams(result_text, self.ngrams)
        with file_writer.as_default():
            for ngram_size, pct in percentage_ngrams_final.items():
                tf.summary.scalar(
                    f"{ngram_size}-gram %",
                    pct,
                    step=epoch,
                )


def train(
    model_name,
    train_dataset,
    val_dataset,
    word_embeddings,
    tokenizer,
    ngrams,
    seq_length=100,
    hyperparameter_tuning=False,
    learning_rate=0.001,
    hidden_units=1024,
    num_layers=1,
    dff=2048,
    num_heads=8,
):
    model = create_model(
        model_name,
        word_embeddings,
        train_dataset,
        learning_rate=learning_rate,
        hidden_units=hidden_units,
        num_layers=num_layers,
        dff=dff,
        num_heads=num_heads,
        verbose=not hyperparameter_tuning,
    )

    # Create unique log directory for this run
    log_dir = f"logs/fit/{model_name}/" + datetime.datetime.now().strftime(
        "%Y%m%d-%H%M%S"
    )
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,  # Generate histograms of weights every epoch
        write_graph=True,  # Visualize the model graph
        update_freq="epoch",  # Update logs at the end of each epoch
    )

    if model_name == "transformer":
        one_step_model_for_callback = TransformerOneStep(model)
    else:
        one_step_model_for_callback = OneStep(model)

    checkpoint_dir = "./training_checkpoints"
    best_checkpoint_filepath = os.path.join(
        checkpoint_dir, f"{model_name}_best.weights.h5"
    )
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=best_checkpoint_filepath,
        save_weights_only=True,
        monitor="val_loss",
        mode="min",
        save_best_only=True,
    )

    if model_name == "transformer":
        generate_text_callback = GenerateTextCallbackTransformer(
            one_step_model_for_callback,
            log_dir=log_dir,
            ngrams=ngrams,
            seq_length=seq_length,
            tokenizer=tokenizer,
        )
    else:
        generate_text_callback = GenerateTextCallback(
            one_step_model_for_callback,
            log_dir=log_dir,
            ngrams=ngrams,
            tokenizer=tokenizer,
        )

    callbacks = [checkpoint_callback]
    if not hyperparameter_tuning:
        callbacks.append(tensorboard_callback)
        callbacks.append(generate_text_callback)

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1 if not hyperparameter_tuning else 2,
    )

    return model, history, log_dir, best_checkpoint_filepath


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="rnn",
        choices=["rnn", "lstm", "lstm2", "transformer"],
        help="Model architecture to use",
    )
    args = parser.parse_args()

    # Load word embeddings and tokenizer
    word_embeddings = np.load("word_embeddings.npy")
    with open("tokenizer.pkl", "rb") as f:
        torch_tokenizer = pickle.load(f)

    idx2word = torch_tokenizer.index_word

    text, _, ngrams = load_text("shubhammaindola/harry-potter-books")

    (
        train_dataset,
        val_dataset,
        test_dataset,
    ) = create_dataset(
        text,
        torch_tokenizer,
        batch_size=BATCH_SIZE,
        seq_length=SEQ_LENGTH,
    )

    model, _, log_dir, best_checkpoint_filepath = train(
        args.model,
        train_dataset,
        val_dataset,
        word_embeddings,
        torch_tokenizer,
        ngrams,
        seq_length=SEQ_LENGTH,
        learning_rate=LR,
        hidden_units=HIDDEN_UNITS,
        num_layers=NUM_LAYERS,  # For transformer
        dff=DFF,  # For transformer
        num_heads=NUM_HEADS,  # For transformer
    )

    print(f"\nLoading best weights from: {best_checkpoint_filepath}")
    try:
        model.load_weights(best_checkpoint_filepath)
        print("Best weights loaded successfully.")
    except Exception as e:
        print(
            f"Error loading weights: {e}. Proceeding with the final weights from training."
        )

    train_loss = model.evaluate(train_dataset)
    print(f"\nTrain Loss: {train_loss}")
    val_loss = model.evaluate(val_dataset)
    print(f"\nValidation Loss: {val_loss}")
    test_loss = model.evaluate(test_dataset)
    print(f"\nTest Loss: {test_loss}")

    if args.model == "transformer":
        one_step_model = TransformerOneStep(model)
        next_token = torch_tokenizer.texts_to_sequences(["."])[0]
        next_token = tf.constant(next_token)
        result = [next_token]

        for _ in range(1000):
            if len(result) > SEQ_LENGTH:
                input_seq = result[-SEQ_LENGTH:]
            else:
                input_seq = result
            next_token = one_step_model.generate_one_step(input_seq)
            result.append(next_token)
    else:
        one_step_model = OneStep(model)
        states = None
        next_token = torch_tokenizer.texts_to_sequences(["."])[0]
        next_token = tf.constant(next_token)
        result = [next_token]

        for _ in range(1000):
            next_token, states = one_step_model.generate_one_step(
                next_token, states=states
            )
            result.append(next_token)

    # Convert token IDs back to words using index_word dictionary
    token_ids = [tensor.numpy()[0] for tensor in result]
    words = []
    for token_id in token_ids:
        words.append(idx2word[int(token_id)])

    # Join words to form the generated text
    result_text = " ".join(words)
    print(result_text, "\n\n" + "_" * 80)
    file_writer = tf.summary.create_file_writer(log_dir)
    with file_writer.as_default():
        tf.summary.text("Final Generated Text", result_text, step=0)

    percentage_ngrams_final = percentage_ngrams(result_text, ngrams)
    with file_writer.as_default():
        for ngram_size, pct in percentage_ngrams_final.items():
            tf.summary.scalar(
                f"Final sample {ngram_size}-gram %",
                pct,
                step=0,
            )
        tf.summary.scalar("Final Train Loss", train_loss, step=0)
        tf.summary.scalar("Final Validation Loss", val_loss, step=0)
        tf.summary.scalar("Final Test Loss", test_loss, step=0)


if __name__ == "__main__":
    main()
