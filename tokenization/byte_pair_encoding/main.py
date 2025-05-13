import kagglehub
import os
import pickle
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
from nltk.tokenize import RegexpTokenizer
import re

BATCH_SIZE = 64
LR = 0.001
HIDDEN_UNITS = 1024
SEQ_LENGTH = 300
# For transformer
NUM_LAYERS = 2
D_MODEL = 256
DFF = 1024
NUM_HEADS = 12

EPOCHS = 20
BUFFER_SIZE = 10000

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

    return text, all_words, unique_words, ngrams


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


def tokenize_text(text, splits):
    # regex_tokenizer = RegexpTokenizer(r"\b\w+'\w+|\w+|\S")
    regex_tokenizer = RegexpTokenizer(r"\s+|[\w']+|[^\w\s]")
    words = regex_tokenizer.tokenize(text)

    tokens = []
    for word in words:
        if word in splits:
            tokens.extend(splits[word])
        else:
            tokens.extend(list(word))
    return tokens


def create_dataset(
    text, train_split=0.8, val_split=0.1, batch_size=32, seq_length=100, verbose=True
):
    """
    Splits the text into training, validation, and test datasets.
    train_split: proportion of data to use for training.
    val_split: proportion of data to use for validation.
    The remaining data will be used for testing.
    """
    with open("vocab.pkl", "rb") as f:
        vocab = pickle.load(f)

    print("Vocab:\n", vocab)

    with open("splits.pkl", "rb") as f:
        splits = pickle.load(f)

    token_to_id = {tok: idx for idx, tok in enumerate(vocab)}
    id_to_token = {idx: tok for tok, idx in token_to_id.items()}

    # print(list(splits.items())[:100])
    # print()

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
        tokens = tokenize_text(text, splits)
        ids = [token_to_id[token] for token in tokens]
        ids_dataset = tf.data.Dataset.from_tensor_slices(ids)
        print(len(ids_dataset), "tokens in dataset")
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

    vocab_size = len(vocab)
    if verbose:
        print(f"Vocabulary size: {vocab_size}")

    return (
        train_dataset,
        val_dataset,
        test_dataset,
        vocab_size,
        token_to_id,
        id_to_token,
        splits,
    )


def create_model(
    model_name,
    vocab_size,
    train_dataset,
    learning_rate,
    hidden_units,
    num_layers,
    d_model,
    dff,
    num_heads,
    verbose=True,
):
    if model_name == "rnn":
        model = BaseRNN(vocab_size, hidden_units)
    elif model_name == "lstm":
        model = LSTM(vocab_size, hidden_units)
    elif model_name == "lstm2":
        model = LSTM2(vocab_size, hidden_units)
    elif model_name == "transformer":
        model = Transformer(vocab_size, num_layers, d_model, dff, num_heads)

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
        splits,
        id_to_token,
        token_to_id,
        start_string=".",
        num_generate=1000,
    ):
        super().__init__()
        # Store the OneStep model instance (important: pass the instance, not the class)
        self.one_step_model = one_step_model
        self.start_string = start_string
        self.num_generate = num_generate
        self.log_dir = log_dir
        self.ngrams = ngrams
        self.splits = splits
        self.id_to_token = id_to_token
        self.token_to_id = token_to_id

    def on_epoch_end(self, epoch, logs=None):
        print(f"\n\n--- Generating text after epoch {epoch + 1} ---")
        states = None
        # Use the start_string provided during initialization
        next_token = self.token_to_id[tokenize_text(self.start_string, self.splits)[0]]
        next_token = tf.constant([next_token])
        result = [next_token]

        for _ in range(self.num_generate):
            next_token, states = self.one_step_model.generate_one_step(
                next_token, states=states
            )
            result.append(next_token)

        token_ids = [tensor.numpy()[0] for tensor in result]
        result_text = ""
        for token_id in token_ids:
            result_text += self.id_to_token[token_id]

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
        splits,
        id_to_token,
        token_to_id,
        start_string=".",
        num_generate=1000,
    ):
        super().__init__()
        # Store the OneStep model instance (important: pass the instance, not the class)
        self.one_step_model = one_step_model
        self.start_string = start_string
        self.num_generate = num_generate
        self.log_dir = log_dir
        self.ngrams = ngrams
        self.seq_length = seq_length
        self.splits = splits
        self.id_to_token = id_to_token
        self.token_to_id = token_to_id

    def on_epoch_end(self, epoch, logs=None):
        print(f"\n\n--- Generating text after epoch {epoch + 1} ---")
        # Use the start_string provided during initialization
        next_token = self.token_to_id[tokenize_text(self.start_string, self.splits)[0]]
        next_token = tf.expand_dims(tf.constant([next_token]), 1)
        result = next_token

        for _ in range(self.num_generate):
            if tf.shape(result)[1] > self.seq_length:
                input_seq = result[:, -self.seq_length :]
            else:
                input_seq = result

            next_token = self.one_step_model.generate_one_step(input_seq)
            result = tf.concat([result, tf.expand_dims(next_token, 1)], axis=1)

        # Convert token IDs back to words using index_word dictionary
        token_ids = result.numpy()[0]
        result_text = ""
        for token_id in token_ids:
            result_text += self.id_to_token[token_id]

        # Join words to form the generated text
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
    vocab_size,
    token_to_id,
    id_to_token,
    splits,
    ngrams,
    seq_length=100,
    hyperparameter_tuning=False,
    learning_rate=0.001,
    hidden_units=1024,
    num_layers=1,
    d_model=512,
    dff=2048,
    num_heads=8,
):
    model = create_model(
        model_name,
        vocab_size,
        train_dataset,
        learning_rate=learning_rate,
        hidden_units=hidden_units,
        num_layers=num_layers,
        d_model=d_model,
        dff=dff,
        num_heads=num_heads,
        verbose=not hyperparameter_tuning,
    )

    # Create unique log directory for this run
    if model_name == "transformer":
        log_dir = (
            f"logs/fit/{model_name}_{num_layers}/"
            + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        )
    else:
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
        one_step_model_for_callback = OneStep(model, id_to_token, token_to_id)

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
            splits=splits,
            id_to_token=id_to_token,
            token_to_id=token_to_id,
        )
    else:
        generate_text_callback = GenerateTextCallback(
            one_step_model_for_callback,
            log_dir=log_dir,
            ngrams=ngrams,
            splits=splits,
            id_to_token=id_to_token,
            token_to_id=token_to_id,
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

    text, _, _, ngrams = load_text("shubhammaindola/harry-potter-books")

    (
        train_dataset,
        val_dataset,
        test_dataset,
        vocab_size,
        token_to_id,
        id_to_token,
        splits,
    ) = create_dataset(text, batch_size=BATCH_SIZE, seq_length=SEQ_LENGTH)

    model, _, log_dir, best_checkpoint_filepath = train(
        args.model,
        train_dataset,
        val_dataset,
        vocab_size,
        token_to_id,
        id_to_token,
        splits,
        ngrams,
        seq_length=SEQ_LENGTH,
        learning_rate=LR,
        hidden_units=HIDDEN_UNITS,
        num_layers=NUM_LAYERS,  # For transformer
        d_model=D_MODEL,  # For transformer
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
        next_token = token_to_id[tokenize_text(".", splits)[0]]
        next_token = tf.expand_dims(tf.constant([next_token]), 1)
        result = next_token

        for _ in range(1000):
            if tf.shape(result)[1] > SEQ_LENGTH:
                input_seq = result[:, -SEQ_LENGTH:]
            else:
                input_seq = result

            next_token = one_step_model.generate_one_step(input_seq)
            result = tf.concat([result, tf.expand_dims(next_token, 1)], axis=1)

        token_ids = result.numpy()[0]
    else:
        one_step_model = OneStep(model, id_to_token, token_to_id)
        states = None
        next_token = token_to_id[tokenize_text(".", splits)[0]]
        next_token = tf.constant([next_token])
        result = [next_token]

        for _ in range(1000):
            next_token, states = one_step_model.generate_one_step(
                next_token, states=states
            )
            result.append(next_token)

        token_ids = [tensor.numpy()[0] for tensor in result]

    result_text = ""
    for token_id in token_ids:
        result_text += id_to_token[token_id]

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
