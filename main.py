import kagglehub
import os
import json
import datetime

# Suppress TensorFlow Info messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = (
    "1"  # 0: all messages, 1: filter out INFO messages, 2: filter out WARNING messages, 3: filter out ERROR messages
)
import tensorflow as tf
from tensorflow import keras
import argparse
from models import BaseRNN, LSTM, LSTM2
import re

BATCH_SIZE = 64
BUFFER_SIZE = 10000
EPOCHS = 20
SEQ_LENGTH = 100
LR = 0.001
HIDDEN_UNITS = 1024


def load_text(kaggle_url):
    # Download latest version
    path = kagglehub.dataset_download(kaggle_url)

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

    print("Harry Potter books loaded. Total characters: ", len(text))

    pattern = r"\b[a-zA-Z\-'’]+\b"
    all_words = re.findall(pattern, text)
    print("Total words: ", len(all_words))
    unique_words = set(all_words)
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


def create_dataset(text, train_split=0.8, val_split=0.1, batch_size=64):
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

    print(f"Train length: {len(train_text)}")
    print(f"Val length: {len(val_text)}")
    print(f"Test length: {len(test_text)}")

    vocab = sorted(set(text))

    ids_from_chars = tf.keras.layers.StringLookup(
        vocabulary=list(vocab), mask_token=None
    )
    chars_from_ids = tf.keras.layers.StringLookup(
        vocabulary=ids_from_chars.get_vocabulary(), invert=True, mask_token=None
    )

    def text_to_dataset(text):
        chars = tf.strings.unicode_split(text, input_encoding="UTF-8")
        ids = ids_from_chars(chars)
        ids_dataset = tf.data.Dataset.from_tensor_slices(ids)
        sequences = ids_dataset.batch(SEQ_LENGTH + 1, drop_remainder=True)
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

    vocab_size = len(ids_from_chars.get_vocabulary())
    print(f"Vocabulary size: {vocab_size}")

    return (
        train_dataset,
        val_dataset,
        test_dataset,
        vocab_size,
        ids_from_chars,
        chars_from_ids,
    )


def create_model(model_name, vocab_size, train_dataset, learning_rate, hidden_units):
    if model_name == "rnn":
        model = BaseRNN(vocab_size, hidden_units)
    elif model_name == "lstm":
        model = LSTM(vocab_size, hidden_units)
    elif model_name == "lstm2":
        model = LSTM2(vocab_size, hidden_units)

    for input_example_batch, target_example_batch in train_dataset.take(1):
        example_batch_predictions = model(input_example_batch)
        print(
            example_batch_predictions.shape,
            "# (batch_size, sequence_length, vocab_size)",
        )

    model.summary()

    loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=loss)
    return model


class GenerateTextCallback(tf.keras.callbacks.Callback):
    def __init__(self, one_step_model, log_dir, start_string=".", num_generate=200):
        super().__init__()
        # Store the OneStep model instance (important: pass the instance, not the class)
        self.one_step_model = one_step_model
        self.start_string = start_string
        self.num_generate = num_generate
        self.log_dir = log_dir

    def on_epoch_end(self, epoch, logs=None):
        print(f"\n\n--- Generating text after epoch {epoch + 1} ---")
        states = None
        # Use the start_string provided during initialization
        next_char = tf.constant([self.start_string])
        result = [next_char]

        for _ in range(self.num_generate):
            next_char, states = self.one_step_model.generate_one_step(
                next_char, states=states
            )
            result.append(next_char)

        result = tf.strings.join(result)
        result_text = result[0].numpy().decode("utf-8")
        print(result_text, "\n\n" + "_" * 80)
        file_writer = tf.summary.create_file_writer(self.log_dir)
        with file_writer.as_default():
            tf.summary.text("Generated Text", result_text, step=epoch)


class OneStep(tf.keras.Model):
    def __init__(self, model, chars_from_ids, ids_from_chars, temperature=1.0):
        super().__init__()
        self.temperature = temperature
        self.model = model
        self.chars_from_ids = chars_from_ids
        self.ids_from_chars = ids_from_chars

        # Create a mask to prevent "[UNK]" from being generated.
        skip_ids = self.ids_from_chars(["[UNK]"])[:, None]
        sparse_mask = tf.SparseTensor(
            # Put a -inf at each bad index.
            values=[-float("inf")] * len(skip_ids),
            indices=skip_ids,
            # Match the shape to the vocabulary
            dense_shape=[len(ids_from_chars.get_vocabulary())],
        )
        self.prediction_mask = tf.sparse.to_dense(sparse_mask)

    @tf.function
    def generate_one_step(self, inputs, states=None):
        # Convert strings to token IDs.
        input_chars = tf.strings.unicode_split(inputs, "UTF-8")
        input_ids = self.ids_from_chars(input_chars).to_tensor()

        # Run the model.
        # predicted_logits.shape is [batch, char, next_char_logits]
        predicted_logits, states = self.model(
            inputs=input_ids, states=states, return_state=True
        )
        # Only use the last prediction.
        predicted_logits = predicted_logits[:, -1, :]
        predicted_logits = predicted_logits / self.temperature
        # Apply the prediction mask: prevent "[UNK]" from being generated.
        predicted_logits = predicted_logits + self.prediction_mask

        # Sample the output logits to generate token IDs.
        predicted_ids = tf.random.categorical(predicted_logits, num_samples=1)
        predicted_ids = tf.squeeze(predicted_ids, axis=-1)

        # Convert from token ids to characters
        predicted_chars = self.chars_from_ids(predicted_ids)

        # Return the characters and model state.
        return predicted_chars, states


def train(
    model_name,
    train_dataset,
    val_dataset,
    vocab_size,
    ids_from_chars,
    chars_from_ids,
    hyperparameter_tuning=False,
    learning_rate=0.001,
    hidden_units=1024,
):
    model = create_model(
        model_name,
        vocab_size,
        train_dataset,
        learning_rate=learning_rate,
        hidden_units=hidden_units,
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

    one_step_model_for_callback = OneStep(model, chars_from_ids, ids_from_chars)

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

    generate_text_callback = GenerateTextCallback(
        one_step_model_for_callback, log_dir=log_dir
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
    )

    return model, history, log_dir, best_checkpoint_filepath


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="rnn",
        choices=["rnn", "lstm", "lstm2"],
        help="Model architecture to use",
    )
    args = parser.parse_args()

    text, _, ngrams = load_text("shubhammaindola/harry-potter-books")

    (
        train_dataset,
        val_dataset,
        test_dataset,
        vocab_size,
        ids_from_chars,
        chars_from_ids,
    ) = create_dataset(text, batch_size=BATCH_SIZE)

    model, _, log_dir, best_checkpoint_filepath = train(
        args.model,
        train_dataset,
        val_dataset,
        vocab_size,
        ids_from_chars,
        chars_from_ids,
        learning_rate=LR,
        hidden_units=HIDDEN_UNITS,
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

    one_step_model = OneStep(model, chars_from_ids, ids_from_chars)
    states = None
    next_char = tf.constant(["."])
    result = [next_char]

    for _ in range(1000):
        next_char, states = one_step_model.generate_one_step(next_char, states=states)
        result.append(next_char)

    result = tf.strings.join(result)
    result_text = result[0].numpy().decode("utf-8")
    print(result_text, "\n\n" + "_" * 80)
    file_writer = tf.summary.create_file_writer(log_dir)
    with file_writer.as_default():
        tf.summary.text("Final Generated Text", result_text, step=0)

    percentage_ngrams_final = percentage_ngrams(result_text, ngrams)
    with file_writer.as_default():
        for ngram_size, pct in percentage_ngrams_final.items():
            tf.summary.scalar(
                f"Final sample correct {ngram_size}-gram percentage",
                pct,
                step=0,
            )
        tf.summary.scalar("Evaluation/Train Loss (Best Weights)", train_loss, step=0)
        tf.summary.scalar("Evaluation/Validation Loss (Best Weights)", val_loss, step=0)
        tf.summary.scalar("Evaluation/Test Loss (Best Weights)", test_loss, step=0)


if __name__ == "__main__":
    main()
