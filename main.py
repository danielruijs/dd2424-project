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

BATCH_SIZE = 64
BUFFER_SIZE = 10000
EPOCHS = 2
SEQ_LENGTH = 100
EMBEDDING_DIM = 256
RNN_UNITS = 1024
LSTM_UNITS = 1024


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

    return text


def split_input_target(sequence):
    input_text = sequence[:-1]
    target_text = sequence[1:]
    return input_text, target_text


def create_dataset(text, train_split=0.8, val_split=0.1):
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
        .batch(BATCH_SIZE, drop_remainder=True)
        .prefetch(tf.data.experimental.AUTOTUNE)
    )
    val_dataset = val_dataset.batch(BATCH_SIZE, drop_remainder=True).prefetch(
        tf.data.experimental.AUTOTUNE
    )
    test_dataset = test_dataset.batch(BATCH_SIZE, drop_remainder=True).prefetch(
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


class Model(keras.Model):
    def __init__(self, vocab_size, embedding_dim, rnn_units):
        super().__init__()
        self.embedding = keras.layers.Embedding(vocab_size, embedding_dim)
        self.rnn = keras.layers.SimpleRNN(
            rnn_units, return_sequences=True, return_state=True
        )
        self.dense = keras.layers.Dense(vocab_size)

    def call(self, inputs, states=None, return_state=False, training=False):
        x = inputs
        x = self.embedding(x, training=training)
        if states is None:
            batch_size = tf.shape(inputs)[0]
            states = [tf.zeros((batch_size, self.rnn.units))]
        x, states = self.rnn(x, initial_state=states, training=training)
        x = self.dense(x, training=training)

        if return_state:
            return x, states
        else:
            return x


def create_model(model_name, vocab_size, train_dataset):
    if model_name == "rnn":
        model = BaseRNN(vocab_size, EMBEDDING_DIM, RNN_UNITS)
    elif model_name == "lstm":
        model = LSTM(vocab_size, EMBEDDING_DIM, LSTM_UNITS)
    elif model_name == "lstm2":
        model = LSTM2(vocab_size, EMBEDDING_DIM, LSTM_UNITS)

    for input_example_batch, target_example_batch in train_dataset.take(1):
        example_batch_predictions = model(input_example_batch)
        print(
            example_batch_predictions.shape,
            "# (batch_size, sequence_length, vocab_size)",
        )

    model.summary()

    loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer="adam", loss=loss)
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
):
    model = create_model(model_name, vocab_size, train_dataset)

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
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}.weights.h5")
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_prefix, save_weights_only=True
    )

    generate_text_callback = GenerateTextCallback(
        one_step_model_for_callback, log_dir=log_dir
    )

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        callbacks=[checkpoint_callback, generate_text_callback, tensorboard_callback],
    )

    return model, history, log_dir


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

    text = load_text("shubhammaindola/harry-potter-books")
    (
        train_dataset,
        val_dataset,
        test_dataset,
        vocab_size,
        ids_from_chars,
        chars_from_ids,
    ) = create_dataset(text)

    model, history, log_dir = train(
        args.model,
        train_dataset,
        val_dataset,
        vocab_size,
        ids_from_chars,
        chars_from_ids,
    )

    test_loss = model.evaluate(test_dataset)
    print(f"\nTest Loss: {test_loss}")

    with open("training_history.json", "w") as f:
        json.dump(history.history, f)

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


if __name__ == "__main__":
    main()
