import kagglehub
import os

# Suppress TensorFlow Info messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = (
    "1"  # 0: all messages, 1: filter out INFO messages, 2: filter out WARNING messages, 3: filter out ERROR messages
)
import tensorflow as tf
from tensorflow import keras

BATCH_SIZE = 64
BUFFER_SIZE = 10000


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


def text_from_ids(ids, chars_from_ids):
    return tf.strings.reduce_join(chars_from_ids(ids), axis=-1).numpy().decode("utf-8")


def split_input_target(sequence):
    input_text = sequence[:-1]
    target_text = sequence[1:]
    return input_text, target_text


def create_dataset(text, seq_length):
    vocab = sorted(set(text))
    print("Vocab size: ", len(vocab))

    chars = tf.strings.unicode_split(text, input_encoding="UTF-8")
    print(chars)

    ids_from_chars = tf.keras.layers.StringLookup(
        vocabulary=list(vocab), mask_token=None
    )

    ids = ids_from_chars(chars)
    print(ids)

    chars_from_ids = tf.keras.layers.StringLookup(
        vocabulary=ids_from_chars.get_vocabulary(), invert=True, mask_token=None
    )

    # print(text_from_ids(ids, chars_from_ids))

    ids_dataset = tf.data.Dataset.from_tensor_slices(ids)

    sequences = ids_dataset.batch(seq_length + 1, drop_remainder=True)

    dataset = sequences.map(split_input_target)

    for input_example, target_example in dataset.take(1):
        print("Input :", text_from_ids(input_example, chars_from_ids))
        print("Target:", text_from_ids(target_example, chars_from_ids))

    dataset = (
        dataset.shuffle(BUFFER_SIZE)
        .batch(BATCH_SIZE, drop_remainder=True)
        .prefetch(tf.data.experimental.AUTOTUNE)
    )

    vocab_size = len(ids_from_chars.get_vocabulary())

    return dataset, vocab_size, ids_from_chars, chars_from_ids


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
            states = [tf.zeros((BATCH_SIZE, self.rnn.units))]
        x, states = self.rnn(x, initial_state=states, training=training)
        x = self.dense(x, training=training)

        if return_state:
            return x, states
        else:
            return x


def create_model(vocab_size, embedding_dim, rnn_units):
    return Model(vocab_size, embedding_dim, rnn_units)


def set_gpu_mode():
    # List available physical GPUs
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            # Restrict TensorFlow to only use the first GPU
            tf.config.set_visible_devices(gpus[0], "GPU")

            # Optionally, set memory growth to prevent TensorFlow from allocating all GPU memory
            tf.config.experimental.set_memory_growth(gpus[0], True)

            print(f"Using GPU: {gpus[0]}")
        except RuntimeError as e:
            print(e)
    else:
        print("No GPU found, using CPU.")


def main():
    set_gpu_mode()
    seq_length = 100
    embedding_dim = 256
    rnn_units = 1024

    text = load_text("shubhammaindola/harry-potter-books")
    dataset, vocab_size, ids_from_chars, chars_from_ids = create_dataset(
        text, seq_length
    )
    model = create_model(vocab_size, embedding_dim, rnn_units)

    for input_example_batch, target_example_batch in dataset.take(1):
        example_batch_predictions = model(input_example_batch)
        print(
            example_batch_predictions.shape,
            "# (batch_size, sequence_length, vocab_size)",
        )

    model.summary()

    sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
    sampled_indices = tf.squeeze(sampled_indices, axis=-1).numpy()

    print("Input:\n", text_from_ids(input_example_batch[0], chars_from_ids))
    print()
    print("Next Char Predictions:\n", text_from_ids(sampled_indices, chars_from_ids))

    loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer="adam", loss=loss)

    # Directory where the checkpoints will be saved
    checkpoint_dir = "./training_checkpoints"
    # Name of the checkpoint files
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}.weights.h5")
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_prefix, save_weights_only=True
    )

    EPOCHS = 20
    history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])


if __name__ == "__main__":
    main()
