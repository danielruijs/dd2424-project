import tensorflow as tf
from tensorflow import keras


class BaseRNN(keras.Model):
    def __init__(self, vocab_size, rnn_units):
        super().__init__()
        self.embedding = keras.layers.Embedding(vocab_size, vocab_size)
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


class LSTM(keras.Model):
    def __init__(self, vocab_size, lstm_units):
        super().__init__()
        self.embedding = keras.layers.Embedding(vocab_size, vocab_size)
        self.lstm = keras.layers.LSTM(
            lstm_units, return_sequences=True, return_state=True
        )
        self.dense = keras.layers.Dense(vocab_size)

    def call(self, inputs, states=None, return_state=False, training=False):
        x = inputs
        x = self.embedding(x, training=training)
        if states is None:
            batch_size = tf.shape(inputs)[0]
            states = [
                tf.zeros((batch_size, self.lstm.units)),
                tf.zeros((batch_size, self.lstm.units)),
            ]
        x, h, c = self.lstm(x, initial_state=states, training=training)
        states = [h, c]
        x = self.dense(x, training=training)

        if return_state:
            return x, states
        else:
            return x


class LSTM2(keras.Model):
    def __init__(self, vocab_size, lstm_units):
        super().__init__()
        self.embedding = keras.layers.Embedding(vocab_size, vocab_size)
        self.lstm1 = keras.layers.LSTM(
            lstm_units, return_sequences=True, return_state=True
        )
        self.lstm2 = keras.layers.LSTM(
            lstm_units, return_sequences=True, return_state=True
        )
        self.dense = keras.layers.Dense(vocab_size)

    def call(self, inputs, states=None, return_state=False, training=False):
        x = inputs
        x = self.embedding(x, training=training)
        if states is None:
            batch_size = tf.shape(inputs)[0]
            states = [
                tf.zeros((batch_size, self.lstm1.units)),
                tf.zeros((batch_size, self.lstm1.units)),
                tf.zeros((batch_size, self.lstm2.units)),
                tf.zeros((batch_size, self.lstm2.units)),
            ]
        x, h1, c1 = self.lstm1(x, initial_state=states[:2], training=training)
        x, h2, c2 = self.lstm2(x, initial_state=states[2:], training=training)
        states = [h1, c1, h2, c2]
        x = self.dense(x, training=training)

        if return_state:
            return x, states
        else:
            return x


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
