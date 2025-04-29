import tensorflow as tf
from tensorflow import keras


class BaseRNN(keras.Model):
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


class LSTM(keras.Model):
    def __init__(self, vocab_size, embedding_dim, lstm_units):
        super().__init__()
        self.embedding = keras.layers.Embedding(vocab_size, embedding_dim)
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
    def __init__(self, vocab_size, embedding_dim, lstm_units):
        super().__init__()
        self.embedding = keras.layers.Embedding(vocab_size, embedding_dim)
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
