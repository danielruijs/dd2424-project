import os

# Suppress TensorFlow Info messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = (
    "1"  # 0: all messages, 1: filter out INFO messages, 2: filter out WARNING messages, 3: filter out ERROR messages
)
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
from main import (
    load_text,
    create_dataset,
    train,
    OneStep,
)

HP_HIDDEN_UNITS = hp.HParam('hidden_units', hp.Discrete([512, 1024, 2048]))
HP_LR = hp.HParam('learning_rate', hp.Discrete([0.001, 0.0001, 0.01]))
HP_BATCH_SIZE = hp.HParam('batch_size', hp.Discrete([32, 64, 128]))
HP_MODEL = hp.HParam('model', hp.Discrete(['rnn', 'lstm', 'lstm2']))

METRIC_VAL_LOSS = 'val_loss'

def run(run_dir, hparams, train_dataset, val_dataset, vocab_size, ids_from_chars, chars_from_ids):
    model, _, _ = train(
        hparams[HP_MODEL],
        train_dataset,
        val_dataset,
        vocab_size,
        ids_from_chars,
        chars_from_ids,
        hyperparameter_tuning=True
    )

    val_loss = model.evaluate(val_dataset)

    one_step_model = OneStep(model, chars_from_ids, ids_from_chars)
    states = None
    next_char = tf.constant(["."])
    result = [next_char]

    for _ in range(1000):
        next_char, states = one_step_model.generate_one_step(next_char, states=states)
        result.append(next_char)

    result = tf.strings.join(result)
    result_text = result[0].numpy().decode("utf-8")

    with tf.summary.create_file_writer(run_dir).as_default():
        hp.hparams(hparams)  # record the values used in this trial
        tf.summary.scalar(METRIC_VAL_LOSS, val_loss, step=0)
        tf.summary.text("Final Generated Text", result_text, step=0)


def main():
    with tf.summary.create_file_writer('logs/hparam_tuning').as_default():
        hp.hparams_config(
            hparams=[HP_MODEL, HP_HIDDEN_UNITS, HP_LR, HP_BATCH_SIZE],
            metrics=[hp.Metric(METRIC_VAL_LOSS, display_name='val_loss')],
        )

    text = load_text("shubhammaindola/harry-potter-books")
    (
        train_dataset,
        val_dataset,
        _,
        vocab_size,
        ids_from_chars,
        chars_from_ids,
    ) = create_dataset(text)

    session_num = 0

    for model_name in HP_MODEL.domain.values:
        for hidden_units in HP_HIDDEN_UNITS.domain.values:
            for learning_rate in HP_LR.domain.values:
                for batch_size in HP_BATCH_SIZE.domain.values:
                    hparams = {
                        HP_MODEL: model_name,
                        HP_HIDDEN_UNITS: hidden_units,
                        HP_LR: learning_rate,
                        HP_BATCH_SIZE: batch_size,
                    }
                    run_name = "run-%d" % session_num
                    print('--- Starting trial: %s' % run_name)
                    print({h.name: hparams[h] for h in hparams})
                    run('logs/hparam_tuning/' + run_name, hparams, train_dataset, val_dataset, vocab_size, ids_from_chars, chars_from_ids)
                    session_num += 1

    

if __name__ == "__main__":
    main()
