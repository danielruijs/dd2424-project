import os

# Suppress TensorFlow Info messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = (
    "3"  # 0: all messages, 1: filter out INFO messages, 2: filter out WARNING messages, 3: filter out ERROR messages
)
import tensorflow as tf
from keras import mixed_precision
from tensorboard.plugins.hparams import api as hp
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from main import load_text, create_dataset, train, percentage_ngrams
from transformer_model import TransformerOneStep
from models import OneStep

mixed_precision.set_global_policy('float32')

# HP_HIDDEN_UNITS = hp.HParam("hidden_units", hp.Discrete([256, 512, 768, 1024]))
# HP_LR = hp.HParam("learning_rate", hp.Discrete([0.01, 0.001, 0.0001]))
# HP_BATCH_SIZE = hp.HParam("batch_size", hp.Discrete([32, 64, 128]))
# HP_MODEL = hp.HParam("model", hp.Discrete(["rnn", "lstm", "lstm2"]))

HP_NUM_LAYERS = hp.HParam("num_layers", hp.Discrete([1, 2]))
HP_D_MODEL = hp.HParam("d_model", hp.Discrete([256, 512]))
HP_NUM_HEADS = hp.HParam("num_heads", hp.Discrete([4, 8, 12]))
HP_SEQ_LEN = hp.HParam("seq_len", hp.Discrete([100, 200, 300]))

METRIC_VAL_LOSS = "val_loss"
METRIC_TRAIN_LOSS = "train_loss"


def run(
    run_dir,
    hparams,
    train_dataset,
    val_dataset,
    vocab_size,
    ids_from_chars,
    chars_from_ids,
    ngrams,
):
    model, _, _, best_checkpoint_filepath = train(
        "transformer",
        train_dataset,
        val_dataset,
        vocab_size,
        ids_from_chars,
        chars_from_ids,
        ngrams,
        seq_length=hparams[HP_SEQ_LEN],
        hyperparameter_tuning=True,
        num_layers=hparams[HP_NUM_LAYERS],
        d_model=hparams[HP_D_MODEL],
        dff=hparams[HP_D_MODEL] * 4,
        num_heads=hparams[HP_NUM_HEADS],
    )

    print(f"\nLoading best weights from: {best_checkpoint_filepath}")
    try:
        model.load_weights(best_checkpoint_filepath)
        print("Best weights loaded successfully.")
    except Exception as e:
        print(
            f"Error loading weights: {e}. Proceeding with the final weights from training."
        )

    train_loss = model.evaluate(train_dataset, verbose=2)
    print(f"Train loss: {train_loss}")
    val_loss = model.evaluate(val_dataset, verbose=2)
    print(f"Validation loss: {val_loss}")

    result_text = "."
    one_step_model = TransformerOneStep(model, chars_from_ids, ids_from_chars)

    for _ in range(1000):
        generated_seq = tf.constant([result_text])
        if len(result_text) > hparams[HP_SEQ_LEN]:
            input_text = result_text[-hparams[HP_SEQ_LEN]:]
            input_seq = tf.constant([input_text])
        else:
            input_seq = generated_seq
        next_char = one_step_model.generate_one_step(input_seq)
        next_char_str = next_char[0].numpy().decode("utf-8")
        result_text += next_char_str

    percentage_ngrams_final = percentage_ngrams(result_text, ngrams)
    print("Percentage of n-grams in generated text:")
    print(percentage_ngrams_final)

    with tf.summary.create_file_writer(run_dir).as_default():
        hp.hparams(hparams)  # record the values used in this trial
        tf.summary.scalar(METRIC_VAL_LOSS, val_loss, step=0)
        tf.summary.scalar(METRIC_TRAIN_LOSS, train_loss, step=0)
        tf.summary.text("Final Generated Text", result_text, step=0)
        for ngram_size, pct in percentage_ngrams_final.items():
            tf.summary.scalar(
                f"{ngram_size}-gram %",
                pct,
                step=0,
            )
    print("-" * 50)


def main():
    text, _, ngrams = load_text("shubhammaindola/harry-potter-books", verbose=False)
    metrics = [hp.Metric(f"{ngram_size}-gram %") for ngram_size in ngrams.keys()]
    metrics.append(hp.Metric(METRIC_TRAIN_LOSS, display_name="train_loss"))
    metrics.append(hp.Metric(METRIC_VAL_LOSS, display_name="val_loss"))

    with tf.summary.create_file_writer("logs/hparam_tuning_transformer").as_default():
        hp.hparams_config(
            hparams=[HP_NUM_LAYERS, HP_D_MODEL, HP_NUM_HEADS, HP_SEQ_LEN,],
            metrics=metrics,
        )

    session_num = 0


    for seq_len in HP_SEQ_LEN.domain.values:    
        (
            train_dataset,
            val_dataset,
            _,
            vocab_size,
            ids_from_chars,
            chars_from_ids,
        ) = create_dataset(text, batch_size=32, seq_length=seq_len, verbose=False)
        for num_layers in HP_NUM_LAYERS.domain.values:
            for d_model in HP_D_MODEL.domain.values:
                for num_heads in HP_NUM_HEADS.domain.values:
                    hparams = {
                        HP_NUM_LAYERS: num_layers,
                        HP_D_MODEL: d_model,
                        HP_NUM_HEADS: num_heads,
                        HP_SEQ_LEN: seq_len,
                    }
                    run_name = "run-%d" % session_num
                    print("--- Starting trial: %s" % run_name)
                    print({h.name: hparams[h] for h in hparams})
                    run(
                        "logs/hparam_tuning_transformer/" + run_name,
                        hparams,
                        train_dataset,
                        val_dataset,
                        vocab_size,
                        ids_from_chars,
                        chars_from_ids,
                        ngrams,
                    )
                    session_num += 1


if __name__ == "__main__":
    main()
