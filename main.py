import os

# Suppress TensorFlow Info messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = (
    "1"  # 0: all messages, 1: filter out INFO messages, 2: filter out WARNING messages, 3: filter out ERROR messages
)
import tensorflow as tf


def main():
    print("Hello from dd2424-project!")


if __name__ == "__main__":
    main()
