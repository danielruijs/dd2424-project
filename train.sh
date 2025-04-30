#!/usr/bin/env bash

MODEL=$1
LOG_FILE="${MODEL}.log"

rm -f "${LOG_FILE}" 2>/dev/null || true
echo "Starting Training for model: ${MODEL}"

# Start TensorBoard with nohup
# nohup tensorboard --logdir logs/fit/ --port 8080 >/dev/null 2>&1 &
# http://localhost:8080/

# Start training with nohup
nohup python main.py --model "${MODEL}" > "${LOG_FILE}" 2>&1 &
tail -f "${LOG_FILE}"