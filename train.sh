#!/usr/bin/env bash

MODEL=$1
LOG_FILE="${MODEL}.log"

rm -f "${LOG_FILE}" 2>/dev/null || true
echo "Starting Training for model: ${MODEL}"

uv run main.py --model "${MODEL}" 2>&1 | tee "${LOG_FILE}"
