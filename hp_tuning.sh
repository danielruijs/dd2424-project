#!/usr/bin/env bash

LOG_FILE="hp_tuning.log"

rm -f "${LOG_FILE}" 2>/dev/null || true
echo "Starting Hyperparameter tuning"

# Start training with nohup
nohup python hp_tuning.py > "${LOG_FILE}" 2>&1 &
tail -f "${LOG_FILE}"