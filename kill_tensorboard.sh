#!/usr/bin/env bash

kill $(ps -e | grep 'tensorboard' | awk '{print $1}')
echo "Killed TensorBoard process"