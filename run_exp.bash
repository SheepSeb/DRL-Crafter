#!/bin/bash

# Define the log directory and training script
LOGDIR="./logdir"
TRAIN_SCRIPT="./train_script.sh"
PARAMS="$@"

# Clean the log directory
if [ -d "$LOGDIR" ]; then
    rm -rf "$LOGDIR"/*
else
    mkdir -p "$LOGDIR"
fi

# Run the training script with the provided parameters
bash "$TRAIN_SCRIPT" $PARAMS