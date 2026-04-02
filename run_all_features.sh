#!/bin/bash
# Run SAE feature extraction across all 3 datasets for the MiniLM 128x4 model.
# Swaps ACTIVE_DATASET in config.py between runs.
set -e

MODAL="./venv/bin/modal"
DATASETS=("fineweb-edu-10BT" "redpajama-10B" "pile-uncopyrighted")

for ds in "${DATASETS[@]}"; do
    echo "============================================================"
    echo "Processing dataset: $ds"
    echo "============================================================"

    # Swap ACTIVE_DATASET
    sed -i '' "s/^ACTIVE_DATASET = .*/ACTIVE_DATASET = \"$ds\"/" config.py
    grep "^ACTIVE_DATASET" config.py

    # Run features
    $MODAL run features.py

    echo ""
done

echo "All feature extraction complete!"
