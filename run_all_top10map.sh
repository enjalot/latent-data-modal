#!/bin/bash
# Run top10map across all 3 datasets for the MiniLM 128x4 model.
set -e

MODAL="./venv/bin/modal"
DATASETS=("fineweb-edu-10BT" "redpajama-10B" "pile-uncopyrighted")

for ds in "${DATASETS[@]}"; do
    echo "============================================================"
    echo "top10map: $ds"
    echo "============================================================"

    sed -i '' "s/^ACTIVE_DATASET = .*/ACTIVE_DATASET = \"$ds\"/" config.py
    grep "^ACTIVE_DATASET" config.py

    $MODAL run top10map.py

    echo ""
done

echo "All top10map complete!"
