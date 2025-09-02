#!/bin/bash
echo "Checking training status..."
if ps aux | grep -v grep | grep "train_ibert.py" > /dev/null; then
    echo "✓ Training still running"
    echo "Latest output:"
    tail -5 training_final.log
else
    echo "Training completed or stopped"
    echo "Check training_final.log for details"
fi

# Check if model directory was created
if [ -d "models/ibert_devstral_v1" ]; then
    echo "✓ Model directory created"
    ls -la models/ibert_devstral_v1/
fi