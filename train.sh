#!/bin/bash

# Ensure the script exits on any command failure
set -e

# Define paths to data, configurations, and results folder
DATA_CSV="../data/tabular/coronal-REBLs.csv"  # Update with the correct path
AUGMENTATION_CSV="../data/tabular/axial-REBLs.csv"  # Update with the correct path
CONFIG_YAML="./params/config.yaml"  # Update with the correct config file path
TARGET_FOLDER="./example_training"

# Merge augmentation data into primary data (if required preprocessing is done beforehand)
# Uncomment and modify the following block if merging requires an additional step:
# python preprocess.py --data_primary "$DATA_CSV" --data_aug "$AUGMENTATION_CSV" --output_combined "path/to/combined.csv"

# Run training with optional feature importance calculation
python run.py --data "$DATA_CSV" --augmentation "$AUGMENTATION_CSV" --target_folder "$TARGET_FOLDER" --config_path "$CONFIG_YAML" train

# Notify the user of successful training
echo "Training completed. Results saved in $TARGET_FOLDER"
