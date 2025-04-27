import pandas as pd
import numpy as np
import json
import os
import logging
import sys
import argparse
from sklearn.model_selection import train_test_split  # Added for stratified split

# Configure logging first
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Parse command line arguments
parser = argparse.ArgumentParser(description='Train or fine-tune the digit recognition model')
parser.add_argument('--skip-tensorflow', action='store_true', help='Skip TensorFlow dependency for development/testing')
args = parser.parse_args()

# Check for dependencies and provide helpful error messages
TENSORFLOW_AVAILABLE = False
if not args.skip_tensorflow:
    try:
        import tensorflow as tf
        from tensorflow.keras.utils import to_categorical
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping  # Added for callbacks
        TENSORFLOW_AVAILABLE = True
    except ImportError:
        logger.error("TensorFlow is not installed. For MacOS, you may need to install a special version.")
        logger.error("Try: .venv/bin/pip install tensorflow-macos tensorflow-metal")
        logger.error("Or visit: https://developer.apple.com/metal/tensorflow-plugin/")
        logger.error("You can also use --skip-tensorflow for development and testing.")
        sys.exit(1)

# Import our modules - using relative imports
try:
    from preprocessing import preprocess_sequence
    if TENSORFLOW_AVAILABLE:
        from model import build_lstm_model
except ImportError as e:
    logger.error(f"Error importing local modules: {e}")
    logger.error("Make sure you're running this script with: .venv/bin/python src/train_finetune.py")
    logger.error("Or use the provided shell script: ./finetune.sh")
    sys.exit(1)

# --- Configuration ---
DATA_FILE = "collected_data/airboard_data_2025-04-26.json" # New JSON data file
ORIGINAL_WEIGHTS_PATH = "models/lstm_digit_model.weights.h5" # Weights trained on synthetic data
FINETUNED_WEIGHTS_PATH = "models/lstm_digit_model_finetuned.weights.h5" # New weights file
FINETUNED_MODEL_PATH = "models/airboard_model_v2.keras" # Full model with .keras extension
MODELS_DIR = "models"

SEQUENCE_LEN = 50
NUM_FEATURES = 2 # Ensure this matches your data (x, y)
NUM_CLASSES = 10
INPUT_SHAPE = (SEQUENCE_LEN, NUM_FEATURES)

# Training parameters
TRAIN_FROM_SCRATCH = True  # Set to True to ignore original weights and train from scratch
TRAINING_EPOCHS = 50 if TRAIN_FROM_SCRATCH else 15  # More epochs for training from scratch
TRAINING_BATCH_SIZE = 32
TRAINING_LR = 1e-3 if TRAIN_FROM_SCRATCH else 1e-4  # Higher learning rate for training from scratch

# --- 1. Load and Prepare Data ---
logger.info(f"Loading data from {DATA_FILE}...")
if not os.path.exists(DATA_FILE):
    logger.error(f"Data file not found: {DATA_FILE}")
    logger.error(f"Make sure you have collected data in the {DATA_FILE} path.")
    logger.error("If you don't have collected data yet, you need to run the server and collect some digit samples first.")
    sys.exit(1)

all_labels = []
all_processed_paths = []

try:
    # Load JSON data file
    with open(DATA_FILE, 'r') as f:
        loaded_data = json.load(f)
    
    logger.info(f"Loaded {len(loaded_data)} samples from JSON file.")
except Exception as e:
    logger.error(f"Error loading JSON data file: {e}")
    sys.exit(1)

logger.info("Processing collected data...")
skipped_count = 0

# Iterate through the loaded JSON data
for item_index, item in enumerate(loaded_data):
    try:
        # Extract label and path data directly from JSON object
        label = int(item['label'])
        path_data = item['path']  # List of {x, y} objects

        # Convert to NumPy array
        raw_path_np = np.array([[p['x'], p['y']] for p in path_data], dtype=np.float32)

        if raw_path_np.shape[0] < 2 or raw_path_np.shape[1] != NUM_FEATURES:
             logger.warning(f"Skipping sample {item_index} due to invalid shape: {raw_path_np.shape}")
             skipped_count += 1
             continue

        # --- NOTE: Kalman Filter Smoothing Removed ---
        # Data is assumed to be pre-smoothed by the frontend application
        # Using the raw path directly for preprocessing
        path_to_process = raw_path_np

        # --- Preprocess the path ---
        processed_path = preprocess_sequence(path_to_process, sequence_length=SEQUENCE_LEN)
        if processed_path is None:
            logger.warning(f"Skipping sample {item_index} because preprocessing failed.")
            skipped_count += 1
            continue

        all_processed_paths.append(processed_path)
        all_labels.append(label)

    except KeyError as e:
        logger.warning(f"Skipping sample {item_index} due to missing key: {e}")
        skipped_count += 1
    except TypeError as e:
        logger.warning(f"Skipping sample {item_index} due to type error: {e}")
        skipped_count += 1
    except ValueError as e:
        logger.warning(f"Skipping sample {item_index} due to value error: {e}")
        skipped_count += 1
    except Exception as e:
        logger.warning(f"Skipping sample {item_index} due to unexpected error: {e}")
        skipped_count += 1

logger.info(f"Successfully processed {len(all_processed_paths)} samples.")
if skipped_count > 0:
     logger.warning(f"Skipped {skipped_count} samples due to errors or invalid data.")

if not all_processed_paths:
    logger.error("No valid data processed. Exiting.")
    sys.exit(1)

# Convert to NumPy arrays for training
X_real = np.array(all_processed_paths, dtype=np.float32)
y_real = np.array(all_labels, dtype=np.int32)

# Skip TensorFlow-dependent code if not available
if not TENSORFLOW_AVAILABLE:
    logger.info("Running in TensorFlow-less mode. Skipping model training.")
    
    # Save the processed data for later use
    os.makedirs(MODELS_DIR, exist_ok=True)
    np.save(f"{MODELS_DIR}/processed_X_data.npy", X_real)
    np.save(f"{MODELS_DIR}/processed_y_data.npy", y_real)
    
    logger.info(f"Processed data saved to {MODELS_DIR}/processed_X_data.npy and {MODELS_DIR}/processed_y_data.npy")
    logger.info("You can use this data to train the model on a system with TensorFlow installed.")
    print("\nData processing complete! The data is ready for training on a system with TensorFlow.")
    sys.exit(0)

# Implement stratified train-test split
logger.info(f"Splitting data into training and validation sets...")
try:
    # Split data (85% train, 15% validation)
    X_train, X_val, y_train, y_val = train_test_split(
        X_real, 
        y_real, 
        test_size=0.15, 
        random_state=42,  # for reproducibility
        stratify=y_real   # Ensure similar digit distribution in splits
    )
    logger.info(f"Train set: X={X_train.shape}, y={y_train.shape}")
    logger.info(f"Validation set: X={X_val.shape}, y={y_val.shape}")
except Exception as e:
    logger.error(f"Error during train/validation split: {e}. Need at least 2 samples per class for stratification.")
    # Fallback if split fails
    if len(X_real) < 10:  # Arbitrary small number for very small datasets
        logger.warning("Dataset too small for validation split, using all data for training.")
        X_train, X_val, y_train, y_val = X_real, None, y_real, None
    else:
        # Try a simple non-stratified split
        logger.warning("Falling back to non-stratified split.")
        try:
            X_train, X_val, y_train, y_val = train_test_split(X_real, y_real, test_size=0.15, random_state=42)
        except Exception as e2:
            logger.error(f"Even simple split failed: {e2}. Using all data for training.")
            X_train, X_val, y_train, y_val = X_real, None, y_real, None

# One-hot encode the splits separately
y_train_one_hot = to_categorical(y_train, num_classes=NUM_CLASSES)
y_val_one_hot = to_categorical(y_val, num_classes=NUM_CLASSES) if y_val is not None else None

logger.info(f"Final training data shapes: X={X_train.shape}, y={y_train_one_hot.shape}")
if y_val is not None:
    logger.info(f"Final validation data shapes: X={X_val.shape}, y={y_val_one_hot.shape}")

# --- 2. Build and Train Model ---
logger.info("Building model structure...")
model = build_lstm_model(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES)

# Try to load pre-trained weights if we're not training from scratch
if not TRAIN_FROM_SCRATCH:
    if os.path.exists(ORIGINAL_WEIGHTS_PATH):
        logger.info(f"Attempting to load pre-trained weights from {ORIGINAL_WEIGHTS_PATH}...")
        try:
            model.load_weights(ORIGINAL_WEIGHTS_PATH)
            logger.info("Pre-trained weights loaded successfully.")
        except Exception as e:
            logger.warning(f"Failed to load weights: {e}")
            logger.warning("This may be due to a model architecture change.")
            logger.warning("Switching to training from scratch.")
            TRAIN_FROM_SCRATCH = True
            TRAINING_EPOCHS = 50
            TRAINING_LR = 1e-3
    else:
        logger.warning(f"Original weights file not found: {ORIGINAL_WEIGHTS_PATH}")
        logger.warning("Switching to training from scratch.")
        TRAIN_FROM_SCRATCH = True
        TRAINING_EPOCHS = 50
        TRAINING_LR = 1e-3

# --- 3. Configure Training ---
training_mode = "from scratch" if TRAIN_FROM_SCRATCH else "fine-tuning"
logger.info(f"Compiling model for {training_mode} with learning rate {TRAINING_LR}...")
optimizer = Adam(learning_rate=TRAINING_LR)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Define callbacks for improving training
os.makedirs(MODELS_DIR, exist_ok=True)  # Ensure models directory exists

# Save the best model based on validation accuracy
model_checkpoint_callback = ModelCheckpoint(
    filepath=FINETUNED_MODEL_PATH,  # Save full model in .keras format
    save_weights_only=False,        # Save the full model structure + weights
    monitor='val_accuracy',         # Monitor validation accuracy
    mode='max',                     # Mode should be 'max' for accuracy
    save_best_only=True,            # Only save when validation accuracy improves
    verbose=1
)

# Stop training early if validation loss doesn't improve
early_stopping_callback = EarlyStopping(
    monitor='val_loss',           # Monitor validation loss
    patience=10,                  # Number of epochs with no improvement after which training will stop
    verbose=1,
    mode='min',                   # Mode should be 'min' for loss
    restore_best_weights=True     # Restore model weights from the epoch with the best val_loss
)

callbacks_list = [model_checkpoint_callback, early_stopping_callback]

logger.info(f"Starting {training_mode} for {TRAINING_EPOCHS} epochs...")
try:
    history = model.fit(
        X_train,
        y_train_one_hot,
        epochs=TRAINING_EPOCHS,
        batch_size=TRAINING_BATCH_SIZE,
        validation_data=(X_val, y_val_one_hot) if X_val is not None else None,
        shuffle=True,
        callbacks=callbacks_list,  # Add callbacks
        verbose=1
    )
    logger.info(f"{training_mode.capitalize()} complete.")
except Exception as e:
    logger.error(f"Error during model training: {e}")
    sys.exit(1)

# --- 4. Save Trained Weights ---
# Still saving weights for backward compatibility
os.makedirs(MODELS_DIR, exist_ok=True)
logger.info(f"Saving trained weights to {FINETUNED_WEIGHTS_PATH}...")
try:
    model.save_weights(FINETUNED_WEIGHTS_PATH)
    logger.info("Trained weights saved successfully.")
except Exception as e:
    logger.error(f"Error saving trained weights: {e}")

# --- 5. Final Model Summary ---
# Note: ModelCheckpoint has already saved the best model during training
# We're not calling model.save() again to avoid overwriting the best model with the last model

logger.info("Training script finished successfully.")
print(f"\nTraining complete! The model weights are saved at: {FINETUNED_WEIGHTS_PATH}")
print(f"The best model has been saved at: {FINETUNED_MODEL_PATH} by the ModelCheckpoint callback")
print("You can now use the server with the complete model for improved loading.") 