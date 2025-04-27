import json
import numpy as np
import logging
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any, Optional
import uvicorn
import csv
import os
import pandas as pd
import random
import argparse
import sys

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add command line argument parsing for no-model mode
parser = argparse.ArgumentParser(description='FastAPI server for air-written digit recognition')
parser.add_argument('--mock-model', action='store_true', help='Run with a mock model for testing without TensorFlow')
try:
    args, _ = parser.parse_known_args()
except SystemExit:
    # Prevent argparse from exiting when used with uvicorn
    args = argparse.Namespace(mock_model=False)

# Check for TensorFlow availability
TENSORFLOW_AVAILABLE = False
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
    logger.info("TensorFlow successfully imported.")
except ImportError:
    logger.warning("TensorFlow not available. Will use mock model if --mock-model flag is provided.")
    if not args.mock_model:
        logger.error("TensorFlow is required but not installed. Run with --mock-model for testing or install TensorFlow.")
        logger.error("For Python ≤3.10: pip install tensorflow")
        logger.error("For macOS: pip install tensorflow-macos tensorflow-metal")
        if __name__ == "__main__":  # Only exit if running the script directly
            sys.exit(1)

# Import preprocessing regardless of TensorFlow availability
try:
    from src.preprocessing import preprocess_sequence
    if TENSORFLOW_AVAILABLE:
        from src.model import build_lstm_model
except ImportError as e:
    logger.error(f"Error importing required modules: {e}")
    if __name__ == "__main__":
        sys.exit(1)

# Define Constants and Global Variables for Model
MODEL_WEIGHTS_PATH = "models/lstm_digit_model_finetuned.weights.h5"  # Path relative to backend_ml_model root
FULL_MODEL_PATH = "models/airboard_model_v2.keras"  # Updated path to full model with .keras extension
SEQUENCE_LEN = 50  # Should match preprocessing and model input
NUM_FEATURES = 2   # 2 for (x,y), 3 if using z
NUM_CLASSES = 10
INPUT_SHAPE = (SEQUENCE_LEN, NUM_FEATURES)

# Data saving constants
DATA_SAVE_DIR = "collected_data"
DATA_SAVE_FILE = os.path.join(DATA_SAVE_DIR, "collected_digits.csv")

# Global variable to hold the loaded model
loaded_model = None

# Mock model class for testing without TensorFlow
class MockModel:
    """A simple mock model that returns random predictions for testing without TensorFlow."""
    
    def __init__(self):
        logger.info("Initializing mock model for testing")
        
    def predict(self, input_data, verbose=0):
        """Returns random probabilities for each class."""
        batch_size = input_data.shape[0]
        # Generate random probabilities for each class and normalize to sum to 1
        raw_probas = np.random.rand(batch_size, NUM_CLASSES)
        normalized_probas = raw_probas / np.sum(raw_probas, axis=1, keepdims=True)
        return normalized_probas

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"], # Allow your frontend origin
    # Or use ["*"] to allow all origins during development, but be specific for production
    allow_credentials=True,
    allow_methods=["*"], # Allow all methods (GET, POST, etc.)
    allow_headers=["*"], # Allow all headers
)

@app.get("/next_digit_prompt")
async def get_next_digit_prompt():
    """
    Determines which digit has the fewest collected samples and returns it.
    Prioritizes digits with fewer than TARGET_SAMPLES_PER_DIGIT samples.
    Reads data from the CSV file each time using a robust manual counting approach.
    """
    TARGET_SAMPLES_PER_DIGIT = 50  # Target number of samples to collect per digit
    
    logger.info("Received request for next digit prompt.")
    label_counts = {i: 0 for i in range(NUM_CLASSES)} # Initialize counts for digits 0-9

    if os.path.exists(DATA_SAVE_FILE):
        try:
            # Open the CSV file and read rows manually for more robust counting
            with open(DATA_SAVE_FILE, 'r', newline='') as f:
                reader = csv.reader(f)
                header = next(reader, None)  # Skip header row
                
                if header and 'label' in header:
                    label_index = header.index('label')
                else:
                    label_index = 0  # Default to first column if header is missing or doesn't contain 'label'
                
                row_count = 0
                invalid_count = 0
                
                for row in reader:
                    row_count += 1
                    if row and len(row) > label_index:
                        label_value = row[label_index]
                        try:
                            label_int = int(label_value)
                            if 0 <= label_int < NUM_CLASSES:
                                label_counts[label_int] += 1
                            else:
                                invalid_count += 1
                                logger.warning(f"Skipping out-of-range label in CSV: {label_value}")
                        except (ValueError, TypeError) as e:
                            invalid_count += 1
                            logger.warning(f"Skipping invalid label in CSV: {label_value}, Error: {e}")
                    else:
                        invalid_count += 1
                        logger.warning(f"Skipping malformed row in CSV (missing label column)")
                
                logger.info(f"Processed {row_count} rows, found {invalid_count} invalid labels")
                logger.info(f"Current label counts: {label_counts}")
        except FileNotFoundError:
            logger.warning(f"{DATA_SAVE_FILE} not found despite os.path.exists check. Race condition?")
        except pd.errors.EmptyDataError:
            logger.warning(f"{DATA_SAVE_FILE} is empty. Starting counts at 0.")
        except Exception as e:
            logger.error(f"Error reading or processing {DATA_SAVE_FILE}: {e}")
            # Return a random digit if CSV is broken/unreadable
            return {"digit_to_draw": random.randint(0, NUM_CLASSES - 1)}
    else:
        logger.warning(f"{DATA_SAVE_FILE} not found. Starting counts at 0.")

    # Find digits with counts below the target
    below_target_digits = {digit: count for digit, count in label_counts.items() if count < TARGET_SAMPLES_PER_DIGIT}
    
    if not below_target_digits:
        # All digits have reached the target count!
        logger.info(f"All digits have at least {TARGET_SAMPLES_PER_DIGIT} samples. Suggesting random digit.")
        # Optionally return a special signal, or just pick randomly from all
        chosen_digit = random.randint(0, NUM_CLASSES - 1) 
        # Or: return {"digit_to_draw": -1} # If frontend can handle -1 meaning "complete"
    else:
        # Find the minimum count *among digits that are below the target*
        min_count_below_target = min(below_target_digits.values())
        
        # Find all digits below target that have this minimum count
        digits_to_prompt = [
            digit for digit, count in below_target_digits.items() 
            if count == min_count_below_target
        ]
        
        # Randomly select one digit from this subset
        chosen_digit = random.choice(digits_to_prompt)
        logger.info(f"Digits below target ({TARGET_SAMPLES_PER_DIGIT}): {len(below_target_digits)}. Min count among them: {min_count_below_target}. Prompting for digit: {chosen_digit}")

    return {"digit_to_draw": chosen_digit}

@app.get("/model-status")
async def model_status():
    """Return the current status of the model."""
    global loaded_model
    
    # Define a variable to track how the model was loaded
    model_loading_method = "unknown"
    
    # Try to determine the model loading method
    if hasattr(loaded_model, '_is_full_model'):
        model_loading_method = "full_model"
    elif hasattr(loaded_model, '_is_weights_only'):
        model_loading_method = "weights_only"
    
    if loaded_model is None:
        return {
            "status": "not_loaded",
            "message": "Model is not loaded",
            "tensorflow_available": TENSORFLOW_AVAILABLE,
            "mock_model": args.mock_model
        }
    else:
        return {
            "status": "loaded", 
            "message": "Model is loaded and ready",
            "tensorflow_available": TENSORFLOW_AVAILABLE,
            "mock_model": args.mock_model,
            "model_type": "mock" if isinstance(loaded_model, MockModel) else "tensorflow",
            "loading_method": model_loading_method
        }

@app.on_event("startup")
async def startup_event():
    """Loads the TensorFlow model or mock model when the server starts."""
    global loaded_model
    global INPUT_SHAPE
    logger.info("Server starting up...")
    
    # Use mock model if flag provided or if TensorFlow not available and flag provided
    if args.mock_model:
        logger.warning("Using mock model as requested via --mock-model flag")
        loaded_model = MockModel()
        logger.info("Mock model initialized successfully")
    elif TENSORFLOW_AVAILABLE:
        try:
            # First try to load the full model (architecture + weights)
            logger.info(f"Attempting to load full model from: {FULL_MODEL_PATH}")
            if os.path.exists(FULL_MODEL_PATH):
                try:
                    loaded_model = tf.keras.models.load_model(FULL_MODEL_PATH)
                    # Add metadata to track loading method
                    loaded_model._is_full_model = True
                    loaded_model._is_weights_only = False
                    logger.info("Full model loaded successfully.")
                    
                    # Perform a dummy prediction to finalize model loading if needed
                    try:
                        dummy_input = np.zeros((1, *INPUT_SHAPE))
                        _ = loaded_model.predict(dummy_input, verbose=0)
                        logger.info("TensorFlow model loaded successfully and verified with dummy prediction.")
                        return  # Early return if full model loads successfully
                    except Exception as pred_error:
                        logger.error(f"Error during dummy prediction with full model: {pred_error}")
                        logger.warning("Full model loaded but prediction verification failed.")
                        loaded_model = None  # Reset to try weights-only approach
                except Exception as model_error:
                    logger.error(f"Error loading full model: {model_error}")
                    logger.warning("Falling back to loading weights-only approach...")
            else:
                logger.warning(f"Full model file not found: {FULL_MODEL_PATH}")
                logger.warning("Falling back to weights-only approach...")
            
            # Fall back to loading weights only if full model approach failed
            logger.info(f"Attempting to load model weights from: {MODEL_WEIGHTS_PATH}")
            # Verify the model weights file exists
            if not os.path.exists(MODEL_WEIGHTS_PATH):
                logger.error(f"Model weights file not found: {MODEL_WEIGHTS_PATH}")
                logger.error(f"Current working directory: {os.getcwd()}")
                logger.error("Please ensure the model weights file exists at the specified path")
                return
                
            # Build the model structure first
            # Ensure INPUT_SHAPE is correctly defined before this call
            model_structure = build_lstm_model(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES)
            
            try:
                # Try loading the weights
                model_structure.load_weights(MODEL_WEIGHTS_PATH)
                loaded_model = model_structure # Assign to global variable
                
                # Add metadata to track loading method
                loaded_model._is_full_model = False
                loaded_model._is_weights_only = True
                
                # Perform a dummy prediction to finalize model loading if needed
                try:
                    dummy_input = np.zeros((1, *INPUT_SHAPE))
                    _ = loaded_model.predict(dummy_input, verbose=0)
                    logger.info("TensorFlow model loaded successfully and verified with dummy prediction.")
                except Exception as pred_error:
                    logger.error(f"Error during dummy prediction: {pred_error}")
                    logger.warning("Model loaded but prediction verification failed.")
            except Exception as weight_error:
                error_msg = str(weight_error)
                logger.error(f"Error loading model weights: {error_msg}")
                
                # Provide specific guidance based on error type
                if "not compatible with" in error_msg:
                    logger.error("Model architecture may have changed since weights were saved.")
                    logger.error("Consider setting TRAIN_FROM_SCRATCH=True in train_finetune.py and retraining.")
                elif "No such file" in error_msg:
                    logger.error(f"File not found. Ensure {MODEL_WEIGHTS_PATH} exists.")
                elif "Permission denied" in error_msg:
                    logger.error(f"Permission denied when accessing {MODEL_WEIGHTS_PATH}.")
                
                loaded_model = None
                
                # Fall back to mock model if requested
                if args.mock_model:
                    logger.warning("Falling back to mock model due to weight loading error")
                    loaded_model = MockModel()
                    logger.info("Mock model initialized as fallback")
        except Exception as e:
            logger.error(f"Unexpected error during model loading: {e}", exc_info=True)
            loaded_model = None
            
            # Fall back to mock model if requested
            if args.mock_model:
                logger.warning("Falling back to mock model due to unexpected error")
                loaded_model = MockModel()
                logger.info("Mock model initialized as fallback")
    else:
        # TensorFlow not available
        if args.mock_model:
            logger.warning("TensorFlow not available, using mock model as requested")
            loaded_model = MockModel()
            logger.info("Mock model initialized successfully")
        else:
            logger.error("TensorFlow not available and --mock-model flag not provided.")
            logger.error("Server will run but WebSocket connections will fail.")
            loaded_model = None
    
    # Ensure data save directory exists
    logger.info(f"Ensuring data save directory exists: {DATA_SAVE_DIR}")
    os.makedirs(DATA_SAVE_DIR, exist_ok=True)
    # Add header to CSV if file doesn't exist
    if not os.path.exists(DATA_SAVE_FILE):
         try:
              with open(DATA_SAVE_FILE, 'w', newline='') as f:
                   writer = csv.writer(f)
                   writer.writerow(['label', 'path_json']) # Write header
              logger.info(f"Created data save file with header: {DATA_SAVE_FILE}")
         except Exception as e:
              logger.error(f"Failed to create data save file header: {e}")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint that receives drawing paths, performs recognition,
    and sends back the prediction.
    """
    await websocket.accept()
    client_id = f"{websocket.client.host}:{websocket.client.port}"
    logger.info(f"WebSocket connection accepted from: {client_id}")

    if loaded_model is None:
        error_msg = "Model not loaded, cannot perform predictions."
        logger.error(error_msg)
        await websocket.send_json({
            "error": error_msg,
            "tensorflow_available": TENSORFLOW_AVAILABLE,
            "solution": "Restart server with --mock-model flag for testing without TensorFlow"
        })
        await websocket.close(code=1011, reason="Model not available")
        return

    try:
        while True:
            data_str = await websocket.receive_text()
            logger.info(f"Received data string from {client_id}") # Log raw string

            try:
                # Parse the JSON string
                message_data: Dict[str, Any] = json.loads(data_str)
                
                # Check the message format based on keys
                if set(message_data.keys()) == {'path'}:
                    # This is a prediction request (only path provided)
                    logger.info(f"Received prediction request from {client_id}")
                    
                    path_data = message_data['path']
                    if not isinstance(path_data, list) or len(path_data) < 2:
                        logger.warning(f"Received invalid path data format from {client_id}.")
                        await websocket.send_json({"error": "Invalid path data format, need list of at least 2 points"})
                        continue
                    
                    # --- Continue with Prediction Logic ---
                    try:
                        # Convert path to NumPy array for processing
                        path_array = np.array([[p['x'], p['y']] for p in path_data], dtype=np.float32)
                        if path_array.shape[1] != NUM_FEATURES:
                            raise ValueError(f"Expected {NUM_FEATURES} features, got {path_array.shape[1]}")
                    except (KeyError, ValueError) as e:
                        logger.warning(f"Error converting path data to NumPy array: {e}")
                        await websocket.send_json({"error": "Invalid point format in path data"})
                        continue

                    # Process the path data
                    path_to_process = path_array
                    preprocessed_path = preprocess_sequence(path_to_process, sequence_length=SEQUENCE_LEN)
                    
                    if preprocessed_path is None:
                        logger.warning(f"Preprocessing failed for prediction request from {client_id}.")
                        await websocket.send_json({"error": "Preprocessing failed"})
                        continue
                    
                    # Prepare input for model
                    input_tensor = np.expand_dims(preprocessed_path, axis=0)
                    
                    # Make prediction
                    prediction_probs = loaded_model.predict(input_tensor, verbose=0)[0]
                    predicted_digit = int(np.argmax(prediction_probs))
                    confidence = float(np.max(prediction_probs))
                    
                    logger.info(f"Prediction for {client_id}: Digit={predicted_digit}, Confidence={confidence:.4f}")
                    await websocket.send_json({
                        "prediction": predicted_digit,
                        "confidence": round(confidence, 4)
                    })
                    logger.info(f"Sent prediction {predicted_digit} to {client_id}")
                
                elif set(message_data.keys()) >= {'path', 'label'}:
                    # This is a labeled data submission
                    logger.info(f"Received labeled data from {client_id}")
                    
                    path_data: Optional[List[Dict[str, float]]] = message_data.get('path')
                    label: Optional[int] = message_data.get('label')

                    if path_data is None or label is None or not isinstance(path_data, list) or len(path_data) < 2 or not isinstance(label, int):
                        logger.warning(f"Received invalid labeled path data format from {client_id}.")
                        await websocket.send_json({"error": "Invalid labeled path data format"})
                        continue

                    # --- Validate Label Before Saving ---
                    try:
                        label_int = int(label)
                        if not (0 <= label_int < NUM_CLASSES):
                            raise ValueError(f"Label out of range (0-{NUM_CLASSES-1})")
                    except (ValueError, TypeError) as e:
                        logger.error(f"Received invalid label type or value from WebSocket: Label='{label}', Error='{e}'. Skipping save.")
                        await websocket.send_json({"error": f"Invalid label: {e}"})
                        continue
                    
                    # --- Save the Received Labeled Data with Validated Label ---
                    try:
                        with open(DATA_SAVE_FILE, 'a', newline='') as f:
                            writer = csv.writer(f)
                            # Save validated label_int and the path data as a JSON string
                            writer.writerow([label_int, json.dumps(path_data)])
                        logger.info(f"Saved data for label: {label_int}, path length: {len(path_data)}")
                    except Exception as e:
                        logger.error(f"Failed to save data: {e}")
                        await websocket.send_json({"error": f"Failed to save data: {e}"})
                        continue  # Skip prediction if we couldn't save the data
                    
                    # Also run prediction on the saved data to give feedback
                    try:
                        # Convert path to NumPy array for processing
                        path_array = np.array([[p['x'], p['y']] for p in path_data], dtype=np.float32)
                        if path_array.shape[1] != NUM_FEATURES:
                            raise ValueError(f"Expected {NUM_FEATURES} features, got {path_array.shape[1]}")
                    except (KeyError, ValueError) as e:
                        logger.warning(f"Error converting path data to NumPy array: {e}")
                        await websocket.send_json({"error": "Invalid point format in path data"})
                        continue

                    # Use the frontend-smoothed path data directly
                    path_to_process = path_array
                    preprocessed_path = preprocess_sequence(path_to_process, sequence_length=SEQUENCE_LEN)

                    if preprocessed_path is None:
                        logger.warning("Preprocessing failed for prediction.")
                        await websocket.send_json({"prediction": "?", "error": "Preprocessing failed"})
                        continue

                    input_tensor = np.expand_dims(preprocessed_path, axis=0)
                    prediction_probs = loaded_model.predict(input_tensor, verbose=0)[0]
                    predicted_digit = int(np.argmax(prediction_probs))
                    confidence = float(np.max(prediction_probs))

                    logger.info(f"Prediction for {client_id}: Digit={predicted_digit}, Confidence={confidence:.4f}")
                    await websocket.send_json({
                        "prediction": predicted_digit,
                        "confidence": round(confidence, 4),
                        "saved": True,
                        "label": label_int
                    })
                    logger.info(f"Sent prediction {predicted_digit} and save confirmation to {client_id}")
                
                else:
                    # Unrecognized message format
                    logger.warning(f"Received unrecognized message format from {client_id}: keys={message_data.keys()}")
                    await websocket.send_json({"error": "Invalid data format received. Expected either 'path' for prediction or both 'path' and 'label' for saving data."})

            except json.JSONDecodeError:
                logger.warning(f"Received non-JSON message from {client_id}: {data_str[:100]}...") # Log truncated
                await websocket.send_json({"error": "Invalid JSON received"})
            except Exception as e:
                logger.error(f"Error processing message from {client_id}: {e}", exc_info=True)
                # Attempt to inform client before breaking loop/disconnecting
                try:
                    await websocket.send_json({"error": "Internal server error during processing"})
                except Exception:
                    pass # Ignore if sending error fails
                break # Exit loop on processing error

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected from: {client_id}")
    except Exception as e:
        # Catch errors during initial connection accept or loop setup
        logger.error(f"WebSocket error with {client_id}: {e}", exc_info=True)
        # Ensure connection is closed if possible
        try:
             await websocket.close(code=1011)
        except Exception:
             pass

# Basic root endpoint for testing if server is running
@app.get("/")
async def read_root():
    return {
        "message": "Air Writer Backend is running",
        "tensorflow_available": TENSORFLOW_AVAILABLE,
        "mock_model": args.mock_model if hasattr(args, 'mock_model') else False,
        "model_loaded": loaded_model is not None
    }

# Note: Running the server requires Uvicorn.
# From the 'backend_ml_model' directory root (with venv active):
# PYTHONPATH=. uvicorn src.server:app --reload --host 0.0.0.0 --port 8000
# To use mock model without TensorFlow:
# PYTHONPATH=. uvicorn src.server:app --reload --host 0.0.0.0 --port 8000 --mock-model 