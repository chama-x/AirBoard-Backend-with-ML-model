# AirBoard: Air-Written Digit Recognition

An interactive web application that recognizes digits drawn in the air using machine learning.

## Project Overview

AirBoard uses computer vision and deep learning to recognize digits drawn in the air. The system consists of:

- **Frontend**: A web-based interface that captures hand movements
- **Backend ML Model**: A TensorFlow LSTM model that processes gesture data and predicts digits

## Features

- Real-time digit recognition from motion data
- Data collection mode for improving the model
- Fine-tuning capabilities to adapt the model to new users
- WebSocket-based communication for low-latency predictions

## Backend ML Model

The backend uses a TensorFlow LSTM model to analyze hand movement trajectories and predict which digit was drawn.

### Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd AirBoard
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the server**:
   ```bash
   ./run_server.sh  # or: python -m src.server
   ```
   
   Options:
   - `--mock-model`: Run without TensorFlow (for testing)
   - `--port PORT`: Specify custom port (default: 8000)
   - `--host HOST`: Specify host (default: 0.0.0.0)
   - `--reload`: Enable auto-reload (development)

### Training the Model

1. **Collect training data** using the web interface by drawing digits when prompted.

2. **Fine-tune the model with your data**:
   ```bash
   ./finetune.sh
   ```

3. The new model will be saved in the `models/` directory.

## API Endpoints

- `GET /`: Health check
- `GET /model-status`: Check model loading status
- `GET /next_digit_prompt`: Get suggestion for next digit to draw (for data collection)
- `WebSocket /ws`: Real-time communication for drawing and predictions

## Model Architecture

The digit recognition model uses an LSTM (Long Short-Term Memory) neural network to process sequences of 2D coordinates representing hand movements through time.

## Development

### Project Structure
```
backend_ml_model/
├── src/
│   ├── server.py           # FastAPI server
│   ├── model.py            # Model architecture
│   ├── preprocessing.py    # Data preprocessing
│   └── train_finetune.py   # Training script
├── models/                 # Saved models
├── collected_data/         # Training data
├── requirements.txt        # Dependencies
├── run_server.sh           # Server startup script
└── finetune.sh             # Model training script
```

### Requirements

- Python 3.8+
- TensorFlow 2.x (or TensorFlow-metal for M1/M2 Macs)
- FastAPI
- NumPy
- Other dependencies in `requirements.txt`

## License

[Specify license here]

## Acknowledgments

[Add any acknowledgments, references, or credits] 