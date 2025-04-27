#!/bin/bash
# Script to run the FastAPI server with or without the mock model

# Set text colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Ensure we're in the backend_ml_model directory
cd "$(dirname "$0")"

# Default values
USE_MOCK_MODEL=false
HOST="0.0.0.0"
PORT=8000
RELOAD=false

# Process command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --mock-model)
      USE_MOCK_MODEL=true
      shift
      ;;
    --port)
      PORT="$2"
      shift 2
      ;;
    --host)
      HOST="$2"
      shift 2
      ;;
    --reload)
      RELOAD=true
      shift
      ;;
    *)
      echo -e "${RED}Unknown option: $1${NC}"
      echo "Usage: $0 [--mock-model] [--port PORT] [--host HOST] [--reload]"
      exit 1
      ;;
  esac
done

# Prepare command
CMD=".venv/bin/python -c \"import sys; sys.argv = sys.argv[:1]"

if [ "$USE_MOCK_MODEL" = true ]; then
    CMD="$CMD + ['--mock-model']"
    echo -e "${YELLOW}Running with mock model (no TensorFlow required)${NC}"
else
    echo -e "${GREEN}Running with real TensorFlow model${NC}"
fi

# Convert bash boolean to Python boolean
RELOAD_PY="False"
if [ "$RELOAD" = true ]; then
    RELOAD_PY="True"
fi

CMD="$CMD; import uvicorn; uvicorn.run('src.server:app', host='$HOST', port=$PORT, reload=$RELOAD_PY)\""

# Run the server
echo -e "${GREEN}Starting server on $HOST:$PORT...${NC}"
eval $CMD 