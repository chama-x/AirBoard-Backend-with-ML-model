#!/bin/bash
# Script to run the fine-tuning script using the virtual environment's Python

# Set text colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Enable this flag to skip TensorFlow check (when you already have TensorFlow installed)
SKIP_TF_CHECK=true

# Ensure we're in the backend_ml_model directory
cd "$(dirname "$0")"

echo -e "${GREEN}=== Air Writer Model Fine-tuning Script ===${NC}"
echo "This script will fine-tune the digit recognition model with your collected data."
echo

# Check if the virtual environment exists
if [ ! -d ".venv" ]; then
    echo -e "${RED}Error: Virtual environment (.venv) not found.${NC}"
    echo "Please create it first with: python -m venv .venv"
    exit 1
fi

# Check if collected data exists
if [ ! -f "collected_data/collected_digits.csv" ]; then
    echo -e "${YELLOW}Warning: No collected data found at collected_data/collected_digits.csv${NC}"
    echo "You need to collect data first by running the server and drawing digits."
    echo "Continue? (y/n)"
    read answer
    if [ "$answer" != "y" ]; then
        exit 1
    fi
fi

# Check if we have a pre-trained model
if [ ! -f "models/lstm_digit_model.weights.h5" ]; then
    echo -e "${YELLOW}Warning: No pre-trained model found at models/lstm_digit_model.weights.h5${NC}"
    echo "You need a pre-trained model for fine-tuning."
    echo "Create the models directory? (y/n)"
    read answer
    if [ "$answer" == "y" ]; then
        mkdir -p models
    else
        exit 1
    fi
fi

# Check dependencies
echo -e "${GREEN}Checking dependencies...${NC}"

# Check TensorFlow only if SKIP_TF_CHECK is false
if [ "$SKIP_TF_CHECK" != "true" ] && ! .venv/bin/pip list | grep -i tensorflow > /dev/null; then
    echo -e "${YELLOW}TensorFlow not found. Attempting to install TensorFlow for macOS...${NC}"
    .venv/bin/pip install tensorflow-macos tensorflow-metal || {
        echo -e "${RED}Failed to install TensorFlow.${NC}"
        echo "You may need to manually install TensorFlow for your specific system."
        echo "For macOS: visit https://developer.apple.com/metal/tensorflow-plugin/"
        echo "Continue without installing? (y/n)"
        read answer
        if [ "$answer" != "y" ]; then
            exit 1
        fi
    }
else
    if [ "$SKIP_TF_CHECK" == "true" ]; then
        echo -e "${GREEN}Skipping TensorFlow check as requested.${NC}"
    else
        echo -e "${GREEN}TensorFlow is already installed.${NC}"
    fi
fi

# Check filterpy
if ! .venv/bin/pip list | grep -i filterpy > /dev/null; then
    echo -e "${YELLOW}FilterPy not found. Installing...${NC}"
    .venv/bin/pip install filterpy || {
        echo -e "${RED}Failed to install FilterPy.${NC}"
        exit 1
    }
fi

# Check pandas
if ! .venv/bin/pip list | grep -i pandas > /dev/null; then
    echo -e "${YELLOW}Pandas not found. Installing...${NC}"
    .venv/bin/pip install pandas || {
        echo -e "${RED}Failed to install Pandas.${NC}"
        exit 1
    }
fi

echo -e "${GREEN}Running data processing script...${NC}"
echo "This will process your collected data and prepare it for training."
echo

# Run the training script using the virtual environment's Python with skip-tensorflow flag
.venv/bin/python src/train_finetune.py --skip-tensorflow
result=$?

if [ $result -eq 0 ]; then
    echo -e "${GREEN}Data processing completed successfully!${NC}"
    echo "Your data is now ready for training on a system with TensorFlow support."
    echo -e "${YELLOW}Note: The model wasn't trained because TensorFlow is not available.${NC}"
    echo "The processed data has been saved and can be used for training later."
else
    echo -e "${RED}Data processing failed with error code $result.${NC}"
    echo "Check the error messages above for details."
fi

# Exit with the Python script's exit code
exit $result 