"""Configuration settings shared across modules"""

import os

RANDOM_SEED = 42

# Data paths
DATA_DIR = os.path.join(os.getcwd(), 'datasets')
MODEL_DIR = os.path.join(os.getcwd(), 'models')

# CNN Model parameters
INPUT_SHAPE = (128, 128, 1)
NUM_CLASSES = 5
BATCH_SIZE = 32
EPOCHS = 5

# Training parameters
LEARNING_RATE = 1e-3

# Quantization parameters
QUANT_PARAMS = {
    'input_weight_bits': 8,
    'weight_bits': 8,
    'activation_bits': 8,
    'per_tensor_activations': True
}
Q_EPOCHS = 2
Q_BATCH_SIZE = 100

# Create necessary directories if not found
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
