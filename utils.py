"""
Utility Functions

Common functions used across different scripts.
"""

import gc
import os
import random
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

# TensorFlow import must come after tf_init
import tf_init
import tensorflow as tf

def set_random_seeds(seed):
    """Sets random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def clear_memory():
    """Clears TensorFlow session and garbage collector"""
    gc.collect()
    tf.keras.backend.clear_session()

def preprocess_for_quantization(data):
    """
    Scales data to [0, 255] range for 8-bit quantization
    
    Args:
        data: Input data to be scaled
    
    Returns:
        numpy.ndarray: Scaled data as uint8
    """
    data_scaled = ((data - data.min()) / (data.max() - data.min()) * 255).astype('uint8')
    return data_scaled

def load_datasets(data_dir):
    """
    Loads training, validation and test datasets
    
    Args:
        data_dir: Directory containing the dataset files
    
    Returns:
        tuple: (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    
    X_train = np.load(os.path.join(data_dir, 'X_train2.npy'))
    y_train = np.load(os.path.join(data_dir, 'y_train2.npy'))
    X_val = np.load(os.path.join(data_dir, 'X_val2.npy'))
    y_val = np.load(os.path.join(data_dir, 'y_val2.npy'))
    X_test = np.load(os.path.join(data_dir, 'X_test2.npy'))
    y_test = np.load(os.path.join(data_dir, 'y_test2.npy'))
    
    # Convert one-hot encoded labels to integers
    y_train = np.argmax(y_train, axis=1)
    y_val = np.argmax(y_val, axis=1)
    y_test = np.argmax(y_test, axis=1)
    
    return X_train, y_train, X_val, y_val, X_test, y_test

def evaluate_model(predictions, y_test):
    """
    Evaluates a model's performance by calculating accuracy and F1 score metrics.

    Args:
        predictions : array-like
            The predicted labels from the model
        y_test : array-like 
            The true labels from the test set

    Returns:
        tuple
            A tuple containing:
            - accuracy (float): The accuracy score of the model
            - f1 (float): The weighted F1 score of the model
    """
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average='weighted')
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    return accuracy, f1
