"""Handles Akida conversion and evaluation"""

import os
import tf_init
from cnn2snn import convert, check_model_compatibility
import tensorflow as tf
from utils import load_datasets, preprocess_for_quantization, evaluate_model
from visualization import plot_confusion_matrix, plot_sample_prediction
from config import *

import numpy as np

def convert_to_akida(model_path):
   
    # Load and preprocess test data (train excluded for brevity)
    print("Loading test data...")
    data_dir = DATA_DIR
    y_test = np.load(os.path.join(data_dir, 'y_test2.npy'))
    X_test = np.load(os.path.join(data_dir, 'X_test2.npy'))
    X_test_q = preprocess_for_quantization(X_test)
    X_test_batched = X_test.astype('uint8').reshape(-1, 128, 128, 1)
    
    # Convert one-hot encoded labels to integers
    y_test = np.argmax(y_test, axis=1)
    
    # Show dataset info (shape, type)
    print("X_test_q shape:", X_test_q.shape)
    print("X_test_batched shape:", X_test_batched.shape)
    print("X_test_batched type:", X_test_batched.dtype)
    print("y_test shape:", y_test.shape)
    
    
    """Converts quantized model to Akida format and evaluates it"""
    print("Loading model...")
    model = tf.keras.models.load_model(model_path)
    
    
    # Convert to Akida
    print("Converting model to Akida format...")
    model_akida = convert(model)
    model_akida.summary()
    
    
    # Evaluate
    accuracy = model_akida.evaluate(X_test_batched, y_test)
    print('Test accuracy after Akida conversion:', accuracy)
    
    
    # Generate predictions
    akida_preds = model_akida.predict(X_test_batched)
    akida_preds = tf.argmax(tf.squeeze(akida_preds), axis=1)
    
    
    # Get stats from function
    print("Akida model stats (from function):")
    evaluate_model(akida_preds, y_test)
    
    # Visualize results
    plot_confusion_matrix(y_test, akida_preds, "Akida Model")
    plot_sample_prediction(model_akida, X_test_batched[65], y_test[65])
    
    return model_akida

if __name__ == "__main__":
    model_path = os.path.join(MODEL_DIR, 'quantized/quantized_model.h5')
    convert_to_akida(model_path)
