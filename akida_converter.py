"""Handles Akida conversion and evaluation"""

import tf_init
from cnn2snn import convert, check_model_compatibility
import tensorflow as tf
from utils import load_datasets, preprocess_for_quantization, evaluate_model
from visualization import plot_confusion_matrix, plot_sample_prediction
from config import *

def convert_to_akida(model_path):
    """Converts quantized model to Akida format and evaluates it"""
    # Load quantized model with compatibility options
    try:
        # First attempt: standard loading
        model = tf.keras.models.load_model(model_path)
    except TypeError as e:
        if "fn" in str(e):
            print("Handling compatibility issue with model loading...")
            # Second attempt: with custom_objects to handle compatibility issues
            model = tf.keras.models.load_model(
                model_path, 
                compile=False,
                custom_objects={}
            )
        else:
            # If it's a different TypeError, re-raise it
            raise
    
    # Check compatibility
    print("Model compatible for Akida conversion:", check_model_compatibility(model))
    
    # Convert to Akida
    model_akida = convert(model)
    
    # Load and preprocess test data
    _, _, _, _, X_test, y_test = load_datasets(DATA_DIR)
    X_test_q = preprocess_for_quantization(X_test)
    X_test_batched = X_test_q.astype('uint8').reshape(-1, 128, 128, 1)
    
    # Show dataset info (shape, type)
    print("X_test_q shape:", X_test_q.shape)
    print("X_test_batched shape:", X_test_batched.shape)
    print("X_test_batched type:", X_test_batched.dtype)
    
    # Evaluate
    accuracy = model_akida.evaluate(X_test_batched, y_test)
    print('Test accuracy after Akida conversion:', accuracy)
    
    
    # Generate predictions
    akida_preds = model_akida.predict(X_test_batched)
    akida_preds = tf.argmax(akida_preds, axis=1)
    
    
    # Get stats from function
    print("Akida model stats (from function):")
    evaluate_model(akida_preds, y_test)
    
    # Visualize results
    plot_confusion_matrix(y_test, akida_preds, "Akida Model")
    plot_sample_prediction(model_akida, X_test_q[0], y_test[0])
    
    return model_akida

if __name__ == "__main__":
    model_path = os.path.join(MODEL_DIR, 'quantized/quantized_model.h5')
    convert_to_akida(model_path)
