"""Handles model quantization and fine-tuning"""

import os
import tf_init  # This must be the first TensorFlow-related import
import tensorflow as tf
from quantizeml.models import quantize, QuantizationParams
from utils import load_datasets, preprocess_for_quantization, set_random_seeds, evaluate_model
from visualization import plot_confusion_matrix
from config import *

SAVE_Q_MODEL = True
PREPROCESS_DATA = False
RUN_PARAMETER = "(No Preprocessing, no batch size q_fit)"
    
def quantize_model(model_path):
    """Quantizes the model and performs fine-tuning"""

    set_random_seeds(RANDOM_SEED)

    # Load and preprocess data
    X_train, y_train, X_val, y_val, X_test, y_test = load_datasets(DATA_DIR)
    
    
    # Load original model
    model = tf.keras.models.load_model(model_path)
    # Print original model summary
    print("Original model summary:")
    model.summary()
    
    # get original model stats
    predictions = model.predict(X_test)
    predictions = tf.argmax(predictions, axis=1)
    print("cnn_model stats")
    raw_accuracy, raw_f1 = evaluate_model(predictions, y_test)
    plot_confusion_matrix(y_test, predictions, "Original Model")
    
    
    if PREPROCESS_DATA:
        X_train = preprocess_for_quantization(X_train)
        X_val = preprocess_for_quantization(X_val)
        X_test = preprocess_for_quantization(X_test)
    
    # print data shapes
    print(f"X_train shape: {X_train.shape}")
    print(f"X_val shape: {X_val.shape}")
    print(f"X_test shape: {X_test.shape}")

    
    # Quantize model
    qparams = QuantizationParams(**QUANT_PARAMS)
    quantized_model = quantize(
        model,
        qparams=qparams,
        samples=X_train,
        num_samples=1024,
        batch_size=Q_BATCH_SIZE,
        epochs=Q_EPOCHS
    )
    
    # Compile quantized model
    """
    We use from_logits=True in SparseCategoricalCrossentropy for two key reasons:
    1. Numerical Stability: Raw logits (pre-softmax values) are more numerically stable 
       for loss calculation than probabilities. This is especially important in quantized
       models where numerical precision is reduced.
    2. Performance: Computing softmax during training can be computationally expensive.
       When from_logits=True, the loss function combines softmax and cross-entropy
       more efficiently in a single operation.

    If from_logits=False (default), the model's final layer should include a softmax.
    If from_logits=True, the model outputs raw logits and the loss function handles
    the softmax internally.
    """
    #TODO Check if this is necessary.
    quantized_model.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=LEARNING_RATE),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    
    # Print model summary
    quantized_model.summary()
    
    # evaluate the quantized model
    predictions = quantized_model.predict(X_test)
    predictions = tf.argmax(predictions, axis=1)
    print("quantized_model stats")
    raw_quantized_accuracy, raw_quantized_f1 = evaluate_model(predictions, y_test)
    plot_confusion_matrix(y_test, predictions, "Quantized Model")

    # Train quantized model
    quantized_model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        validation_data=(X_val, y_val)
    )
    
    # Evaluate tuned model
    tuned_predictions = quantized_model.predict(X_test)
    tuned_predictions = tf.argmax(tuned_predictions, axis=1)
    tuned_accuracy, tuned_f1 = evaluate_model(tuned_predictions, y_test)
    plot_confusion_matrix(y_test, tuned_predictions, "Quantized Model")
    
    # compare the results across different steps
    # Print comparison of results
    print(f"\n{RUN_PARAMETER}", ", Model Performance Comparison:")
    print("-" * 50)
    print(f"{'Model Type':<20} {'Accuracy':<15} {'F1 Score':<15}")
    print("-" * 50)
    print(f"{'Original CNN':<20} {raw_accuracy:<15.4f} {raw_f1:<15.4f}")
    print(f"{'Raw Quantized':<20} {raw_quantized_accuracy:<15.4f} {raw_quantized_f1:<15.4f}")
    print(f"{'Tuned Quantized':<20} {tuned_accuracy:<15.4f} {tuned_f1:<15.4f}")
    print("-" * 50)
    
    # Save quantized model if needed
    quantized_path = os.path.join(MODEL_DIR, 'quantized/quantized_model.h5')
    
    if SAVE_Q_MODEL:
        quantized_model.save(quantized_path)
        print(f"Quantized model saved to {quantized_path}")
    
    return quantized_model, quantized_path

if __name__ == "__main__":
    model_path = os.path.join(MODEL_DIR, 'cnn/cnn_model.h5')
    quantize_model(model_path)
