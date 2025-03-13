"""CNN model creation, training, and evaluation"""

import os
import tensorflow as tf
from utils import load_datasets, clear_memory, set_random_seeds, evaluate_model
from visualization import plot_confusion_matrix
from sklearn.metrics import accuracy_score, f1_score
from config import *

# Choose whether to save the model
SAVE_MODEL = False
MODEL_VERSION = "1.0"


def create_cnn_model():
    """Creates a basic CNN model architecture"""
    inputs = tf.keras.layers.Input(shape=INPUT_SHAPE)
    x = tf.keras.layers.Conv2D(16, (3, 3), padding='same')(inputs)  # Use the input layer as the starting point
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D((3, 3), padding='same')(x)  
    x = tf.keras.layers.Dropout(0.2)(x)

    x = tf.keras.layers.Conv2D(32, (3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D((3, 3), padding='same')(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    x = tf.keras.layers.Conv2D(64, (3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D((3, 3), padding='same')(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(96)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(NUM_CLASSES)(x)

    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    return model

def train_and_evaluate():
    """Main function to train and evaluate the CNN model"""
    clear_memory()
    
    # Set reproducibility
    set_random_seeds(RANDOM_SEED)
    
    # Load datasets
    X_train, y_train, X_val, y_val, X_test, y_test = load_datasets(DATA_DIR)
    
    # Create and compile model
    model = create_cnn_model()
    model.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=LEARNING_RATE),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    
    # Print model summary
    model.summary()
    
    # Train model
    model.fit(X_train, y_train, epochs=EPOCHS, 
              batch_size=BATCH_SIZE, validation_data=(X_val, y_val))
    
    # Evaluate model
    predictions = model.predict(X_test)
    predictions = tf.argmax(predictions, axis=1)
    # Calculate and print accuracy and f1 score
    accuracy, f1 = evaluate_model(predictions, y_test)
    
    
    # Plot results
    plot_confusion_matrix(y_test, predictions, "Original CNN Model")
    
    # Save model if required
    model_path = os.path.join(MODEL_DIR, f'cnn/cnn_model_{MODEL_VERSION}.h5')
    if SAVE_MODEL:
        model.save(model_path)
        print(f"Model saved to {model_path}")
    
    return model, model_path

if __name__ == "__main__":
    train_and_evaluate()
