import tf_init
from utils import set_random_seeds
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, classification_report,accuracy_score, roc_auc_score, roc_curve
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from scipy.signal import resample
import gc
import glob
import matplotlib.pyplot as plt
import keras_tuner as kt  
import os
import seaborn as sns
from tensorflow.keras.utils import to_categorical
from scipy import signal
from keras.layers import BatchNormalization,Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, LeakyReLU, MaxPooling2D, Dropout, GlobalAveragePooling2D, Dense,ReLU
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.models import load_model
import keras_tuner as kt
import time
from keras import layers
import keras_tuner as kt
import keras
import shutil
from tensorflow.keras.callbacks import ModelCheckpoint
from quantizeml.models import quantize, QuantizationParams
#from cnn2snn import convert, check_model_compatibility, quantize_layer
from keras.optimizers import Adam
import matplotlib.cm as cm
def clear_memory():
    gc.collect()
    tf.keras.backend.clear_session()
def preprocess_for_quantization(data):
    # Scale data to [0, 255] range for 8-bit quantization
    data_scaled = ((data - data.min()) / (data.max() - data.min()) * 255).astype('uint8')
    return data_scaled
def create_cnn_model():
    model = keras.Sequential([
        # Input layer TODO: Why do we normalise and scale the input data? this is done also in the preprocess_for_quantization function
        #TODO: Consider using a Batch Normalisation layer instead for better performance
        #TODO: This is redundant, it removes the added value of the preprocess_for_quantization function returns back to decimal
        # We can print after the rescaling layer to see the values (see data analysis file).
        keras.layers.Rescaling(1. / 255, input_shape=(128, 128, 1)),
        # First Convolutional Block
        layers.Conv2D(16, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D((3, 3), padding='same'),
        layers.Dropout(0.2),
        
        # Second Convolutional Block
        layers.Conv2D(32, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D((3, 3), padding='same'),
        layers.Dropout(0.2),
        
        # Third Convolutional Block
        layers.Conv2D(64, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D((3, 3), padding='same'),
        layers.Dropout(0.2),
        
        # Flatten and Dense Layers
        layers.Flatten(),
        layers.Dense(96),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.Dropout(0.2),
        
        # Output Layer
        layers.Dense(5)  
    ], name='cnn_model')
    
    return model
# Instantiate the model

set_random_seeds(42)

clear_memory()
model = create_cnn_model()

# Optional: Compile the model
model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),  # Change to sparse_categorical_crossentropy
    metrics=['accuracy']
)

# Print model summary for verification
model.summary()

# Set GPU as the visible device for TensorFlow
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


save_dir = 'datasets'
X_train = np.load(os.path.join(save_dir, 'X_train2.npy'))
y_train = np.load(os.path.join(save_dir, 'y_train2.npy'))
X_val = np.load(os.path.join(save_dir, 'X_val2.npy'))
y_val = np.load(os.path.join(save_dir, 'y_val2.npy'))
X_test = np.load(os.path.join(save_dir, 'X_test2.npy'))
y_test = np.load(os.path.join(save_dir, 'y_test2.npy'))

# Convert one-hot encoded labels to integer labels
y_train = np.argmax(y_train, axis=1)
y_val = np.argmax(y_val, axis=1)
y_test = np.argmax(y_test, axis=1)


#Normalize the dataset TODO: Check if this is necessary before standard model training.
X_train_q = preprocess_for_quantization(X_train)
X_val_q = preprocess_for_quantization(X_val)
X_test_q = preprocess_for_quantization(X_test)


#Train the model
model.fit(X_train_q, y_train, epochs=5,  validation_data=(X_val_q, y_val))

#Original Model Evaluation
original_preds = model.predict(X_test_q)
original_preds = np.argmax(original_preds, axis=1)
print(original_preds[0])
#y_true = np.argmax(y_test, axis=1)
y_true = y_test
print(f"Original Model F1 Score: {classification_report(y_true, original_preds, output_dict=True)['weighted avg']['f1-score']:.4f}")
ConfusionMatrixDisplay(confusion_matrix(y_true, original_preds)).plot()
plt.title("Original Model Confusion Matrix")
plt.show()


#Convert to MicroPython and Evaluate it
qparams = QuantizationParams(input_weight_bits=8, weight_bits=8, activation_bits=8, per_tensor_activations=True)
quantized_model = quantize(model, qparams=qparams)
print("Model input shape:", quantized_model.input_shape)
print("Model output shape:", quantized_model.output_shape)
print("X_test shape", X_test_q.shape)
quantized_model.compile(  # Use sparse_categorical_crossentropy here too
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer='adam',
    metrics=['accuracy'])
score = quantized_model.evaluate(X_test_q, y_test, verbose=0)
print('Test accuracy after quantization and compiling:', score[1])
quantized_preds = quantized_model.predict(X_test_q)
quantized_preds = np.argmax(quantized_preds, axis=1)
quantized_f1 = classification_report(y_true, quantized_preds, output_dict=True)['weighted avg']['f1-score']
print(f"Quantized Model F1 Score: {quantized_f1:.4f}")
quantized_cm = confusion_matrix(y_true, quantized_preds)
ConfusionMatrixDisplay(quantized_cm, display_labels=range(5)).plot()
plt.title("Quantized Model (Compiled) Confusion Matrix")
plt.show()

model.fit(X_train_q, y_train, epochs=5, validation_data=(X_val_q, y_val))
score = quantized_model.evaluate(X_test_q.astype('uint8'), y_test, verbose=0)
print('Test accuracy after fine tuning:', score[1])
quantized_preds = quantized_model.predict(X_test_q)
quantized_preds = np.argmax(quantized_preds, axis=1)
quantized_f1 = classification_report(y_true, quantized_preds, output_dict=True)['weighted avg']['f1-score']
print(f"Quantized Model F1 Score: {quantized_f1:.4f}")
quantized_cm = confusion_matrix(y_true, quantized_preds)
ConfusionMatrixDisplay(quantized_cm, display_labels=range(5)).plot()
plt.title("Quantized Model (fine-tuned) Confusion Matrix")
plt.show()



#Convert to Akida model and Evaluate it
#print("Model compatible for Akida conversion:", check_model_compatibility(model))
""" model_akida = convert(quantized_model)
print(X_test_q.astype('uint8').shape)
X_test_batched = X_test_q.astype('uint8').reshape(-1, 128, 128, 1)
accuracy = model_akida.evaluate(X_test_batched, y_test)
print('Test accuracy after conversion:', accuracy)
akida_preds = model_akida.predict(X_test_batched)
# Convert predictions to class labels (integers)
akida_preds = np.argmax(akida_preds, axis=1)
print(type(akida_preds))
print(akida_preds.dtype)
print(akida_preds.shape)
print(akida_preds[:5])


# Test a single example
print("testing ...")
sample_image = 0
image = X_test_q[sample_image]  # Use X_test_q (quantized data)

# Reshape the image for the Akida model
outputs = model_akida.predict(image.reshape(1, 128, 128, 1))

print('Input Label: %i' % y_test[sample_image])

f, axarr = plt.subplots(1, 2)
axarr[0].imshow(X_test_q[sample_image].reshape((128, 128)), cmap=cm.Greys_r) # Use X_test_q
axarr[0].set_title('Class %d' % y_test[sample_image])
axarr[1].bar(range(5), outputs.squeeze()) # Number of classes is 5
axarr[1].set_xticks(range(5))
plt.show()

print(outputs.squeeze()) """
#akida_preds = np.argmax(akida_preds, axis=-1).flatten()
#ConfusionMatrixDisplay(confusion_matrix(y_true, akida_preds)).plot()
#plt.title("Akida Model Confusion Matrix")
#plt.show()