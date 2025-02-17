"""
Visualization Helpers

Functions for plotting and visualizing model results.
"""

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plot_sample_prediction(model, image, true_label):
    """
    Plots a sample image and its prediction
    
    Args:
        model: The trained model
        image: Input image
        true_label: True class label
    """
    # Reshape image for prediction
    image_reshaped = image.reshape(1, 128, 128, 1)
    outputs = model.predict(image_reshaped)

    # Create subplot
    f, axarr = plt.subplots(1, 2)
    
    # Plot image
    axarr[0].imshow(image.reshape((128, 128)), cmap=cm.Greys_r)
    axarr[0].set_title(f'Class {true_label}')
    
    # Plot prediction bars
    axarr[1].bar(range(5), outputs.squeeze())
    axarr[1].set_xticks(range(5))
    plt.show()

def plot_confusion_matrix(y_true, y_pred, title):
    """
    Plots confusion matrix
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        title: Plot title
    """
    plt.figure(figsize=(10, 8))
    ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred)).plot()
    plt.title(title)
    plt.show()
