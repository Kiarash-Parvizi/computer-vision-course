import numpy as np
import tensorflow as tf
from keras.models import Sequential
from tensorflow.keras import layers, models

import matplotlib.pyplot as plt


def start(modelPath: str, X_test, y_test):
    # Load model from file
    model = models.load_model(modelPath)

    # Predict the classes for the test set
    y_pred = model.predict(X_test)
    y_pred = y_pred.argmax(axis=1)
    y_test = y_test.argmax(axis=1)
    
    # Print some of the incorrectly classified images
    incorrect_indices = np.where(y_pred != y_test)[0]
    print('number of incorrect predictions =', len(incorrect_indices))
    for i in range(6):
        index = incorrect_indices[i]
        image = X_test[index].reshape(28, 28)
        label = y_test[index]
        predicted_label = y_pred[index]
        print('Index:', index)
        print('True label:', label)
        print('Predicted label:', predicted_label)
        plt.imshow(image, cmap='gray')
        plt.show()


