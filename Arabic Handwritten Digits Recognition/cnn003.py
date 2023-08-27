import numpy as np
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.models import load_model
from tensorflow.keras.layers import LeakyReLU

from tensorflow import keras
import tensorflow as tf

from keras.callbacks import EarlyStopping, ModelCheckpoint


def start(X_train,X_test,y_train,y_test):
    # define the early stopping callback
    early_stop = EarlyStopping(monitor='val_accuracy', patience=10, verbose=1, mode='max')
    
    # CNN model
    model = Sequential([
        Conv2D(32, (3, 3), activation=tf.nn.gelu, input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation=tf.nn.gelu),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(64, activation=tf.nn.gelu),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])

    # compile
    model.compile(optimizer=keras.optimizers.Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    # Define model checkpoint to save the weights of the best model
    model_checkpoint = ModelCheckpoint('storage/best_model_cnn003.h5', monitor='val_accuracy', mode='max', save_best_only=True, verbose=1)
    
    # train
    history = model.fit(X_train, y_train,
                        batch_size=64,
                        epochs=16,
                        validation_data=(X_test, y_test),
                        callbacks=[early_stop, model_checkpoint])
    # best model
    model = load_model('storage/best_model_cnn003.h5')
    
    # evaluate
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print('Test loss:', test_loss)
    print('Test accuracy:', test_acc)

    # save
    model.save('storage/cnn003.h5')

