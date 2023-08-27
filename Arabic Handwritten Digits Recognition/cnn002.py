import numpy as np
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

from tensorflow import keras

from keras.regularizers import l2
from keras.callbacks import EarlyStopping


def start(X_train,X_test,y_train,y_test):
    # define the early stopping callback
    early_stop = EarlyStopping(monitor='val_loss', patience=5)
    
    # CNN model with L2 regularization
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', kernel_regularizer=l2(0.01), input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.01)),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
        Dropout(0.5),
        Dense(10, activation='softmax', kernel_regularizer=l2(0.01))
    ])

    # compile
    model.compile(optimizer=keras.optimizers.Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    
    # train
    history = model.fit(X_train, y_train,
                        batch_size=32,
                        epochs=16,
                        validation_data=(X_test, y_test),
                        callbacks=[early_stop])
    
    # evaluate
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print('Test loss:', test_loss)
    print('Test accuracy:', test_acc)

    # save
    model.save('storage/cnn002.h5')

