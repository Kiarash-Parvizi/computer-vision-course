import numpy as np
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout


def start(X_train,X_test,y_train,y_test):
    #X_train = X_train.reshape()
    # model architecture
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),   # dropout layer
        Dense(10, activation='softmax')
    ])

    # compile
    model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    
    # train
    history = model.fit(X_train, y_train,
                        batch_size=32,
                        epochs=16,
                        validation_data=(X_test, y_test))
    
    # evaluate
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print('Test loss:', test_loss)
    print('Test accuracy:', test_acc)

    # save
    model.save('storage/cnn001.h5')

