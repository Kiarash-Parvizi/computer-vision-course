import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

def start(X_train,X_test,y_train,y_test):
    # model architecture
    model = Sequential([
        Dense(32, activation='relu', input_shape=(784,)),
        Dense(32, activation='relu'),
        Dense(32, activation='relu'),
        Dense(10, activation='softmax')
    ])
    
    # compile
    model.compile(optimizer='Adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    # train
    history = model.fit(X_train.reshape(-1, 784), y_train,
                        batch_size=32,
                        epochs=8,
                        validation_data=(X_test.reshape(-1, 784), y_test))
    
    # evaluate
    test_loss, test_acc = model.evaluate(X_test.reshape(-1, 784), y_test)
    print('Test loss:', test_loss)
    print('Test accuracy:', test_acc)

    # save
    model.save('storage/ann2.h5')

