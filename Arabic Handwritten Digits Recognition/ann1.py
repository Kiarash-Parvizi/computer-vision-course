import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD

def start(X_train,X_test,y_train,y_test):
    # architecture
    model = Sequential([
        Dense(16, activation='relu', input_shape=(784,)),
        Dense(16, activation='relu'),
        Dense(10, activation='softmax')
    ])
    
    # compile
    #model.compile(optimizer=SGD(learning_rate=0.1),
    #              loss='sparse_categorical_crossentropy',
    #              metrics=['accuracy'])
    model.compile(optimizer = 'SGD', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    
    # train
    history = model.fit(X_train.reshape(-1, 784), y_train,
                        batch_size=16,
                        epochs=10,
                        validation_data=(X_test.reshape(-1, 784), y_test))
    
    # evaluation on the test dataset
    test_loss, test_acc = model.evaluate(X_test.reshape(-1, 784), y_test)
    print('Test loss:', test_loss)
    print('Test accuracy:', test_acc)

    # store the model in a file
    model.save('storage/ann1.h5')

