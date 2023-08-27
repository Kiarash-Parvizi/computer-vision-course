import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2

# Load the MNIST data from files
X_train = pd.read_csv('../dataset/X_train.csv').values
y_train = pd.read_csv('../dataset/y_train.csv').values
X_test  = pd.read_csv('../dataset/X_test.csv' ).values
y_test  = pd.read_csv('../dataset/y_test.csv' ).values

X_train_raw = X_train.astype(np.uint8)
y_train_raw = y_train.astype(np.uint8)
X_test_raw  = X_test.astype(np.uint8)
y_test_raw  = y_test.astype(np.uint8)

X_train = X_train.astype('float32') / 255.0
X_test  = X_test.astype('float32') / 255.0

X_train = X_train.reshape(-1,28,28)
X_test = X_test.reshape(-1,28,28)

def eroded_cnt(img):
    # Apply erosion to the image
    erosion = cv2.erode(img, np.ones((5,5), np.uint8), iterations = 1)
    return np.count_nonzero(erosion)

# apply One hot encoding on the y
#from sklearn.preprocessing import OneHotEncoder
#enc = OneHotEncoder()
#y_train = enc.fit_transform(y_train).toarray()
#y_test  = enc.fit_transform(y_test ).toarray()
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

# Apply the rotation and flipping operations to all the images
X_train = np.rot90(X_train, k=1, axes=(1, 2))
X_train = np.flip(X_train, axis=1)

X_test = np.rot90(X_test, k=1, axes=(1, 2))
X_test = np.flip(X_test, axis=1)

# data augmentation:
runAug = True
#runAug = False
if runAug:
    # the kernel for the opening operation
    kernel = np.ones((5,5), np.uint8)
    # Apply opening operation to train images
    for i, img in enumerate(X_train):
        #print(y_train[i])
        #eCnt = eroded_cnt(img)
        #if eCnt < 20:
        #    continue
        if y_train[i][0] != 1: continue
        thresholded = (img > 0.0).astype(np.uint8)
        # morph
        opened = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, kernel)
        X_train[i] = opened
    
    # Apply opening operation to test images
    for i, img in enumerate(X_test):
        #eCnt = eroded_cnt(img)
        #if eCnt < 20:
        #    continue
        if y_test[i][0] != 1: continue
        thresholded = (img > 0.0).astype(np.uint8)
        # morph
        opened = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, kernel)
        X_test[i] = opened

    # aug:
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    
    # create an instance of the ImageDataGenerator class
    datagen = ImageDataGenerator(
        rotation_range=12,  # randomly rotate images by up to 12 degrees
        width_shift_range=0.11,  # randomly shift images horizontally by up to 11%
        height_shift_range=0.11,  # randomly shift images vertically by up to 11%
        horizontal_flip=False,  # do not randomly flip images horizontally
        vertical_flip=False,  # don't flip images vertically
        )
    
    # fit the ImageDataGenerator to the training data
    #datagen.fit(X_train.reshape(-1, 28, 28, 1))
    
    # create an iterator for the augmented data
    augmented_data = datagen.flow(X_train.reshape(-1, 28, 28, 1), y_train, batch_size=32)
    #print(augmented_data[0])
    
    # Loop over the generator object and reshape each batch of data
    # Get the augmented data and corresponding labels
    aug_X_train = np.empty((len(X_train)*2, 28, 28), dtype=np.float32)
    aug_y_train = np.empty((len(X_train)*2, 10), dtype=np.uint8)
    
    for i in range(len(X_train)):
        batch = augmented_data.next()
        aug_X_train[2*i] = batch[0][0,:,:,0]
        aug_X_train[2*i+1] = batch[0][1,:,:,0]
        #print(batch[1][0])
        aug_y_train[2*i] = batch[1][0]
        aug_y_train[2*i+1] = batch[1][1]
        if i % 1000 == 0:
            print("step: ", i, " | remaining steps: ", len(X_train)-i)
    
    # Concatenate the original data with the augmented data
    X_train = np.concatenate([X_train, aug_X_train], axis=0)
    y_train = np.concatenate([y_train, aug_y_train], axis=0)

    # store the results
    print('saving to file')
    np.save('storage/X_train.npy', X_train)
    np.save('storage/y_train.npy', y_train)
    np.save('storage/X_test.npy', X_test)
    np.save('storage/y_test.npy', y_test)
else:
    # load the augmented data
    print('loading augmented data from a file')
    X_train = np.load('storage/X_train.npy')
    y_train = np.load('storage/y_train.npy')
    X_test = np.load('storage/X_test.npy')
    y_test = np.load('storage/y_test.npy')

# Print the new shapes of the data
print('number of images in X_train: ', len(X_train))
print('number of images in X_test : ', len(X_test))
print('shape of X_train:', X_train.shape)
print('shape of X_test:', X_test.shape)

while True:
    random_idxs = np.random.choice(len(X_train), size=6, replace=False)
    images = X_train[random_idxs]
    labels = y_train[random_idxs]
    
    # Plot the images and their labels
    fig, axs = plt.subplots(2, 3, figsize=(8, 6))
    axs = axs.flatten()
    
    for i in range(len(axs)):
        axs[i].imshow(images[i], cmap='gray')
        #axs[i].set_title('Label: {}'.format(enc.inverse_transform([labels[i]])[0]))
        axs[i].set_title('Label: {}'.format(np.argmax(labels[i])))
        axs[i].axis('off')
    
    plt.tight_layout()
    plt.show()
    break


