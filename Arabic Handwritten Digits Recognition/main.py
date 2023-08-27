from load_display_dataset import *
#print(X_train[0])

import ann1
ann1.start(X_train,X_test,y_train,y_test)

import ann2
ann2.start(X_train,X_test,y_train,y_test)

import cnn000
cnn000.start(X_train,X_test,y_train,y_test)

import cnn001
cnn001.start(X_train,X_test,y_train,y_test)
import cnn002
cnn002.start(X_train,X_test,y_train,y_test)

import cnn003
cnn003.start(X_train,X_test,y_train,y_test)

import print_incorrect_predictions
print_incorrect_predictions.start('storage/cnn003.h5', X_test, y_test)

