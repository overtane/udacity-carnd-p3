
# # Behavioral cloning with Keras
# 
# 
# Here are the steps you'll take to build the network:
# 
# 1. First load the training data and do a train/validation split.
# 2. Preprocess data.
# 3. Build a convolutional neural network to classify traffic signs.
# 4. Evaluate performance of final neural network on testing data.
# 

import pickle
import numpy as np
from sklearn.model_selection import train_test_split
import math


# ## Load the Data
# 
# Start by importing the data from the pickle file.

training_file = '/Users/olli/Udacity/SDC/P2/traffic-signs-data/train.p'
with open(training_file, mode='rb') as f:
    train = pickle.load(f)


# ## Validate the Network
# Split the training data into a training and validation set.
# 

X_train, X_val, y_train, y_val = train_test_split(train['features'].astype('float'), train['labels'], test_size=0.20, random_state=0)


# ## Preprocess the Data
# 
# Now that you've loaded the training data, preprocess the data such that it's in the range between -0.5 and 0.5.

def norm(t):
    tmin = t.min();
    tmax = t.max();
    return (t - tmin) / (tmax - tmin) - 0.5

for i in range(len(X_train)):
    X_train[i] = norm(X_train[i])

for i in range(len(X_val)):
    X_val[i] = norm(X_val[i])

print(X_train.min(), X_train.max())
print(X_val.min(), X_val.max())


# Build the network here

from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, Flatten, MaxPooling2D, Dropout
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils

nfilters=32
kernel_size=(3,3)

model = Sequential()
model.add(Convolution2D(nfilters, kernel_size[0], kernel_size[1],
                        border_mode='valid',
                        input_shape=(32,32,3)))
model.add(Activation('relu'))
model.add(Dropout(0.50))
model.add(Convolution2D(nfilters, kernel_size[0], kernel_size[1]))
model.add(Activation('relu'))
model.add(Dropout(0.50))
model.add(Flatten())
model.add(Dense(1024)) # don't need to calculate input shape for fully connected layer!
model.add(Activation('relu'))
model.add(Dense(256)) # don't need to calculate input shape for fully connected layer!
model.add(Activation('relu'))
model.add(Dense(43))
model.add(Activation('softmax'))

model.summary()

# Compile and train the model here.

# one hot encode
Y_train = np_utils.to_categorical(y_train, 43)
Y_val   = np_utils.to_categorical(y_val, 43)

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(),
              metrics=['accuracy'])

history = model.fit(X_train, Y_train,
                    batch_size=128, nb_epoch=2,
                    verbose=1, validation_data=(X_val, Y_val))



# ## Testing
# Once you've picked out your best model, it's time to test it.

# Load test data

testing_file = '/Users/olli/Udacity/SDC/P2/traffic-signs-data/test.p'
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

X_test = test['features'].astype('float')
y_test = test['labels']

for i in range(len(X_test)):
    X_test[i] = norm(X_test[i])

# Preprocess data & one-hot encode the labels
Y_test = np_utils.to_categorical(y_test, 43)

# Evaluate model on test data
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])


