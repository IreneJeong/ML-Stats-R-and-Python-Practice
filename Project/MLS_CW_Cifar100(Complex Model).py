#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 17:51:11 2023

@author: jeongdahye
"""

from tensorflow.keras.datasets import cifar10
from tensorflow.keras import utils
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import SGD, Adam, RMSprop

# importing of service libraries
import numpy as np
import matplotlib.pyplot as plt

print('Libraries imported.')


# Random seed 
import random
import tensorflow as tf
import os


def set_seed(seed=42):
    '''
    Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY.
    '''
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    session_conf = tf.compat.v1.ConfigProto(
        intra_op_parallelism_threads=1,
        inter_op_parallelism_threads=1)
    sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
    tf.compat.v1.keras.backend.set_session(sess)
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    
set_seed()

# Data Preprocessing 
from tensorflow.keras.datasets import cifar100
(input_X_train, output_y_train),(input_X_test, output_y_test)=cifar100.load_data()

print('input_X_train shape: ', input_X_train.shape)
print(input_X_train.shape[0], 'train samples')
print(input_X_test.shape[0], 'test samples')

IMG_CHANNELS = 3
IMAGE_SIZE = input_X_train.shape[1]

print('Image variables initialisation')

N_CLASSES =len(np.unique(output_y_train))

# output data one-hot encoding : Only for small number of classes(CIFAR10 )
#output_y_train=utils.to_categorical(output_y_train, N_CLASSES)
#output_y_test=utils.to_categorical(output_y_test, N_CLASSES)

# To normalize the value in between 0 and 1 (there are 255 kinds)
input_X_train=input_X_train.astype('float32')
input_X_test=input_X_test.astype('float32')

input_X_train /=255
input_X_test /=255


#Define the function for plotting the history of the training of the model
def plot_history(history):
    val_loss = history.history['val_loss' ]
    loss =     history.history['loss' ]
    acc =      history.history['accuracy' ]
    val_acc =  history.history['val_accuracy' ]

    epochs    = range(1,len(acc)+1,1)

    plt.plot  ( epochs,     acc, 'r--', label='Training acc'  )
    plt.plot  ( epochs, val_acc,  'b', label='Validation acc')
    plt.title ('Training and validation accuracy')
    plt.ylabel('acc')
    plt.xlabel('epochs')
    plt.legend()

    plt.figure()

    plt.plot  ( epochs,     loss, 'r--', label='Training loss' )
    plt.plot  ( epochs, val_loss ,  'b', label='Validation loss' )
    plt.title ('Training and validation loss')
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.legend()

    plt.figure()
    
    
# Data Augmentation
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen=ImageDataGenerator(rotation_range=40, 
                           width_shift_range=0.2, 
                           height_shift_range=0.2, 
                           zoom_range=0.2, 
                           horizontal_flip=True, 
                           fill_mode='nearest')

# rotation_range is a value in degrees (0 - 180) for randomly rotating pictures
# width_shift and height_shift are ranges for randomly translating pictures vertically or horizontally
# zoom_range is for randomly zooming pictures 
# horizontal_flip is for randomly flipping the images horizontally
# fill_mode fills in new pixels that can appear after a rotation or a shift

# Generate augmented images
batch_size = 32
input_X_train_aug = np.zeros_like(input_X_train)

for i, batch in enumerate(datagen.flow(input_X_train, batch_size=batch_size)):
    input_X_train_aug[i*batch_size:(i+1)*batch_size] = batch
    if (i+1)*batch_size >= len(input_X_train):
        break    


from keras.layers import BatchNormalization
# Complex DNN model definition
model = Sequential()
KERNEL=5
# hidden 1 : conv + conv + pool + dropout 
model.add(Conv2D(32, KERNEL, padding='same', input_shape=(IMAGE_SIZE, IMAGE_SIZE, IMG_CHANNELS)))
model.add(Activation('relu'))
model.add(Conv2D(32,  KERNEL, padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(BatchNormalization())

# hidden 2 : conv + conv + pool + dropout 
model.add(Conv2D(64,  KERNEL, padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(BatchNormalization())
 
# hidden 3 : flatten + droupout 
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))

# output 
model.add(Dense(N_CLASSES))
model.add(Activation('softmax'))


#training constants
BATCH_SIZE1 = 512
N_EPOCH1 = 60
VERBOSE1 = 2
VALIDATION_SPLIT1 = 0.2
learning_rate1 = 0.0001
opt = Adam(learning_rate = learning_rate1)

print('Main variables initialised.')

from tensorflow.keras.callbacks import EarlyStopping

# define early stopping callback
earlystop_callback = EarlyStopping(
    monitor='val_loss', # monitor validation loss
    min_delta=0.001, # minimum change in the monitored quantity to qualify as an improvement
    patience=5, # number of epochs with no improvement after which training will be stopped
    verbose=1 # prints a message when early stopping is triggered
)

# Model Compile
model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
model.summary()

# Model Fitting - W/O Data Augementation
history = model.fit(input_X_train, output_y_train, 
                    batch_size=BATCH_SIZE1, epochs=N_EPOCH1, 
                    validation_split=VALIDATION_SPLIT1, 
                    verbose=VERBOSE1, callbacks=[earlystop_callback])


plot_history(history)
test_loss,test_acc=model.evaluate(input_X_test, output_y_test, verbose=2)

print("test accuracy: ",test_acc)

# Result with data augmentation
model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
model.summary()

history=model.fit(input_X_train_aug, output_y_train,  batch_size=BATCH_SIZE1, 
                  epochs=N_EPOCH1, validation_split=VALIDATION_SPLIT1, 
                    verbose=VERBOSE1,  callbacks=[earlystop_callback])


plot_history(history)
test_loss,test_acc=model.evaluate(input_X_test, output_y_test, verbose=2)

print("test accuracy: ",test_acc)