#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 18:34:45 2023

@author: jeongdahye
"""

#import numpy as np
#import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
#from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import GridSearchCV
from skopt import gp_minimize
from skopt.space import Real, Categorical, Integer
from skopt.utils import use_named_args
from keras.wrappers.scikit_learn import KerasClassifier

# Load CIFAR-10 dataset
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Convert pixel values to float and normalize
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Define the CNN model
def create_model(learning_rate=0.001, dropout_rate=0.0, filters=32, kernel_size=(3, 3), 
                 pool_size=(2, 2), activation='relu'):
    model = Sequential()
    model.add(Conv2D(filters=filters, kernel_size=kernel_size, activation=activation, 
                     input_shape=(32, 32, 3)))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(rate=dropout_rate))
    model.add(Conv2D(filters=filters*2, kernel_size=kernel_size, activation=activation))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(rate=dropout_rate))
    model.add(Flatten())
    model.add(Dense(units=64, activation=activation))
    model.add(Dropout(rate=dropout_rate))
    model.add(Dense(units=10, activation='softmax'))
    
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Create a KerasClassifier from the create_model function
keras_model = KerasClassifier(build_fn=create_model)

# Define the hyperparameter search space
param_grid = {'learning_rate': [0.0001, 0.001, 0.01, 0.1],
              'dropout_rate': [0.0, 0.1, 0.2, 0.5],
              'filters': [16, 32, 64],
              'kernel_size': [(3, 3), (5, 5), (7, 7)],
              'pool_size': [(2, 2), (3, 3)],
              'activation': ['relu', 'sigmoid']}

# Perform Grid Search to find the best hyperparameters
grid = GridSearchCV(estimator=keras_model, param_grid=param_grid, cv=3)
grid_result = grid.fit(X_train, y_train)

# Print the best hyperparameters found using Grid Search
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))



# Define the objective function for Bayesian Optimization
def objective(params):
    learning_rate, dropout_rate, filters, kernel_size, pool_size, activation = params
    model = create_model(learning_rate=learning_rate, dropout_rate=dropout_rate, filters=filters, 
                         kernel_size=kernel_size, pool_size=pool_size, activation=activation)
    history = model.fit(X_train, y_train, epochs=20, validation_split=0.1, verbose=0)
    score = history.history['val_accuracy'][-1]
    return -score


# Define the search space for Bayesian Optimization
space = [Real(1e-6, 1e-2, prior='log-uniform', name='lr'), 
              Integer(32, 256, name='batch_size'), 
              Integer(16, 128, name='filters'), 
              Integer(1, 4, name='num_blocks'), 
              Integer(2, 5, name='kernel_size')]

# Define the objective function for Bayesian Optimization
@use_named_args(space)
def evaluate_model(**params):
    # Print the hyperparameters being tuned
    print(f"Training with {params}")
    
    # Create a new model with the given hyperparameters
    model = create_model(params)
    
    # Compile the model
    model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=params['lr']), metrics=['accuracy'])
    
    # Fit the model on the training data
    model.fit(X_train, y_train, batch_size=params['batch_size'], epochs=50, verbose=0)
    
    # Evaluate the model on the validation data
    _, accuracy = model.evaluate(X_test, y_test, verbose=0)
    
    # Return the negative accuracy (Bayesian Optimization tries to minimize the objective function)
    return -accuracy

# Run Bayesian Optimization
result = gp_minimize(evaluate_model, space, n_calls=50, random_state=35)

# Print the best hyperparameters found
print(f"Best hyperparameters: {result.x}")
