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
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
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

"""
Best: 0.548240 using {'activation': 'relu', 'dropout_rate': 0.0, 'filters': 64, 
'kernel_size': (3, 3), 'learning_rate': 0.001, 'pool_size': (2, 2)}
"""


from skopt.space import Real, Integer, Categorical
from tensorflow.keras.layers import Activation
from keras.layers import GlobalAveragePooling2D
from tensorflow.keras import utils
import numpy as np
from keras.layers import BatchNormalization
y_train
np.unique(y_train)
N_classes=len(np.unique(y_train))
N_classes
y_train=utils.to_categorical(y_train, N_classes)
y_test=utils.to_categorical(y_test, N_classes)

space = [Real(1e-6, 1e-2, prior='log-uniform', name='lr'),
         Categorical(['adam', 'sgd', 'rmsprop'], name='optimizer'),
         Categorical(['relu', 'sigmoid', 'tanh'], name='activation'),
         Integer(16, 128, name='filters'),
         Integer(32, 256, name='batch_size'),
         Integer(1, 3, name='num_layers'),
         Categorical([True, False], name = 'No_of_batch_norm'),
         Real(0.0, 0.5, prior='uniform', name='dropout_rate')]


def create_model(learning_rate=0.001, dropout_rate=0.0, filters=32, kernel_size=(3, 3), optimizer='adam',
                 pool_size=(2, 2), activation='relu', num_layers=1, No_of_batch_norm= True,batch_size=128):
    model = Sequential()
    model.add(Conv2D(filters, kernel_size, input_shape=(32,32,3)))
    model.add(BatchNormalization())
    
    for i in range(num_layers):
        model.add(Conv2D(filters, kernel_size))
        if No_of_batch_norm:
            model.add(BatchNormalization())
        model.add(Activation(activation))  # use the activation string to map to corresponding function
    model.add(GlobalAveragePooling2D())
    model.add(Dropout(dropout_rate))
    
    model.add(Dense(N_classes, activation='softmax'))
    # Map the optimizer string to the corresponding function
    if optimizer == 'adam':
       opt = Adam(learning_rate=learning_rate)
    elif optimizer == 'sgd':
       opt = SGD(learning_rate=learning_rate)
    elif optimizer == 'rmsprop':
       opt = RMSprop(learning_rate=learning_rate)
    else:
       raise ValueError(f'Invalid optimizer: {optimizer}')
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model


# Define the objective function for Bayesian Optimization
def objective(params):
    learning_rate, dropout_rate, batch_size, filters, optimizer, activation, num_layers, No_of_batch_norm= params
    model = create_model(learning_rate=learning_rate,dropout_rate=dropout_rate,  batch_size=batch_size,  filters=filters, 
                         optimizer=optimizer, activation = activation , num_layers=num_layers, No_of_batch_norm=No_of_batch_norm)
    history = model.fit(X_train, y_train, epochs=20, validation_split=0.1, verbose=0)
    score = history.history['val_accuracy'][-1]
    return -score


# Define the objective function for Bayesian Optimization
@use_named_args(space)
def evaluate_model(**params):
    # Print the hyperparameters being tuned
    print(f"Training with {params}")
    
    # Create a new model with the given hyperparameters
    model = create_model(params)
    
    # Compile the model
    optimizer_dict = {'adam': Adam, 'sgd': SGD, 'rmsprop': RMSprop}
    optimizer_func = optimizer_dict[params['optimizer']](lr=params['lr'])
    model.compile(loss='categorical_crossentropy', optimizer=optimizer_func, metrics=['accuracy'])
    
    # Fit the model on the training data
    model.fit(X_train, y_train, batch_size=params['batch_size'], epochs=50, verbose=0)
    
    # Evaluate the model on the validation data
    _, accuracy = model.evaluate(X_test, y_test, verbose=0)
    
    # Return the negative accuracy (Bayesian Optimization tries to minimize the objective function)
    return -accuracy

# Run Bayesian Optimization
result = gp_minimize(evaluate_model, space, n_calls=50, random_state=35)

# Print the best hyperparameters found
print(f"Best hyperparameters:", result.x)

best_model = create_model(*result.x)
best_model.fit(X_train, y_train, epochs=20, validation_split=0.1, verbose=0)
_, test_accuracy = best_model.evaluate(X_test, y_test, verbose=0)
print(f"Test accuracy: {test_accuracy:.4f}")

#Visualization#
import matplotlib.pyplot as plt
from skopt.plots import plot_convergence

# Plot the convergence plot
plot_convergence(result)
#plt.show()

# Extract the hyperparameters and negative accuracies from the optimization result
params = result.x_iters
accs = [-result.func_vals[i] for i in range(len(result.func_vals))]

# Create a scatter plot of hyperparameters vs. accuracy
fig, axs = plt.subplots(nrows=len(space), figsize=(8, 30))
for i, param in enumerate(space):
    x = [p[i] for p in params]
    axs[i].scatter(x, accs)
    axs[i].set_xlabel(param.name)
    axs[i].set_ylabel('Negative Accuracy')

plt.show()

import pandas as pd

# create a DataFrame from the grid search results
results_df = pd.DataFrame(grid_result.cv_results_)

# extract the columns with the hyperparameters and the mean test score
param_cols = [col for col in results_df.columns if col.startswith('param_')]
param_cols.append('mean_test_score')
param_cols
print(results_df.columns)

param_cols.remove('mean_test_score')
pivot_table = pd.pivot_table(results_df, values='mean_test_score', index=param_cols)



# create a pivot table to show mean test score by hyperparameter combination
pivot_table = pd.pivot_table(results_df, values='mean_test_score', index=param_cols)

print(pivot_table)

