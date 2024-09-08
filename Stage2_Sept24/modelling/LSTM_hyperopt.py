
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.models import load_model
from sklearn.utils import class_weight
from tensorflow.keras.utils import to_categorical
import tensorflow.keras.layers as layers
# from data_generator import NpyDataGenerator
from hyperopt import hp, tpe, Trials, fmin
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os

from tqdm import tqdm
from collections import Counter
import numpy as np
from tensorflow.keras.optimizers import RMSprop, SGD, Adam,Adagrad
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.utils.multiclass import unique_labels
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
# import einops
from data_generator_v2 import NpyDataGenerator

train_path = "/media/kashraf/TOSHIBA EXT/Dissertation/stage2/wavelet/Mean_power_wt/Beta/train/"
test_path = "/media/kashraf/TOSHIBA EXT/Dissertation/stage2/wavelet/Mean_power_wt/Beta/test/"
train_gen = NpyDataGenerator(train_path,batch_size=32)
validation_gen = NpyDataGenerator(test_path,batch_size=1,shuffle=False)
print("Training samples: ",train_gen.num_samples)
print("Test samples: ",validation_gen.num_samples)



import pickle
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Flatten, Bidirectional
from tensorflow.keras.optimizers import Adam, RMSprop, SGD, Nadam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
num_classes= 4
# Define hyperparameter search space
space = {
    'num_layers': hp.choice('num_layers', [1, 2, 3, 4, 5]),
    'units': hp.choice('units', [16, 32, 64, 128, 256]),
    'dropout': hp.uniform('dropout', 0, 0.5),
    'recurrent_dropout': hp.uniform('recurrent_dropout', 0, 0.5),
    'learning_rate': hp.loguniform('learning_rate', -5, -2),
    'activation': hp.choice('activation', ['tanh', 'relu', 'sigmoid']),
    'optimizer': hp.choice('optimizer', ['adam', 'rmsprop', 'sgd']),
    'l2_reg': hp.loguniform('l2_reg', -6, -3),
    'clipvalue': hp.uniform('clipvalue', 1, 5),
    'bidirectional': hp.choice('bidirectional', [True, False])
}

# Initialize Trials object
trials = Trials()

# Function to save the best hyperparameters to a pickle file
def save_best_params(best_params):
    with open('best_params.pkl', 'wb') as f:
        pickle.dump(best_params, f)

# Function to load the best hyperparameters from the pickle file
def load_best_params():
    try:
        with open('best_params.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

# Objective function
def objective(params):
    model = Sequential()

    for i in range(params['num_layers']):
        if i == 0:  # First layer with input shape
            if params['bidirectional']:
                model.add(Bidirectional(LSTM(params['units'], activation='tanh', 
                                             recurrent_activation='sigmoid',
                                             dropout=params['dropout'], recurrent_dropout=0, 
                                             return_sequences=True, kernel_regularizer=l2(params['l2_reg'])),
                                             input_shape=(176, 64)))
            else:
                model.add(LSTM(params['units'], activation='tanh', recurrent_activation='sigmoid',
                               dropout=params['dropout'], recurrent_dropout=0, 
                               return_sequences=True, kernel_regularizer=l2(params['l2_reg']),
                               input_shape=(176, 64)))
        elif i == params['num_layers'] - 1:  # Last LSTM layer
            if params['bidirectional']:
                model.add(Bidirectional(LSTM(params['units'], activation='tanh', 
                                             recurrent_activation='sigmoid',
                                             dropout=params['dropout'], recurrent_dropout=0, 
                                             kernel_regularizer=l2(params['l2_reg']))))
            else:
                model.add(LSTM(params['units'], activation='tanh', recurrent_activation='sigmoid',
                               dropout=params['dropout'], recurrent_dropout=0, 
                               kernel_regularizer=l2(params['l2_reg'])))
        else:  # Middle LSTM layers
            if params['bidirectional']:
                model.add(Bidirectional(LSTM(params['units'], activation='tanh', 
                                             recurrent_activation='sigmoid',
                                             dropout=params['dropout'], recurrent_dropout=0, 
                                             return_sequences=True, kernel_regularizer=l2(params['l2_reg']))))
            else:
                model.add(LSTM(params['units'], activation='tanh', recurrent_activation='sigmoid',
                               dropout=params['dropout'], recurrent_dropout=0, 
                               return_sequences=True, kernel_regularizer=l2(params['l2_reg'])))
    
    # Add output layers
    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax'))

    # Optimizer and other code remain the same

    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer=params["optimizer"], metrics=['accuracy'])

    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)

    # Train the model
    model.fit(train_gen, epochs=50, batch_size=32, validation_data=validation_gen, 
              callbacks=[early_stopping], verbose=2)

    # Evaluate the model on the validation set
    loss, accuracy = model.evaluate(validation_gen, verbose=1)

    # Return the negative accuracy as the objective value (for minimization)
    return {'loss': -accuracy, 'status': STATUS_OK}


# Callback to print and save the best parameters after each trial
def trial_callback(trial):
    best_trial = trials.best_trial
    print(f"Trial {len(trials.trials)} completed with result: {trial['result']}")
    print(f"Best parameters so far: {best_trial['result']}")
    save_best_params(best_trial['result'])

# Load the best hyperparameters from previous session (if any)
best_params = load_best_params()
if best_params:
    print(f"Loaded best parameters from previous session: {best_params}")

# Run the hyperparameter optimization
best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=25, trials=trials, verbose=True, 
            catch_eval_exceptions=True)

# Save the final best hyperparameters to the pickle file
save_best_params(best)
print('Best hyperparameters:', best)


