#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.models import load_model
from sklearn.utils import class_weight
import tensorflow.keras.layers as layers
from data_generator import NpyDataGenerator

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
from tensorflow.keras.layers import *
import einops
from tensorflow.keras.regularizers import l2
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

   


# In[3]:


train_path = "/media/kashraf/Elements/Dissertation/data/preprocessed/visual/single_trial/3D_wt_data/beta/train"
test_path = "/media/kashraf/Elements/Dissertation/data/preprocessed/visual/single_trial/3D_wt_data/beta/test"
train_gen = NpyDataGenerator(train_path,batch_size=4)
validation_gen = NpyDataGenerator(test_path,batch_size=1,shuffle=False)
print("Training samples: ",train_gen.num_samples)
print("Test samples: ",validation_gen.num_samples)


# ### Search Space and objective function

# In[4]:



# Define the search space
space = {
    'n_filters': hp.choice('n_filters', [32, 64, 128]),
    'n_layers': hp.choice('n_layers', [2, 3, 4, 5]),
    'kernel_size': hp.choice('kernel_size', [(3, 3, 3), (5, 5, 5)]),
    'pool_size': hp.choice('pool_size', [(2, 2, 2), (3, 3, 3)]),
    'dropout_rate': hp.uniform('dropout_rate', 0, 0.5),
    'learning_rate': hp.loguniform('learning_rate', -5, -1),
    'batch_size': hp.choice('batch_size', [4, 8, 16,32]),
    'n_epochs': hp.choice('n_epochs', [50, 100, 150]),
    'optimizer': hp.choice('optimizer', [Adam, SGD,RMSprop,Adagrad]),
    'pooling_type': hp.choice('pooling_type', ['max', 'average']),
    'l2_lambda': hp.uniform('l2_reg', 0, 0.01),
    'use_residual': hp.choice('use_residual', [True, False])
}

## Create results folder
# Define the file name to save the trial results
if not os.path.exists('results'):
    os.makedirs('results')

num_files = len([name for name in os.listdir('results') if name.endswith('.json')])
file_name = os.path.join('results', f'{num_files+1}.pickle')

# Define the objective function
def objective(params):
    model = tf.keras.Sequential()
    model.add(Conv3D(params['n_filters'], params['kernel_size'], activation='relu', padding='same', input_shape=(112, 112, 176, 1), kernel_regularizer=l2(params['l2_lambda'])))
    model.add(BatchNormalization())
    
    if params['pooling_type'] == 'max':
        model.add(MaxPooling3D(params['pool_size']))
    else:
        model.add(tf.keras.layers.AveragePooling3D(params['pool_size']))

    model = tf.keras.Sequential()
    model.add(Conv3D(params['n_filters'], params['kernel_size'], activation='relu', padding='same', input_shape=(112, 112, 176, 1), kernel_regularizer=l2(params['l2_lambda'])))
    model.add(BatchNormalization())
    
    if params['pooling_type'] == 'max':
        model.add(MaxPooling3D(params['pool_size']))
    else:
        model.add(tf.keras.layers.AveragePooling3D(params['pool_size']))

    for i in range(params['n_layers']-1):
        if params['use_residual'] and i > 0:
            residual = model
        else:
            residual = None

        model.add(Conv3D(params['n_filters'], params['kernel_size'], activation='relu', padding='same', kernel_regularizer=l2(params['l2_lambda'])))
        model.add(BatchNormalization())

        if params['pooling_type'] == 'max':
            model.add(MaxPooling3D(params['pool_size']))
        else:
            model.add(tf.keras.layers.AveragePooling3D(params['pool_size']))
            
        model.add(Dropout(params['dropout_rate']))

        if residual is not None:
            model = tf.keras.Sequential()
            model.add(residual)
            model.add(Conv3D(params['n_filters'], (1, 1, 1), activation='relu', padding='same', kernel_regularizer=l2(params['l2_lambda'])))
            model.add(BatchNormalization())
            model.add(tf.keras.layers.Add()([residual, model]))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(params['dropout_rate']))
    model.add(Dense(4, activation='softmax'))

    optimizer = params['optimizer'](learning_rate=params['learning_rate'])
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(train_gen, epochs=params['n_epochs'], validation_data=validation_gen)

    score = model.evaluate(validation_gen, verbose=0)
    # Save the trial results to a pickle file
    with open(file_name, 'ab') as f:
        trials = Trials
        pickle.dump(trials, f)
    


    return {'loss': -score[1], 'status': STATUS_OK}


# ### Run Trials

# In[ ]:


if __name__ == '__main__':
    # Load the data

    # Create the Trials object and load previous trials if available
    trials = Trials()
    if os.path.exists('results/trials.p'):
        with open('results/trials.p', 'rb') as f:
            trials = pickle.load(f)

    # Run the hyperparameter optimization
    best = fmin(fn=objective, space=space, algo=tpe.suggest, trials=trials, max_evals=10)

    # Save the trials object to a pickle file
    with open('results/trials.p', 'wb') as f:
        pickle.dump(trials, f)

    print(best)

