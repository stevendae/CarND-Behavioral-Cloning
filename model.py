import os
import csv
import numpy as np
import tensorflow as tf
from utils import INPUT_SHAPE, batch_generator


# Data Directory for Generator - samples[]

samples = []
with open('./driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

# Split Training and Validation Set

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)


# Keras Model

def build_model(keep_prob,):

    from keras.models import Sequential
    from keras.layers.core import Dense, Flatten, Dropout
    from keras.layers.convolutional import Convolution2D as Conv2D
    from keras.callbacks import ModelCheckpoint

    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1., input_shape=INPUT_SHAPE))
    model.add(Conv2D(24,31,98, activation = relu, subsample=(2,2)))
    model.add(Conv2D(36,14,47, activation = relu, subsample=(2,2)))
    model.add(Conv2D(48,5,22, activation = relu, subsample=(2,2)))
    model.add(Conv2D(64,3,20, activation = relu, subsample=(1,1)))
    model.add(Conv2D(64,1,18, activation = relu, subsample=(1,1)))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dropout(keep_prob))
    model.add(Dense(50))
    model.add(Dropout(keep_prob))
    model.add(Dense(10))
    model.add(Dense(1))
    model.summary()

    
                                   

    return model

def compile_fit_model(lr, nb_epoch, model):

    checkpointer = ModelCheckpoint(filepath="model-{epoch:02d}-{val_loss:.2f}.h5", /
                                   monitor='val_acc', /
                                   verbose = 1, /
                                   save_best_only=True,/
                                   mode = max)

    model.optimizers.Adam(lr)
    model.compile(loss='mse')
    model.fit_generator(train_generator, samples_per_epoch = len(train_samples) /
                        nb_epoch=10, verbose=2, callbacks = [checkpointer], /
                        validation_data = validation_generator, nb_val_samples=len(validation_samples) /
                        class_weight = None, max_q_size = 10, nb_worker = 1,/
                        pickle_safe = False, initial_epoch = 0)
    
                        

def main():

    data = 
    
    

