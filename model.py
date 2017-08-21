import os
import csv
import numpy as np
import tensorflow as tf
from utils import INPUT_SHAPE, generator
from keras.models import Sequential
from keras.layers.core import Dense, Flatten, Dropout, Lambda
from keras.layers.convolutional import Convolution2D as Conv2D
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam


np.random.seed(0) #initialize random state

# Data Directory for Generator - samples[]

def load_data():

    samples = []
    with open('./data/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append(line)

    # Split Training and Validation Set

    from sklearn.model_selection import train_test_split
    train_samples, valid_samples = train_test_split(samples, test_size=0.2)

    return train_samples, valid_samples


# Keras Model

def build_model(keep_prob):



    model = Sequential()
    model.add(Lambda(lambda x: x/255.0-0.5, input_shape=INPUT_SHAPE))
    model.add(Conv2D(24,5,5, activation = 'relu', subsample=(2,2)))
    model.add(Conv2D(36,5,5, activation = 'relu', subsample=(2,2)))
    model.add(Conv2D(48,5,5, activation = 'relu', subsample=(2,2)))
    model.add(Conv2D(64,3,3, activation = 'relu', subsample=(1,1)))
    model.add(Conv2D(64,3,3, activation = 'relu', subsample=(1,1)))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dropout(0.25))
    model.add(Dense(10))
    model.add(Dense(1))
    model.summary()
                                    
    return model

# Save Best Model, Optimizer, Compiler, Train Method

def compile_fit_model(lr, nb_epoch, model, train_samples, valid_samples, batch_size, a_shift, t_range):
    

    checkpointer = ModelCheckpoint(filepath="model-{epoch:02d}-{val_loss:.5f}.h5",
                                   monitor='val_loss',
                                   verbose = 1)

    nb_train = len(train_samples)
    nb_valid = len(valid_samples)
    
    train_gen = generator(train_samples, batch_size, True, a_shift, t_range)
    validation_gen = generator(valid_samples, batch_size, False, a_shift, t_range)

    model.compile(loss='mse', optimizer=Adam(lr))
    model.fit_generator(train_gen, nb_train, 
                        nb_epoch, verbose=2, callbacks = [checkpointer],
                        validation_data = validation_gen, nb_val_samples=nb_valid,
                        class_weight = None, max_q_size = 10, nb_worker = 1,
                        pickle_safe = False, initial_epoch = 0)
    

def main():

    learning_rate = 0.0001
    number_of_epochs = 1
    batch_size_p = 32
    keep_probability = 0.5
    angle_shift = 0.25
    translation_range = 30
    
    train_data, valid_data = load_data()
    model = build_model(keep_probability)
    

    #print (next(train_gen))
    #print (next(validation_gen))
    
    compile_fit_model(learning_rate,number_of_epochs,model,train_data, valid_data, batch_size_p, angle_shift, translation_range)
    
    
if __name__ == '__main__':
    main()    

