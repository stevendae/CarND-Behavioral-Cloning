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


np.random.seed(0)

def load_data():

    """
    Load in image filename data from csv
    """
    
    samples = []
    with open('./data/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append(line)

    """
    Split sample data in training and validation sets
    """

    from sklearn.model_selection import train_test_split
    train_samples, valid_samples = train_test_split(samples, test_size=0.2)

    return train_samples, valid_samples

def build_model(keep_prob):

    """
    Build Keras Neural Network Model
    """

    model = Sequential()
    model.add(Lambda(lambda x: x/127.5-1.0, input_shape=INPUT_SHAPE))
    model.add(Conv2D(24,5,5, activation = 'relu', subsample=(2,2)))
    model.add(Conv2D(36,5,5, activation = 'relu', subsample=(2,2)))
    model.add(Conv2D(48,5,5, activation = 'relu', subsample=(2,2)))
    model.add(Conv2D(64,3,3, activation = 'relu', subsample=(1,1)))
    model.add(Conv2D(64,3,3, activation = 'relu', subsample=(1,1)))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dropout(keep_prob))
    model.add(Dense(10))
    model.add(Dense(1))
    model.summary()
                                    
    return model



def compile_fit_model(lr, nb_epoch, model, train_samples, valid_samples, batch_size, a_shift, t_range):

    """
    Checkpoint Saver - Save if completed epoch has lowest validation loss
    """
    checkpointer = ModelCheckpoint(filepath="model-{epoch:02d}-{val_loss:.5f}.h5",
                                   monitor='val_loss',
                                   verbose = 1,
                                   save_best_only=True)

    
    """
    Set Number of Samples Per Epoch
    """
    
    nb_train = 20000
    nb_valid = len(valid_samples)


    """
    Instantiate training and validation generator objects
    """

    train_gen = generator(train_samples, batch_size, True, a_shift, t_range)
    validation_gen = generator(valid_samples, batch_size, False, a_shift, t_range)

    """
    Compile and Train Model
    """

    model.compile(loss='mse', optimizer=Adam(lr))
    model.fit_generator(train_gen, nb_train, 
                        nb_epoch, verbose=2, callbacks = [checkpointer],
                        validation_data = validation_gen, nb_val_samples=nb_valid,
                        class_weight = None, max_q_size = 10, nb_worker = 1,
                        pickle_safe = False, initial_epoch = 0)
    

def main():

    """
    Fine-Tune Parameters
    """
    learning_rate = 0.0001
    number_of_epochs = 10
    batch_size_p = 32
    keep_probability = 0.5
    angle_shift = 0.2
    translation_range = 50

    """
    Pipeline
    """
    train_data, valid_data = load_data()
    model = build_model(keep_probability)
    compile_fit_model(learning_rate,number_of_epochs,model,train_data, valid_data, batch_size_p, angle_shift, translation_range)
    
    
if __name__ == '__main__':
    """
    Initiate when called
    """
    main()    

