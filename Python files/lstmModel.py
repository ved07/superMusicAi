import tensorflow as tf
import numpy as np
import glob
from keras.models import Sequential
from keras.layers import Dense, LSTM, BatchNormalization as BatchNorm, Dropout, Activation
from dataSequencing import DataStructuring
from keras.callbacks import ModelCheckpoint
class LSTMModel:
    def __init__(self):
        datastructuring = DataStructuring(100)
        self.lengthOfNotes = datastructuring.LengthOfNotes()
        self.model = Sequential()

        self.inputs = datastructuring.sequenceData()[0]
        self.outputs = datastructuring.sequenceData()[1]

    def layersInstantiate(self):
        self.model.add(LSTM(512,
                            input_shape=(self.inputs.shape[1],self.inputs.shape[2]),
                            recurrent_dropout=0.3,
                            return_sequences=True,
                            recurrent_activation='sigmoid'))
        self.model.add(LSTM(512,
                            recurrent_dropout=0.3,
                            return_sequences=True,
                            recurrent_activation='sigmoid'))
        self.model.add(LSTM(512,
                            recurrent_activation='sigmoid'))
        self.model.add(BatchNorm())
        self.model.add(Dropout(0.3))
        self.model.add(Dense(256))
        self.model.add(Activation('relu'))
        self.model.add(BatchNorm())
        self.model.add(Dropout(0.3))
        self.model.add(Dense(self.lengthOfNotes))
        self.model.add(Activation('softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    def train(self):
        filepath = "weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
        checkpoint = ModelCheckpoint(
            filepath,
            monitor='loss',
            verbose=0,
            save_best_only=True,
            mode='min'
        )
        callbacks_list = [checkpoint]
        print(self.inputs.shape, self.outputs.shape)
        self.model.fit(self.inputs, self.outputs, epochs=200, batch_size=128, callbacks=callbacks_list)



