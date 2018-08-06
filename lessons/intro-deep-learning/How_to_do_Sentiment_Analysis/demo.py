from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.data_utils import pad_sequences, VocabularyProcessor

import pandas as pd
import numpy as np
from keras.utils.np_utils import to_categorical
from sklearn.cross_validation import train_test_split
import random

# IMDB Dataset loading
# train, test, _ = imdb.load_data(path='imdb.pkl', n_words=10000,
                                # valid_portion=0.1)

load_model = 0
save_model = 1

dataframe = pd.read_csv('ign.csv').ix[:, 1:3]
# Fill null values with empty strings
dataframe.fillna(value='', inplace=True)

# score phrase and title
print(dataframe.columns.values)

# Extract the required columns for inputs and outputs
totalX = dataframe.title
totalY = dataframe.score_phrase

# Convert the strings in the input into integers corresponding to the dictionary positions
# Data is automatically padded so we need to pad_sequences manually
vocab_proc = VocabularyProcessor(15) # max document length
totalX = np.array(list(vocab_proc.fit_transform(totalX)))


# totalX contains a matrix with the word idi (indexes) of the sentences

# We will have 11 classes in total for prediction, indices from 0 to 10
vocab_proc2 = VocabularyProcessor(1)
totalY = np.array(list(vocab_proc2.fit_transform(totalY))) - 1

# here totalY is numbered dictionary entries (0, 1, ... to 10)
print(totalY[5])
# Convert the indices into 11 dimensional vectors
totalY = to_categorical(totalY, 11)
# here totalY is a binary matrix
print(totalY[5])

# Split into training and testing data
trainX, testX, trainY, testY = train_test_split(totalX, totalY, test_size=0.1)

print(trainX[0])
print(testX[0])
print(trainY[0])
print(testY[0])

# trainX, trainY = train
# testX, testY = test

# # Data preprocessing
# # Sequence padding
# trainX = pad_sequences(trainX, maxlen=100, value=0.)
# testX = pad_sequences(testX, maxlen=100, value=0.)
# # Converting labels to binary vectors
# trainY = to_categorical(trainY, nb_classes=2)
# testY = to_categorical(testY, nb_classes=2)

# Network building
# 15 words max, so 15 input data
net = tflearn.input_data([None, 15])
# dictionary has 10k words max
# Turns positive integers (indexes) into dense vectors of fixed size.
net = tflearn.embedding(net, input_dim=10000, output_dim=256)
# Long Short Term Memory Recurrent Layer.
# Each input would have a size of i15x128 and each of these 128
# sized vectors are fed into the LSTM layer one at a time.
# All the intermediate outputs are collected and then passed on to the second LSTM layer.
net = tflearn.lstm(net, 256, dropout=0.8)
# The output is then sent to a fully connected layer that would give us our final 11 classes
net = tflearn.fully_connected(net, 11, activation='softmax')
# We use the adam optimizer instead of standard SGD since it converges much faster
net = tflearn.regression(net, optimizer='adam', learning_rate=0.001,
                         loss='categorical_crossentropy')

# Training
model = tflearn.DNN(net, tensorboard_verbose=0)

model.load('gamemodel.tfl')

# model.fit(trainX, trainY, validation_set=(testX, testY), show_metric=True,
          # batch_size=32)

model.save('gamemodel.tfl')

print("done training")

testIdx = 663

prediction = model.predict(np.reshape(trainX[testIdx], (-1, 15)))

print(dataframe.title[testIdx])
for i in range(0,10):
    print(totalY[testIdx][i])
    print(prediction[0][i])

