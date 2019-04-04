# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 12:31:40 2018

@author: Charlie
"""
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense, Dropout, TimeDistributed, Activation
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping, History
from keras.models import Sequential
from keras.optimizers import Adam
import keras.utils as ku 

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.python.client import device_lib

#check the status of GPU
print(device_lib.list_local_devices())

#Fix randomness for consistent result
np.random.seed(12)

#Read in the Two Towers text data line by line
corpus = []

with open('two_towers_partial_v2.txt', 'r') as f:
    for line in f.readlines():
        l = line.strip()
        #filter out empty lines
        if l == '':
            continue
        corpus.append(l)

tokenizer = Tokenizer()

tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1

#Now convert the corpus into a flat dataset of sequences
input_sequence = []
for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i + 1]
        input_sequence.append(n_gram_sequence)
"""At this point, input_sequence contains lists that contain
different length combinations of each sentence"""

#Before we proceed, we need to pad all lists to have same length
max_sequence_len = max([len(x) for x in input_sequence])
input_sequence = np.array(pad_sequences(input_sequence, maxlen = max_sequence_len, padding = 'pre'))

#Now split the data into x and y where y is corresponds to the last element in each array inside 
#input_sequence
x_train, y_train = input_sequence[:, :-1], input_sequence[:, -1]
plt.hist(y_train, bins = 100)
y_train_ohe = ku.to_categorical(y_train, num_classes = total_words)

#Model building
input_len = max_sequence_len - 1

history = History()

model = Sequential()
model.add(Embedding(total_words, 100, input_length = input_len))
model.add(LSTM(700, return_sequences = True))
model.add(Dropout(0.3))
model.add(LSTM(700, return_sequences = True))
model.add(Dropout(0.3))
model.add(LSTM(700))
model.add(Dropout(0.3))
model.add(Dense(total_words, activation = 'softmax'))

epochs = 250

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
model.fit(x_train, y_train_ohe, batch_size = 100, epochs = epochs, verbose = 1, callbacks = [history])
model.save('models/two_towers_partial_v2_epoch_250_embed_100_three_lstm_700_700_700.h5')

#Let us try and generate some text
seed_list = ["aragorn", "boromir", "legolas", "hobbit"]
seed_result = []
num_next_words = 50
#This creates a dictionary that has key of index and value of the word in tokenizer
reverse_word_map = dict(map(reversed, tokenizer.word_index.items()))
reverse_word_map[0] = " "
for seed_text in seed_list:
    for j in range(num_next_words):
        #texts_to_sequence returns a nested list, so we need [0]
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen = max_sequence_len - 1, padding = 'pre')
        predicted = model.predict_classes(token_list, verbose = 0)
        
        predicted_int = int(predicted)
        output_word = reverse_word_map[predicted_int]
        seed_text += " " + output_word
    seed_result.append(seed_text)