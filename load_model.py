# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 23:39:23 2018

@author: Charlie
"""

"""This script is used for further training saved models"""
import re
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense, Dropout, TimeDistributed, Activation
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping, History
from keras.models import Sequential, load_model
from keras.optimizers import Adam
import keras.utils as ku 

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.python.client import device_lib

#check the status of GPU
print(device_lib.list_local_devices())

#Fix randomness for consistent result
np.random.seed(12)

"""Build corpus here"""
corpus = ""

with open('two_towers_partial_v2.txt', 'r') as f:
    for line in f.readlines():
        l = line.strip().lower()
        #filter out empty lines
        if l == '':
            continue
        corpus += l + " "

"""https://stackoverflow.com/questions/367155/splitting-a-string-into-words-and-punctuation"""
corpus_mod = re.findall(r"[\w']+|[.,‘’!?;]", corpus)

"""Remove all numbers since they most likely don't contribute"""
for word in corpus_mod:
    if (word.isdigit()):
        corpus_mod.remove(word)

"""Apply Tokenizer"""
tokenizer = Tokenizer(filters='"#$%&()*+-/:;<=>@[\]^_`{|}~')

tokenizer.fit_on_texts(corpus_mod)
total_words = len(tokenizer.word_index) + 1

"""This is index to word mapping dictionary that will be used later"""
reverse_word_map = dict(map(reversed, tokenizer.word_index.items()))

"""Generate subset of data that are less than n sequences. If max_n, we then
diseect the data starting from pair of words to 5 gram sequences"""
input_sequence = []
max_n = 6
for i in range(2, max_n + 1):
    for j in range(len(corpus_mod) - max_n + 1):
        temp_list = corpus_mod[j:j+i]
        token_list = tokenizer.texts_to_sequences([temp_list])[0]
        input_sequence.append(token_list)
  
"""Pad all the inputs adjusted the max_n for consistent input size"""      
input_sequence = np.array(pad_sequences(input_sequence, maxlen = max_n, padding = 'pre'))

"""Divide into x_train and y_train and apply ohe to y_train"""
x_train, y_train = input_sequence[:, :-1], input_sequence[:, -1]

"""Let us see how well the labels are distributed"""
#plt.hist(y_train, bins = 100)

y_train_ohe = ku.to_categorical(y_train, num_classes = total_words)

epochs = 250

history = History()

model = load_model('models/two_towers_partial_v2_epoch_250_embed_128_three_lstm_700_700_700_new_preprocess.h5')
model.fit(x_train, y_train_ohe, batch_size = 4000, epochs = epochs, verbose = 1, callbacks = [history])
model.save('models/two_towers_partial_v2_epoch_500_embed_128_three_lstm_700_700_700_new_preprocess.h5')

#Let us try and generate some text
seed_list = ["aragorn ran towards", "boromir tried his", "legolas looked on", "hobbits were taken", "gandalf called"]
seed_result = []
num_next_words = 50
#This creates a dictionary that has key of index and value of the word in tokenizer
reverse_word_map = dict(map(reversed, tokenizer.word_index.items()))
reverse_word_map[0] = " "
for seed_text in seed_list:
    for j in range(num_next_words):
        #texts_to_sequence returns a nested list, so we need [0]
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen = max_n - 1, padding = 'pre')
        predicted = model.predict_classes(token_list, verbose = 0)
        
        predicted_int = int(predicted)
        output_word = reverse_word_map[predicted_int]
        seed_text += " " + output_word
    seed_result.append(seed_text)