"""
Attentive LSTM Reader Model
---------------------------

At a high level, this model reads both the story and the question forwards and backwards, and represents the document as a weighted sum of its token where each individual token weight is decided by an attention mechanism that reads the question.

At an implementation layer, we have two LSTM's that read the story (one forwards and one backwards) which then get merged by concatenation. These LSTM's return sequences so that the output of each timestep of this layer is an input to the next attention layer.

Similarly,  the question is read by two LSTMs that get concatenated, but this time we don't return sequences, we use the final output of each one.
"""

import rc_data
import os
import numpy as np
from keras.models import Sequential
from keras.models import Model
from keras.layers.embeddings import Embedding
from keras.layers.core import Activation, Dense, Merge, Dropout, RepeatVector
from keras.layers.wrappers import TimeDistributed
from keras.layers import LSTM

DATA_PATH = 'toy_dataset/cnn_processed/'

f = open(os.path.join(DATA_PATH, 'metadata', 'metadata.txt'), 'w')
input_maxlen = int(f.readline().split(':')[1])
query_maxlen = int(f.readline().split(':')[1])
vocab_size = int(f.readline().split(':')[1])
f.close()

embed_weights = np.load(os.path.join(DATA_PATH, 'metadata', 'weights.npy'))
word_dim = weights.shape[1]
lstm_dim = 32

### MODEL
story_input = Input(shape(input_maxlen,), dtype='float32', name="StoryInput")

x = Embedding(input_dim=vocab_size+2,
              output_dim=word_dim,
              input_length=input_maxlen,
              mask_zero=True
              weights=embed_weights)(story_input)

story_lstm_f = LSTM(lstm_dim,
                    return_sequences = True,
                    consume_less='gpu')(x)
story_lstm_b = LSTM(lstm_dim,
                    return_sequences = True,
                    consume_less='gpu',
                    go_backwards=True)(x)

yd = merge([story_lstm_f, story_lstm_b], mode='concat')

query_encoder = Input(shape(query_maxlen,), dtype='float32', name='QueryInput')

x_q = Embedding(input_dim=vocab_size+2,
        output_dim=word_dim,
        input_length=query_maxlen,
        mask_zero=True,
        weights=embed_weights)(query_encoder)

query_lstm_f = LSTM(lstm_dim,
                    consume_less='gpu')(x_q)
query_lstm_b = LSTM(lstm_dim,
                    go_backwards=True,
                    consume_less='gpu')(x_q)

u = merge([query_lstm_f, query_lstm_b], mode='concat')
# do i need masking for this next layer?
u_rpt = RepeatVector(query_maxlen)(query_merged)

story_dense = TimeDistributed(Dense(2*lstm_dim))(yd)
query_dense = TimeDistributed(Dense(2*lstm_dim))(u_rpt)
story_query_sum = merge([story_dense, query_dense], mode='sum')
m = Activation('tanh')(story_query_sum)
s = TimeDistributed(Dense(1, activation='softmax'))(s_q_sum_tanh)
#s-shape = (nb_samples, slen, 1)
#story_merged = (nb_samples, slen, 2hdim)
r = merge([s, story_merged], mode='dot', dot_axes=(1,1))
#r-shape = (nb_samples, 1, 2hdim)
g_r = TimeDistributed(Dense(2*lstm_dim))(r)
g_u = TimeDistributed(Dense(2*lstm_dim))(u)
g_r_plus_g_u = merge([g_r, g_u], mode='sum')
g_d_q = Activation('tanh')
#g_d_q-shape (nb_samples, 1, 2hdim)
