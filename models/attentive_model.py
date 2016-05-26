"""
Attentive LSTM Reader Model
---------------------------

At a high level, this model reads both the story and the question forwards and backwards, and represents the document as a weighted sum of its token where each individual token weight is decided by an attention mechanism that reads the question.
"""

import os
import numpy as np
from keras.models import Sequential
from keras.models import Model
from keras.engine import Input, Merge, merge
from keras.layers.embeddings import Embedding
from keras.layers.core import Activation, Dense, Dropout, RepeatVector, Flatten
from keras.layers.wrappers import TimeDistributed
from keras.layers import LSTM

### MODEL

def get_model(
        data_path='datasets/toy_dataset/cnn_processed', #Path to dataset
        lstm_dim=32, #Dimension of the hidden LSTM layers
        optimizer='rmsprop', #Optimization function to be used
        loss='sparse_categorical_crossentropy' #Loss function to be used
        ):

    metadata_dict = {}
    f = open(os.path.join(data_path, 'metadata', 'metadata.txt'), 'r')
    for line in f:
        entry = line.split(':')
        metadata_dict[entry[0]] = int(entry[1])
    f.close()
    story_maxlen = metadata_dict['input_length']
    query_maxlen = metadata_dict['query_length']
    vocab_size = metadata_dict['vocab_size']
    entity_dim = metadata_dict['entity_dim']

    embed_weights = np.load(os.path.join(data_path, 'metadata', 'weights.npy'))
    word_dim = embed_weights.shape[1]

########## MODEL ############

    story_input = Input(shape=(story_maxlen,), dtype='int32', name="StoryInput")

    x = Embedding(input_dim=vocab_size+2,
                  output_dim=word_dim,
                  input_length=story_maxlen,
                  weights=[embed_weights])(story_input)

    story_lstm_f = LSTM(lstm_dim,
                        return_sequences = True,
                        consume_less='gpu')(x)
    story_lstm_b = LSTM(lstm_dim,
                        return_sequences = True,
                        consume_less='gpu',
                        go_backwards=True)(x)

    yd = merge([story_lstm_f, story_lstm_b], mode='concat')

    query_input = Input(shape=(query_maxlen,), dtype='int32', name='QueryInput')

    x_q = Embedding(input_dim=vocab_size+2,
            output_dim=word_dim,
            input_length=query_maxlen,
            weights=[embed_weights])(query_input)

    query_lstm_f = LSTM(lstm_dim,
                        consume_less='gpu')(x_q)
    query_lstm_b = LSTM(lstm_dim,
                        go_backwards=True,
                        consume_less='gpu')(x_q)

    u = merge([query_lstm_f, query_lstm_b], mode='concat')
# do i need masking for this next layer?
    u_rpt = RepeatVector(story_maxlen)(u)

    story_dense = TimeDistributed(Dense(2*lstm_dim))(yd)
    query_dense = TimeDistributed(Dense(2*lstm_dim))(u_rpt)
    story_query_sum = merge([story_dense, query_dense], mode='sum')
    m = Activation('tanh')(story_query_sum)
    s = TimeDistributed(Dense(1, activation='softmax'))(m)
#s-shape = (nb_samples, slen, 1)
#story_merged = (nb_samples, slen, 2hdim)
    r = merge([s, yd], mode='dot', dot_axes=(1,1))
#r-shape = (nb_samples, 1, 2hdim)
    r_flatten = Flatten()(r)
    g_r = Dense(word_dim)(r_flatten)
    g_u = Dense(word_dim)(u)
    g_r_plus_g_u = merge([g_r, g_u], mode='sum')
    g_d_q = Activation('tanh')(g_r_plus_g_u)
#g_d_q-shape (nb_samples, 1, word_dim)
    result = Dense(entity_dim, activation='softmax')(g_d_q)

    model = Model(input=[story_input, query_input], output=result)
    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=['accuracy'])
    return model
