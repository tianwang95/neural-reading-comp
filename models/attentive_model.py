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
from keras.layers.core import Activation, Dense, Dropout, RepeatVector, Lambda
from keras.layers.wrappers import TimeDistributed
from keras.layers import LSTM
from custom import Reverse, masked_concat, masked_dot, masked_sum
from keras import backend as K

### MODEL

def get_model(
        data_path, #Path to dataset
        lstm_dim, #Dimension of the hidden LSTM layers
        optimizer='rmsprop', #Optimization function to be used
        loss='categorical_crossentropy' #Loss function to be used
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
#   (None, story_maxlen) 

    x = Embedding(input_dim=vocab_size+2,
                  output_dim=word_dim,
                  input_length=story_maxlen,
                  mask_zero=True,
                  weights=[embed_weights])(story_input)
#   (None, story_maxlen, word_dim)

    story_lstm_f = LSTM(lstm_dim,
                        return_sequences = True,
                        consume_less='gpu')(x)
#   (None, story_maxlen, lstm_dim)
    story_lstm_b = LSTM(lstm_dim,
                        return_sequences = True,
                        consume_less='gpu',
                        go_backwards=True)(x)
#   (None, story_maxlen, lstm_dim)

    story_lstm_b_r = Reverse()(story_lstm_b)
#   (None, story_maxlen, lstm_dim)

    yd = masked_concat([story_lstm_f, story_lstm_b_r])
#   (None, story_maxlen, 2*lstm_dim)

    query_input = Input(shape=(query_maxlen,), dtype='int32', name='QueryInput')
#   (None, query_maxlen) 

    x_q = Embedding(input_dim=vocab_size+2,
            output_dim=word_dim,
            input_length=query_maxlen,
            mask_zero=True,
            weights=[embed_weights])(query_input)
#   (None, query_maxlen, word_dim)

    query_lstm_f = LSTM(lstm_dim,
                        consume_less='gpu')(x_q)
#   (None, lstm_dim)
    query_lstm_b = LSTM(lstm_dim,
                        go_backwards=True,
                        consume_less='gpu')(x_q)
#   (None, lstm_dim)

    u = merge([query_lstm_f, query_lstm_b], mode='concat')
#   (None, 2*lstm_dim)


    u_rpt = RepeatVector(story_maxlen)(u)
#   (None, story_maxlen, 2*lstm_dim)


    story_dense = TimeDistributed(Dense(2*lstm_dim))(yd)
#   (None, story_maxlen, 2*lstm_dim)


    query_dense = TimeDistributed(Dense(2*lstm_dim))(u_rpt)
#   (None, story_maxlen, 2*lstm_dim)


    story_query_sum = masked_sum([story_dense, query_dense])
#   (None, story_maxlen, 2*lstm_dim)


    m = Activation('tanh')(story_query_sum)
#   (None, story_maxlen, 2*lstm_dim)
    s = TimeDistributed(Dense(1, activation='softmax'))(m)
#   (None, story_maxlen, 1)


    r = masked_dot([s, yd])
#   dotting (None, story_maxlen, 1) . (None, story_maxlen, 2*lstm_dim)
#   along (1,1)
#   (None, 1, 2*lstm_dim)
"""
    def flatten(x):
        return K.reshape(x, (x.shape[0], x.shape[2]))

    def flatten_output_shape(input_shape):
        return (input_shape[0], input_shape[2]) 

    r_flatten = Lambda(flatten, output_shape=flatten_output_shape)(r)
"""
    r_flatten = Flatten()(r)
#   (None, 2*lstm_dim)
    g_r = Dense(word_dim)(r_flatten)
#   (None, word_dim)
    g_u = Dense(word_dim)(u)
#   (None, word_dim)
    g_r_plus_g_u = merge([g_r, g_u], mode='sum')
#   (None, word_dim)
    g_d_q = Activation('tanh')(g_r_plus_g_u)
#   (None, word_dim)
    result = Dense(entity_dim, activation='softmax')(g_d_q)
#   (None, entity_dim)

    model = Model(input=[story_input, query_input], output=result)
    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=['accuracy'])
    print(model.summary())
    return model
