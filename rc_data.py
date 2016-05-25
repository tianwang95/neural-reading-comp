import os
import numpy as np
import csv
from collections import Counter
import itertools as it

import time

def glove2dict(src_filename):
    reader = csv.reader(open(src_filename),
             delimiter=' ', quoting=csv.QUOTE_NONE)
    return {line[0]: np.array(list(map(float, line[1: ])),
            dtype='float32') for line in reader}

class DataProcessor:

    def __init__(self, dim, vocab_size, word_vector_dict = None):
        self.word_vector = word_vector_dict
        self.dim = dim
        self.input_length = None
        self.query_length = None
        self.weights = None
        self.vocab_size = vocab_size

        self.word_to_idx = {}
        self.add_word('<UNK>') #add unknown word

    def add_word(self, word):
        self.word_to_idx[word] = len(self.word_to_idx) + 1

    def set_word_vector(word_vector_dict):
        self.word_vector = word_vector_dict

    def random_vector(self):
        return np.random.uniform(low=-0.05, high=0.05, size=(self.dim,)).astype('float32')

    def to_idx_doc_question(self, filename):
        """
        returns integer scalar indices for words to be passed into
        and embedding layer

        this pads the input to the self.input_length, length of longest input
        """

        f = open(filename, 'r')
        lines = []
        for _ in xrange(7):
            lines.append(f.readline().strip())
        f.close()

        context = lines[2].split()
        query = lines[4].split()
        answer = lines[6]

        #Process Context:
        X = []
        for word in context:
            if word in self.word_to_idx:
                X.append(self.word_to_idx[word])
            else:
                X.append(self.word_to_idx['<UNK>'])

        for _ in xrange(len(context), self.input_length):
            X.append(0)

        X = np.array(X)

        #Process Question
        Xq = []
        for word in query:
            if word in self.word_to_idx:
                Xq.append(self.word_to_idx[word])
            else:
                Xq.append(self.word_to_idx['<UNK>'])

        for _ in xrange(len(query), self.query_length):
            Xq.append(0)

        Xq = np.array(Xq)

        y = int(answer[len('@entity'):])

        return X, Xq, y

    def get_lengths(self, directories):
        """
        directories: list of paths to directories with questions

        returns input_length, query_length
        """
        for directory in directories:
            for i in os.listdir(directory):
                if i.endswith('.question'):
                    f = open(os.path.join(directory, i), 'r')
                    lines = []
                    for _ in xrange(5):
                        lines.append(f.readline().strip())
                    f.close()

                    context = lines[2]
                    query = lines[4]

                    curr_context_length = context.count(' ') + 1
                    curr_query_length = query.count(' ') + 1
                    if curr_context_length > self.input_length:
                        self.input_length = curr_context_length
                    if curr_query_length > self.query_length:
                        self.query_length = curr_query_length

        return self.input_length, self.query_length

    def set_vocab(self, train_directory):
        """
        train_directory: path to training data
        vocab_size = number of unique words including <UNK>
        """
        c = Counter()
        entity_set = set()

        for i in os.listdir(train_directory):
            if i.endswith('.question'):
                f = open(os.path.join(train_directory, i), 'r')
                lines = []
                for _ in xrange(7):
                    lines.append(f.readline().strip())
                f.close()

                context = lines[2]
                query = lines[4]
                answer = lines[6]

                for word in context.split():
                    if word[0] == '@':
                        entity_set.add(word)
                    else:
                        c[word] += 1
                for word in query:
                    if word[0] == '@':
                        entity_set.add(word)
                    else:
                        c[word] += 1
                if answer[0] == '@':
                    entity_set.add(answer)

        # compute final vocab list
        for word in entity_set:
            self.add_word(word)
        num_to_add = self.vocab_size - len(self.word_to_idx)

        assert(num_to_add <= len(c))

        for word, _ in c.most_common(num_to_add):
            self.add_word(word)

        assert(self.vocab_size == len(self.word_to_idx))

        #compute weights
        if self.word_vector:
            #vocab_size + 2 due to masking, see keras
            self.weights = np.zeros((self.vocab_size + 2, self.dim), dtype='float32')
            for word, idx in self.word_to_idx.iteritems():
                if word in self.word_vector:
                    self.weights[idx, :] = self.word_vector[word]
                else:
                    self.weights[idx, :] = self.random_vector()

    def get_weights(self):
        return self.weights

    def generate_batch_files(self, sources, targets, batch_size):
        """
        save np arrays of with batch_size samples into a file for
        entire source directory. files will be .npy

        sources: list of paths to source directories
        targets: same thing but for targets

        files will be np.array([X, Xq, y])
        """
        assert(len(sources) == len(targets))

        for source, target in it.izip(sources, targets):
            counter = 0
            batch_X, batch_Xq, batch_y = [], [], []
            for fn in os.listdir(source):
                if fn.endswith('.question'):
                    counter += 1
                    X, Xq, y = self.to_idx_doc_question(os.path.join(source, fn))
                    batch_X.append(X)
                    batch_Xq.append(Xq)
                    batch_y.append(y)
                    if counter % batch_size == 0:
                        self.save_batch(batch_X, batch_Xq, batch_y, target, counter)
                        batch_X, batch_Xq, batch_y = [], [], []

            if counter % batch_size != 0:
                self.save_batch(batch_X, batch_Xq, batch_y, target, counter)

    def save_batch(self, batch_X, batch_Xq, batch_y, target, num):
        x_arr = np.array(batch_X)
        xq_arr = np.array(batch_Xq)
        y_arr = np.array(batch_y)
        new = [x_arr, xq_arr, y_arr]

        np.save(os.path.join(target, "batch{}".format(num)), new)

    def do_all(self, sources, targets, metadata_directory, batch_size):
        """
        sources[0] must be training!
        targets[0] must be training!
        """
        a = time.time()
        self.get_lengths(sources)
        b = time.time()
        print "lengths done! - {}s".format(b-a)

        a = time.time()
        self.set_vocab(sources[0])
        b = time.time()
        print "vocab set! - {}s".format(b-a)

        a = time.time()
        self.generate_batch_files(sources, targets, batch_size)
        b = time.time()
        print "batches done! - {}s".format(b-a)

        f = open(os.path.join(metadata_directory, 'metadata.txt'), 'w')
        f.write("input_length:{}\n".format(self.input_length))
        f.write("query_length:{}\n".format(self.query_length))
        f.write("vocab_size:{}\n".format(len(self.word_to_idx)))
        f.close()

        if self.word_vector:
            np.save(os.path.join(metadata_directory, 'weights'), self.weights)

    def get_idx_to_word(self):
        return {v: k for k, v in self.word_to_idx.iteritems()}


def rc_data_generator(directory):
    """
    return ([X, Xq], y)
    """
    filelist = os.listdir(directory)
    for fn in filelist:
        array = np.load(os.path.join(directory, fn))
        yield [array[0], array[1]], array[2]
