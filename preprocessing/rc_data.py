import os
import numpy as np
import csv
from collections import Counter
import itertools as it
from threading import Thread, Lock

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
        self.threads = []
        self.lock = Lock()
        self.max_entity_id = 0
        self.entity_set = set()

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

        y = np.zeros((self.max_entity_id + 1,))

        y[int(answer[len('@entity'):])] = 1

        return X, Xq, y

    def get_file_lengths(self, directory, fn):
        f = open(os.path.join(directory, fn), 'r')
        lines = []
        for line in f:
            lines.append(line.strip())
        f.close()

        context = lines[2]
        query = lines[4]
        entities = lines[8:]

        for entity in entities:
            entity_str = entity.split(':')[0]
            self.lock.acquire()
            self.entity_set.add(entity_str)
            entity_id = int(entity_str[len('@entity'):])
            if entity_id > self.max_entity_id:
                self.max_entity_id = entity_id
            self.lock.release()

        curr_context_length = context.count(' ') + 1
        curr_query_length = query.count(' ') + 1
        self.lock.acquire()
        if curr_context_length > self.input_length:
            self.input_length = curr_context_length
        if curr_query_length > self.query_length:
            self.query_length = curr_query_length
        self.lock.release()

    def get_lengths(self, directories):
        """
        directories: list of paths to directories with questions

        returns input_length, query_length
        """
        for directory in directories:
            for i in os.listdir(directory):
                if i.endswith('.question'):
                    t = Thread(target=self.get_file_lengths, args=(directory, i))
                    t.start()
                    self.threads.append(t)

        for t in self.threads:
            t.join()

    def get_file_vocab(self, train_directory, fn, c):
        f = open(os.path.join(train_directory, fn), 'r')
        lines = []
        for _ in xrange(7):
            lines.append(f.readline().strip())
        f.close()

        context = lines[2]
        query = lines[4]
        answer = lines[6]

        for word in context.split():
            if word[0] != '@':
                self.lock.acquire()
                c[word] += 1
                self.lock.release()
        for word in query:
            if word[0] != '@':
                self.lock.acquire()
                c[word] += 1
                self.lock.release()

    def set_vocab(self, train_directory):
        """
        train_directory: path to training data
        vocab_size = number of unique words including <UNK>
        """
        c = Counter()

        for i in os.listdir(train_directory):
            if i.endswith('.question'):
                t = Thread(target=self.get_file_vocab, args=(train_directory, i, c))
                t.start()
                self.threads.append(t)

        for t in self.threads:
            t.join()

        # compute final vocab list
        for word in self.entity_set:
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

    def make_batch(self, filenames, source, target, batch_idx):
        batch_X, batch_Xq, batch_y = [], [], []
        for fn in filenames:
            if fn.endswith('.question'):
                X, Xq, y = self.to_idx_doc_question(os.path.join(source, fn))
                batch_X.append(X)
                batch_Xq.append(Xq)
                batch_y.append(y)
        self.save_batch(batch_X, batch_Xq, batch_y, target, batch_idx)


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
            all_files = os.listdir(source)
            batch_file_lists = [all_files[x:x+batch_size] for x in xrange(0, len(all_files), batch_size)]
            for i, batch_file_list in enumerate(batch_file_lists):
                t = Thread(target=self.make_batch, args=(batch_file_list, source, target, i)) 
                t.start()
                self.threads.append(t)
            '''
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
            '''

    def save_batch(self, batch_X, batch_Xq, batch_y, target, num):
        X = np.array(batch_X)
        Xq = np.array(batch_Xq)
        y = np.array(batch_y)
        np.savez(os.path.join(target, "batch{}".format(num)),
                X=X, Xq=Xq, y=y)

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

        f = open(os.path.join(metadata_directory, 'metadata.txt'), 'w+')
        f.write("input_length:{}\n".format(self.input_length))
        f.write("query_length:{}\n".format(self.query_length))
        f.write("vocab_size:{}\n".format(len(self.word_to_idx)))
        f.write("entity_dim:{}\n".format(self.max_entity_id + 1))
        f.close()

        if self.word_vector:
            np.save(os.path.join(metadata_directory, 'weights'), self.weights)
        for t in self.threads:
            t.join()
        b = time.time()
        print "batches done! - {}s".format(b-a)

    def get_idx_to_word(self):
        return {v: k for k, v in self.word_to_idx.iteritems()}
