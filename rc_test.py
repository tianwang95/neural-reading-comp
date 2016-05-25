import os
import numpy as np
from collections import Counter
class PreProcessData:

    def __init__(self, dim, glove_home='glove.6B'):
        self.glove_dict = self.glove2dict(os.path.join(glove_home,
                                         'glove.6B.{}d.txt'.format(dim_size)))
        self.dim = dim
        self.input_length = None
        self.query_length = None
        self.word_to_idx = {}
        self.add_word('<UNK>') #add unknown word
        self.weights = None

    def add_word(self, word):
        self.word_to_idx[word] = len(self.word_to_idx) + 1

    def glove2dict(self, src_filename):
        reader = csv.reader(open(src_filename), 
                 delimeter=' ', quoting=csv.QUOTE_NONE)
        return {line[0]: np.array(list(map(float, line[1: ])), 
                dtype='float32') for line in reader}    

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

        context = lines[2]
        query = lines[4]
        answer = lines[6]

        #Process Context:
        X = []
        for word in context.split():
            if word in self.word_to_idx:
                X.append(self.word_to_idx[word])
            else:
                X.append(self.word_to_idx['<UNK>'])

        for _ in xrange(len(context), self.input_length):
            X.append(0)

        X = np.array(X)

        #Process Question
        Xq = []
        for word in query.split():
            if word in self.word_to_idx:
                Xq.append(self.word_to_idx[word])
            else:
                Xq.append(self.word_to_idx['<UNK>'])

        for _ in xrange(len(question), self.query_length):
            X.append(0)

        Xq = np.array(Xq)

        y = int(answer[len('@entity'):])

        return X, Xq, y

    def preprocess(self, directory, vocab_size = None):
        """
        will set up needed variables like input length and vocab size
        
        vocab_size: 1 + |VOCAB| for 0 indexing
        
        return: input length, question length, weight_vectors
        """
        c = Counter()
        entity_set = set()

        for i in os.listdir(directory):
            if i.endswith('.question'):
                f = open(os.path.join(directory, i), 'r')
                lines = []
                for _ in xrange(7):
                    lines.append(f.readline().strip())
                f.close()

                context = lines[2]
                query = lines[4]
                answer = lines[6]

                for word in context:
                    if word[0] == '@':
                        entity_set.add(word)
                    else:
                        c[word] += 1
                for word in question:
                    if word[0] == '@':
                        entity_set.add(word)
                    else:
                        c[word] += 1
                if answer[0] == '@':
                    entity_set.add(answer)

                #keep track of longest length
                curr_context_length = len(context)
                curr_query_length = len(query)
                if curr_context_length > self.input_length:
                    self.input_length = curr_context_length
                if curr_query_length > self.query_length:
                    self.query_length = curr_query_length

        # compute final vocab list
        for word in entity_set:
            self.add_word(word)
        num_to_add = vocab_size - len(self.word_to_idx)

        assert(num_to_add <= len(c))

        for word in c.most_common(num_to_add):
            self.add_word(word)

        assert(vocab_size = len(self.word_to_idx))

        #compute weights to return
        self.weights = np.zeros((vocab_size + 1, self.dim), dtype='float32')
        for word, idx in self.word_to_idx.iteritems():
            if word in self.glove_dict:
                self.weights[idx, :] = self.glove_dict[word]
            else:
                self.weights[idx, :] = self.random_vector()
