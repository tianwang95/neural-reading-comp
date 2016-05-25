import os
import numpy as np
from collections import Counter
class PreProcessData:

    def __init__(self, dim_size, glove_home='glove.6B'):
        self.glove_dict = self.glove2dict(os.path.join(glove_home,
                                         'glove.6B.{}d.txt'.format(dim_size)))
        self.entity_dict = {}
        self.placeholder_vec = None
        self.vec_shape = (dim_size,)
        self.input_length = None
        self.query_length = None

        #if using an embedding layer
        self.word_to_idx = {}
        self.add_word('<UNK>') #add unknown word

    def add_word(self, word):
        self.word_to_idx[word] = len(self.word_to_idx) + 1

    def glove2dict(self, src_filename):
        reader = csv.reader(open(src_filename), 
                 delimeter=' ', quoting=csv.QUOTE_NONE)
        return {line[0]: np.array(list(map(float, line[1: ])), 
                dtype='float32') for line in reader}
    
    def new_entity_vec(self, entity_id):
        self.entity_dict[entity_id] = np.random.rand(self.vec_shape). \
                                         astype('float32')

    def zero_pad(self):
        return np.zero(self.vec_shape, dtype='float32')

    def random_vector(self):
        return np.random.rand(self.vec_shape).astype('float32')

    def embed_doc_question(self, filename):
        """
        filename: name of question file that contains context

        returns: {'X'=embeded document and question, 'y'=answer}
        """
        assert(glove_dict != None)
        assert(input_length != None)
        assert(question_length != None)
        assert(placeholder_vec != None)

        f = open(filename, 'r')
        lines = []
        for line in f:
            lines.append(line.strip())

        f.close()

        context = lines[2]
        query = lines[4]
        answer = lines[6]
        entities = lines[8:]

        #Process entities:
        for entity_str in entities:
            entity_id = int(word[len('@entity'):word.index(':')])
            if entity_id not in self.entity_dict:
                self.new_entity_vec(entity_id)

        #Process Story
        X = []
        for word in context.split():
            if word[0] == '@': #it's an entity:
                entity_id = int(word[len('@entity'):])
                vec = self.entity_dict[entity_id]
            else:
                if word in self.glove_dict:
                    vec = self.glove_dict[word]
                else:
                    vec = self.random_vector()
            X.append(vec)

        for _ in xrange(len(context), self.input_length):
            X.append(self.zero_pad())

        X = np.array(X)
        assert(X.dtype='float32') #make sure we are float32 for GPU

        #Process Question
        Xq = []
        for word in query.split():
            if word[0] == '@': #entity or placeholder
                if word[1] = 'p': #placeholder
                    vec = self.placeholder_vec
                else: #entity
                    vec = self.entity_dict[entity_id]
            else:
                if word in self.glove_dict:
                    vec = self.glove_dict[word]
                else:
                    vec = self.random_vector()
            Xq.append(vec)

        for _ in xrange(len(query), self.query_length):
            Xq.append(self.zero_pad())

        Xq = np.array(Xq)
        assert(Xq.dtype = 'float32')

        y = int(answer[len('@entity'):])

        return X, Xq, y

    def to_idx_doc_question(self, filename):
        """
        returns integer scalar indices for words to be passed into
        and embedding layer
        """

        f = open(filename, 'r')
        lines = []
        for line in f:
            lines.append(line.strip())
        f.close()

        context = lines[2]
        query = lines[4]
        answer = lines[6]
        entities = lines[8:]

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

    def preprocess(self, directory, embed=True, vocab_size = None):
        """
        will set up needed variables like input length and vocab size
        
        return: input length, question length, weight_vectors
        """
        c = Counter()
        entity_set = set()

        vocab_size -= 1
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

                if not embed:
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
                curr_context_length = context.count(' ') + 1
                curr_query_length = context.count(' ') + 1
                if curr_context_length > self.input_length:
                    self.input_length = curr_context_length
                if curr_query_length > self.query_length:
                    self.query_length = curr_query_length

        # compute vectors
