import os
import numpy as np

class DataGenerator(object):

    def __init__(self, batch_size, directory, data_type):
        """
        data_type: "training", "validation", or "test"
        """
        self.question_dir = os.path.join(directory,'questions',data_type)
        self.filelist = os.listdir(self.question_dir)
        self.batch_size = batch_size
        self.cur_file = 0
        self.cur_file_sample_index = 0
        self.cur_file_content = None
        self.cur_file_indices = []
        self.init_next_file()

        metadata_dict = {}
        f = open(os.path.join(directory, 'metadata', 'metadata.txt'), 'r')
        for line in f:
            entry = line.split(':')
            metadata_dict[entry[0]] = int(entry[1])
        f.close()
        self.nb_samples_epoch = metadata_dict[data_type] 

    def get_nb_samples_epoch(self):
        return self.nb_samples_epoch

    def __iter__(self):
        return self


    def __next__(self):
        return self.next()


    def init_next_file(self):
        if self.cur_file >= len(self.filelist):
            np.random.shuffle(self.filelist)
            self.cur_file = 0
        next_fn = self.filelist[self.cur_file]
        self.cur_file += 1
        self.cur_file_content = np.load(os.path.join(self.question_dir, next_fn))
        nb_samples_in_file = self.cur_file_content['X'].shape[0]
        self.cur_file_sample_index = 0
        self.cur_file_indices = np.random.permutation(np.arange(nb_samples_in_file))


    def get_nb_samples(self, nb_samples):
        indices = \
                self.cur_file_indices[self.cur_file_sample_index:self.cur_file_sample_index + nb_samples]
        X = self.cur_file_content['X'][indices]
        Xq = self.cur_file_content['Xq'][indices]
        y = self.cur_file_content['y'][indices]
        self.cur_file_sample_index += nb_samples
        return X, Xq, y


    def next(self):
        nb_samples_in_file = self.cur_file_content['X'].shape[0]
        if nb_samples_in_file - self.cur_file_sample_index < self.batch_size:
            # We first find the number of samples left in this file
            nb_samples_from_cur_file = nb_samples_in_file - self.cur_file_sample_index
            if nb_samples_from_cur_file > 0:
                # We then get the samples from this file 
                X_cur_file, Xq_cur_file, y_cur_file = self.get_nb_samples(nb_samples_from_cur_file)
                # Move on to the next file
            self.init_next_file()
            # We compute the number of samples needed after getting the samples from the first file
            needed_samples = self.batch_size - nb_samples_from_cur_file
            X_next_file, Xq_next_file, y_next_file = self.get_nb_samples(needed_samples)
            if nb_samples_from_cur_file > 0:
                X = np.vstack((X_cur_file, X_next_file))
                Xq = np.vstack((Xq_cur_file, Xq_next_file))
                y = np.vstack((y_cur_file, y_next_file))
            else:
                X = X_next_file
                Xq = Xq_next_file
                y = y_next_file
            return [X, Xq], y
        else:
            next_batch_indices = \
                    self.cur_file_indices \
                    [self.cur_file_sample_index : \
                    self.cur_file_sample_index + self.batch_size]
            self.cur_file_sample_index += self.batch_size
            return [self.cur_file_content['X'][next_batch_indices], \
                    self.cur_file_content['Xq'][next_batch_indices]], \
                    self.cur_file_content['y'][next_batch_indices]
