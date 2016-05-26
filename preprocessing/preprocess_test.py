import rc_data
import os
import numpy as np

sources = ["datasets/toy_dataset/cnn/questions/training",
           "datasets/toy_dataset/cnn/questions/test",
           "datasets/toy_dataset/cnn/questions/validation"]
targets = ["datasets/toy_dataset/cnn_processed/questions/training",
           "datasets/toy_dataset/cnn_processed/questions/test",
           "datasets/toy_dataset/cnn_processed/questions/validation"]
metadata_directory = "datasets/toy_dataset/cnn_processed/metadata"

directory = targets[0]

f = open(os.path.join(metadata_directory, 'metadata.txt'), 'r')
input_length = int(f.readline().split(':')[1])
query_length = int(f.readline().split(':')[1])
vocab_size = int(f.readline().split(':')[1])
f.close()

for i, fn in enumerate(os.listdir(directory)):
    arrays = np.load(os.path.join(directory, fn))
    for sample in arrays['X']:
        assert(len(sample) == input_length)
    for sample in arrays['Xq']:
        assert(len(sample) == query_length)
    for sample in arrays['y']:
        print sample

array = np.load(os.path.join(metadata_directory, "weights.npy"))
assert(array.shape == (vocab_size + 2, 100))
