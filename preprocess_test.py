import rc_test
import os
import numpy as np

sources = ["toy_dataset/cnn/questions/training",
           "toy_dataset/cnn/questions/test",
           "toy_dataset/cnn/questions/validation"]
targets = ["toy_dataset/cnn_processed/questions/training",
           "toy_dataset/cnn_processed/questions/test",
           "toy_dataset/cnn_processed/questions/validation"]
metadata_directory = "toy_dataset/cnn_processed/metadata"

directory = targets[0]

f = open(os.path.join(metadata_directory, 'metadata.txt'), 'r')
input_length = int(f.readline().split(':')[1])
query_length = int(f.readline().split(':')[1])
vocab_size = int(f.readline().split(':')[1])
f.close()

for i, fn in enumerate(os.listdir(directory)):
    array = np.load(os.path.join(directory, fn))
    for sample in array[0]:
        assert(len(sample) == input_length)
    for sample in array[1]:
        assert(len(sample) == query_length)
    for sample in array[2]:
        print sample

array = np.load(os.path.join(metadata_directory, "weights.npy"))
assert(array.shape == (vocab_size + 2, 100))
