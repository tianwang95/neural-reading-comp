import rc_data
import os
import numpy as np

glove_dict = None #rc_data.glove2dict(os.path.join("glove.6B", "glove.6B.100d.txt"))
data_processor = rc_data.DataProcessor(100, 3000, word_vector_dict = glove_dict)

sources = ["toy_dataset/cnn/questions/training",
           "toy_dataset/cnn/questions/test",
           "toy_dataset/cnn/questions/validation"]
targets = ["toy_dataset/cnn_processed/questions/training",
           "toy_dataset/cnn_processed/questions/test",
           "toy_dataset/cnn_processed/questions/validation"]
metadata_directory = "toy_dataset/cnn_processed/metadata"

data_processor.do_all(sources, targets, metadata_directory, 4)
