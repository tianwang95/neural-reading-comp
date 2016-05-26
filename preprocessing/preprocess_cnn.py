#!/usr/bin/env python 

import rc_data
import os
import numpy as np

glove_dict = rc_data.glove2dict(os.path.join("glove.6B", "glove.6B.200d.txt"))
data_processor = rc_data.DataProcessor(200, 100000, word_vector_dict = glove_dict)

sources = ["full_data/cnn/questions/training",
           "full_data/cnn/questions/test",
           "full_data/cnn/questions/validation"]
targets = ["full_data/cnn_processed/questions/training",
           "full_data/cnn_processed/questions/test",
           "full_data/cnn_processed/questions/validation"]
metadata_directory = "full_data/cnn_processed/metadata"

data_processor.do_all(sources, targets, metadata_directory, 10000)
