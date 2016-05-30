#!/usr/bin/env python 

import rc_data
import os
import numpy as np

glove_dict = rc_data.glove2dict(os.path.join("../datasets/glove.6B", "glove.6B.50d.txt"))
data_processor = rc_data.DataProcessor(50, 3000, word_vector_dict = glove_dict)

sources = ["../datasets/sml_dataset/cnn/questions/training",
           "../datasets/sml_dataset/cnn/questions/test",
           "../datasets/sml_dataset/cnn/questions/validation"]
targets = ["../datasets/sml_dataset/cnn_processed/questions/training",
           "../datasets/sml_dataset/cnn_processed/questions/test",
           "../datasets/sml_dataset/cnn_processed/questions/validation"]
metadata_directory = "../datasets/sml_dataset/cnn_processed/metadata"

data_processor.do_all(sources, targets, metadata_directory, 50)
