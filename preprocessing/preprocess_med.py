#!/usr/bin/env python 

import rc_data
import os
import numpy as np

glove_dict = None#rc_data.glove2dict(os.path.join("../datasets/glove.6B", "glove.6B.100d.txt"))
data_processor = rc_data.DataProcessor(100, 70000, word_vector_dict = glove_dict)

sources = ["../datasets/med_dataset/cnn/questions/training",
           "../datasets/med_dataset/cnn/questions/test",
           "../datasets/med_dataset/cnn/questions/validation"]
targets = ["../datasets/med_dataset/cnn_processed/questions/training",
           "../datasets/med_dataset/cnn_processed/questions/test",
           "../datasets/med_dataset/cnn_processed/questions/validation"]
metadata_directory = "../datasets/med_dataset/cnn_processed/metadata"

data_processor.do_all(sources, targets, metadata_directory, 10000)
