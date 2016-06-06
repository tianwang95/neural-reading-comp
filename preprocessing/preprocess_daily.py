#!/usr/bin/env python 

import sys
import os
import numpy as np
module_home = os.environ['NEURAL_PATH']
sys.path.insert(0, module_home)
import rc_data

glove_dict = rc_data.glove2dict(os.path.join("datasets/glove.6B", "glove.6B.200d.txt"))
data_processor = rc_data.DataProcessor(200, 100000, word_vector_dict = glove_dict)

sources = ["datasets/full_data/daily/questions/training",
           "datasets/full_data/daily/questions/test",
           "datasets/full_data/daily/questions/validation"]
targets = ["datasets/full_data/daily_processed/questions/training",
           "datasets/full_data/daily_processed/questions/test",
           "datasets/full_data/daily_processed/questions/validation"]
metadata_directory = "datasets/full_data/daily_processed/metadata"

data_processor.do_all(sources, targets, metadata_directory, 10000)
