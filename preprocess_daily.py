#!/usr/bin/env python 

import rc_data
import os
import numpy as np

glove_dict = rc_data.glove2dict(os.path.join("glove.6B", "glove.6B.200d.txt"))
data_processor = rc_data.DataProcessor(200, 100000, word_vector_dict = glove_dict)

sources = ["full_data/daily/questions/training",
           "full_data/daily/questions/test",
           "full_data/daily/questions/validation"]
targets = ["full_data/daily_processed/questions/training",
           "full_data/daily_processed/questions/test",
           "full_data/daily_processed/questions/validation"]
metadata_directory = "full_data/daily_processed/metadata"

data_processor.do_all(sources, targets, metadata_directory, 10000)
