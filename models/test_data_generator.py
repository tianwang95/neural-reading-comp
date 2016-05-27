import os
import numpy as np
from data_generator import DataGenerator

path = "../datasets/toy_dataset/cnn_processed/"
trainData = DataGenerator(3, path, 'training')

counter = 0
X, y = trainData.next()

for i in xrange(50):
    print "--------------"
    X, y = trainData.next()
    if (X[0][0].shape != (1948,)):
        print "X[0][0] instead of (1948,)"
        print X[0][0].shape
        print "--"
    if (X[1][0].shape != (20,)):
        print "X[1][0] instead of (20,)"
        print X[1][0].shape
        print "--"
    if (y[0].shape != (367,)):
        print "y[0] instead of (367,)"
        print y[0].shape
        print "--"
    if (X[0].shape != (3,1948)):
        print "X[0] instead of (3, 1948)"
        print X[0].shape
        print "--"
    if (X[1].shape != (3,20)):
        print "X[1] instead of (3,20)"
        print X[1].shape
        print "--"
    if (y.shape != (3, 367)):
        print "y instead of (3, 367)"
        print y.shape
        print "--"
