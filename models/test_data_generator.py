import os
import numpy as np
from data_generator import DataGenerator

path = "../datasets/toy_dataset/cnn_processed/"
trainData = DataGenerator(3, path, 'training')

counter = 0
X, y = trainData.next()

for i in xrange(50):
    print i
    X, y = trainData.next()
    if (X[0][0].shape != (1948,)):
        print "X[0][0]"
        print X[0][0].shape
    if (X[1][0].shape != (20,)):
        print "X[1][0]"
        print X[1][0].shape
    if (y[0].shape != (367,)):
        print "y[0]"
        print y[0].shape

    if (X[0].shape != (3,1948)):
        print "X[0]"
        print X[0].shape
    if (X[1].shape != (3,20)):
        print "X[1]"
        print X[1].shape
    if (y.shape != (3, 367)):
        print "y"
        print y.shape
