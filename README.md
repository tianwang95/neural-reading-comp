# Neural Reading Comprehension

This repository is an implementation of ["Teaching Machines to Read and Comprehend" by Hermann et al. (2015)](http://arxiv.org/abs/1506.03340). It comes with the following components:

* Data Preprocessing
* Continuous Random Data Generator
* Neural Networks Implemented in Keras
* Error Analysis 

It works with datasets created with the Google [rc-data](https://github.com/deepmind/rc-data) corpus generrator presented in Herman et al. 2015. The deep learning components are implemented using [Keras](https://keras.io) with [Theano](http://deeplearning.net/software/theano/) backend.

## Authors

* [Kerem Goksel](https://github.com/bkgoksel)
* [Ishan Somshekar](https://github.com/ishansomshekar)
* [Tian Wang](https://github.com/tianwang95)

## Requirements

* Python 2.7
* Numpy 
* Theano 0.8.2
* Keras 1.0.3

## Installation

Clone the repository locally, and then set the NEURAL_PATH environment variable to the absolute path to wehere the repository lives. The repository doesn't include the dataset, so it needs to be placed in the `datasets/` directory. Instructions for obtaining the dataset are on [rc-data](https://github.com/deepmind/rc-data). The repository currently expects the unpacked CNN data files to exist in `datasets/full_dataset/cnn` directory before preprocessing can be run.

## Data Preprocessing

The `rc-data` repository already serves data in a nice preprocessed format (train/test/validation splits, document-question-answer triplets in file) but further preprocessing is needed before the data can be fed into a neural network. The preprocessing module works on the `rc-data` data files and generates a numericaal representation of the dataset. To preprocess the data, do the following:

1. Make sure your dataset is inside `datasets` folder. For example, `datasets/<dataset_name>/cnn`
2. Create the output directory structure i.e. `datasets/<dataset_name>/cnn_processed/metadata`
3. Run the corresponding preprocessing script. Preprocessing scripts can be found in `/preprocessing`, you can copy and modify these scripts for datasets of different sizes. Vocabulary sizes may also be configured in these scripts as well. The scripts support multithreading, but so far it looks like multithreading has a negative effect on performance.

The preprocessing scripts batch together samples and represent them as stacks of numpy arrays. They also create `/metadata/metadata.txt` which holds information like vocab size, maximum document length and maximum question length for the dataset.

## Continuous Random Data Generator

The DataGenerator(found in `models/data_generator.py`) implements a Python iterator  that can iterate over any given preprocessed data set and provide batches of shuffled samples. Bathes can be of any size, the iterator guarantees to provide a random shuffling of the data in any case. It supports iterating infinitely over the dataset and simply reshuffles and repeats over the dataset once it is completely used.

## Neural Networks Implemented on Keras

The repo comes with two main deep learning models. 
The *simple model* is a bidirectional LSTM that runs over a concatenation of the question and the document, and feeds into a softmax activation layer. It is the baseline model that achieves around 45% validation accuracy on the full CNN dataset.
The *attentive model* is the implementation of the *attentive reader* introduced in [Hermann et al. 2015](http://arxiv.org/abs/1506.03340). It runs separate bidirectional LSTM's over the question and the document, and computes attention over each individual token of the document before merging the layers to final Dense and softmax activation layers. It achieves around 65% validation accuracy on the full CNN dataset. This is slightly higher than the accuracy reported in the original paper, likely due to the fact that we initialize our model with pretrained [GLoVe vectors](http://nlp.stanford.edu/pubs/glove.pdf).
The model implementations can be found in `models/simple_model.py` and `models/attentive_model.py`. Both models support the same API where a call to `get_model` returns a Keras model instance that can be used to call train and evaluate functions. The model files just implement the model architecture, the actual calls to training and testing are done within the running scripts in the `scripts` directory.
We had to implement several custom Keras layers to support the architecture required for these neural networks. These custom layers may be found in the `custom/` directory. The most important ones are several Merge layers that support Masking. Standard Keras merge layers don't support masking, whereas our document and question inputs need to have masking since not all the stories and questions are the same length. The custom Merge layers we implemented let us work with these masked layers, but they are not robust enough to be used as full merge layers that support masking. 

## Error Analysis

The neural network running scripts can be configured to save the trained weights after each epoch. These trained weights should be saved to the `results/` directory. Given a trained network with saved weights, the error analysis script can print out samples of document-question pairs where the model answers correctly and wrong. The error analysis script can be found in `error_analysis/error_analysis.py` Run `./error_analysis.py -h` for more details on the usage of the error analysis script.  

## Future work

A potential improvement for the attentive model where the weights for word embeddings of the question and the document get trained together as opposed to separately as they are now.

Support for visualization of intermediate layer outputs. This will be especially useful for visualizing the attention of the attentive model.
