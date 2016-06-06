#!/usr/bin/python

#Example error analysis script

import sys
import os
import getopt
module_home = os.environ['NEURAL_PATH']
sys.path.insert(0, module_home)
from error_analyzer import ErrorAnalyzer
from models import attentive_model


def get_opts(argv):
    model = attentive_model
    dataset = os.path.join(module_home, 'datasets/toy_dataset/cnn_processed')
    weights_path = os.path.join(module_home, 'results/att_model/att_model.00-2.08.hdf5')
    hidden_dim = 128
    generator_set = 'validation' 

    num_errors_to_show = 10
    num_correct_to_show = 5

    try:
        opts, args = getopt.getopt(argv,"hm:d:w:l:g:e:c:", ["help", "model=", "dataset=", "weights=", "lstm_dim=", "generator_set=", "errors=", "corrects="])
    except getopt.GetoptError:
        print 'error_analysis.py -m <model> -d <path_to_dataset> -w <path_to_initial_weights> -l <LSTM dimension> -g <training|test|validation> -e <number_errors_to_show> -c <number corrects to show>'
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print 'error_analysis.py -m <model> -d <path_to_dataset> -w <path_to_initial_weights> -l <LSTM dimension> -g <training|test|validation> -e <number_errors_to_show> -c <number corrects to show>'
            sys.exit()
        elif opt in ('-m', '--model'):
            if arg == "attentive":
                model = attentive_model
            elif arg == "simple":
                model = simple_model
        elif opt in ('-d', '--dataset'):
            dataset = arg
        elif opt in ('-w', '--weights'):
            weights_path = arg
        elif opt in ('-l', '--lstm_dim'):
            hidden_dim = arg
        elif opt in ('-g', '--generator_set'):
            generator_set = arg
        elif opt in ('-e', '--errors'):
            num_errors_to_show = arg
        elif opt in ('-c', '--corrects'):
            num_correct_to_show = arg

    return model, dataset, weights_path, hidden_dim, generator_set, num_errors_to_show, num_correct_to_show


def main(argv):
    model, dataset, weights_path, hidden_dim, generator_set, num_errors_to_show, num_correct_to_show = get_opts(argv)

    analyzer = ErrorAnalyzer(
          model=model, 
          dataset=dataset,
          weights_path=weights_path,
          hidden_dim=hidden_dim,
          generator_set=generator_set
          )

    print "-----------------------------Error analysis begins-------------------------------"
    print "------Getting " + str(num_correct_to_show) + " correct samples------"

    for i in xrange(num_correct_to_show):
        sample = analyzer.get_correct_sample()
        print "---Correct sample (" + str(i) + "):"
        print "Document:"
        print sample['document']
        print "Question:"
        print sample['question']
        print "Prediction:"
        print sample['prediction']
        print "Correct Answer:"
        print sample['answer']

    print "------Getting " + str(num_errors_to_show) + " error samples------"

    for i in xrange(num_errors_to_show):
        sample = analyzer.get_error_sample()
        print "---Error sample (" + str(i) + "):"
        print "Document:"
        print sample['document']
        print "Question:"
        print sample['question']
        print "Prediction:"
        print sample['prediction']
        print "Correct Answer:"
        print sample['answer']

if __name__ == '__main__':
    main(sys.argv)
