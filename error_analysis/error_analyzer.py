import os
import sys
module_home = os.environ['NEURAL_PATH']
sys.path.insert(0, module_home)
import numpy as np
from models.data_generator import DataGenerator


class ErrorAnalyzer(object):

    def __init__(self, 
                model, #Model to analyze on 
                dataset, #Dataset to analyze 
                weights_path, #Path to the weights file to initialize the model with 
                hidden_dim, #Hidden dimension size for the model
                generator_set, #Which data group to use for the generator: 'validation', 'test' or 'training'
                layer_output=None): #Whether to show the output of any layer TODO: Implement this
        self.model = model.get_model(data_path=dataset, lstm_dim=hidden_dim, weights_path=weights_path) 
        self.data_generator = DataGenerator(1, dataset, generator_set, complete=True)
        self.layer_output = layer_output


    def get_error_sample(self):
        return self._get_sample(False)


    def get_correct_sample(self):
        return self._get_sample(True)


    def _get_sample(self, return_correct):
        while True:
            model_input, complete_data = self.data_generator.next()
            doc_questions, answers = model_input
            predictions = self.model.predict(doc_questions)
            for i in xrange(len(predictions)):
                prediction = np.argmax(predictions[i])
                answer = np.argmax(answers[i])
                if (prediction == answer) == return_correct:
                    data = [e.tolist() for e in complete_data[i]]
                    return {
                            'document': ' '.join([word for word in data[0] if word is not None]),
                            'question': ' '.join([word for word in data[1] if word is not None]),
                            'prediction': prediction,
                            'answer': answer
                            }

                    
