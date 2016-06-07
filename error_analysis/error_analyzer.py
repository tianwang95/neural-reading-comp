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
                output_attention=False): #If set, a separate model will be built to output the predicitons of the layer with the given name
        self.model = model.get_model(data_path=dataset, lstm_dim=hidden_dim, weights_path=weights_path) 
        if output_attention:
            self.layer_model = model.get_model(data_path=dataset, lstm_dim=hidden_dim, weights_path=weights_path, output_attention=True)
        self.data_generator = DataGenerator(1, dataset, generator_set, complete=True)


    def get_error_sample(self, with_attention=False):
        return self._get_sample(return_correct=False, get_intermediate=with_attention)


    def get_correct_sample(self, with_attention=False):
        return self._get_sample(return_correct=True, get_intermediate=with_attention)


    def _get_sample(self, return_correct, get_intermediate=False):
        while True:
            model_input, complete_data = self.data_generator.next()
            doc_questions, answers = model_input
            predictions = self.model.predict(doc_questions)
            if get_intermediate:
                assert self.layer_model
                layer_outputs = self.layer_model.predict(doc_questions)
            for i in xrange(len(predictions)):
                prediction = np.argmax(predictions[i])
                answer = np.argmax(answers[i])
                if (prediction == answer) == return_correct:
                    data = [e.tolist() for e in complete_data[i]]
                    if get_intermediate:
                        document = "" 
                        for k in xrange(len(data[0])):
                            word = data[0][k]
                            if word is None:
                                break
                            attention = layer_outputs[i][0][k]
                            if attention > 0.00001:
                                document += "{}({:.4f}) ".format(word, float(attention))
                            else:
                                document += word + " "
                        return {
                                'document': document,
                                'question': ' '.join([word for word in data[1] if word is not None]),
                                'prediction': prediction,
                                'answer': answer
                                }
                    else:
                        return {
                                'document': ' '.join([word for word in data[0] if word is not None]),
                                'question': ' '.join([word for word in data[1] if word is not None]),
                                'prediction': prediction,
                                'answer': answer
                                }


                    
