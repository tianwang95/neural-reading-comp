import os
import numpy as np
from models import attentive_model
from models.data_generator import DataGenerator

dataset = 'datasets/toy_dataset/cnn_processed'
eval_num_errors = 10

model = attentive_model.get_model(data_path=dataset, lstm_dim=128, weights_path=os.path.join('results', 'att_model', 'att_model.04-1.77.hdf5'))
validation_generator = DataGenerator(1, dataset, 'validation', complete=True)

errors = []

for model_input, complete_data in validation_generator:
    question = [word for word in complete_data[0][1] if word is not None]
    print ' '.join(question)

'''
for model_input, complete_data in validation_generator:
    doc_queries, answers = model_input
    if len(errors) >= eval_num_errors:
        break
    predictions = model.predict(doc_queries)
    for i in xrange(len(predictions)):
        prediction = np.argmax(predictions[i])
        answer = np.argmax(answers[i])
        if prediction != answer:
            errors.append({'data': [e.tolist() for e in complete_data[i]], 'prediction': prediction, 'answer': answer})
            if len(errors) >= eval_num_errors:
                break

for error in errors:
    with open('errors.txt', 'w') as f:
        prediction = error['prediction']
        answer = error['answer']
        document = [word for word in error['data'][0] if word is not None]
        question = [word for word in error['data'][1] if word is not None]
        f.write("Document:")
        f.write(' '.join(document))
        f.write("---------------")
        f.write("Question:")
        f.write(' '.join(question))
        f.write("---------------")
        f.write("Prediction:")
        f.write(prediction)
        f.write("Answer:")
        f.write(answer)
        f.write("--------------------------")
'''
