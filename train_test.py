import os
import models
import preprocessing.rc_data.DataProcessor.rc_data_generator as data_generator

nb_epochs = 10
batch_size = 32
dataset = 'datasets/toy_dataset/cnn_processed/questions'
model = models.attentive_model.get_model()

#TRAINING
for _ in nb_epochs:
    for X, y in data_generator(os.path.join(dataset,'training')):
        trainHistory = model.fit(X, y, batch_size=batch_size, nb_epochs=1, verbose=2)

print "Training done"
print trainHistory.history




