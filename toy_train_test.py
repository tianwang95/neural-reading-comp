import os
from models import attentive_model
from models.data_generator import DataGenerator
import theano
theano.config.floatX = 'float32'
theano.config.optimization = 'fast_run'

nb_epoch = 10
batch_size = 3

dataset = 'datasets/toy_dataset/cnn_processed'
model = attentive_model.get_model(data_path=dataset, lstm_dim=32)

#TRAINING
print "Starting training"

train_generator = DataGenerator(batch_size, dataset, 'training')
validation_generator = DataGenerator(batch_size, dataset, 'validation')

print "Calling fit generator"

model.fit_generator(train_generator,
                                   samples_per_epoch=train_generator.nb_samples_epoch,
                                   nb_epoch=nb_epoch,
                                   verbose=2,
                                   validation_data=validation_generator,
                                   nb_val_samples=validation_generator.nb_samples_epoch)

print "Training done"
