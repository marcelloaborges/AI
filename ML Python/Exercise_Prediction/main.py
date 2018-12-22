import numpy as np

from pre_processing_data import Cleaner
from train import Trainer
from test import Tester

# Hyperparameter configration
FEED_SIZE = 14
FC1_UNITS = 64
FC2_UNITS = 32
OUTPUT_SIZE = 1
LR = 1e-3
BATCH_SIZE = 512
EPOCHS = 10
CHECKPOINT = './checkpoint.pth'

## COMENT THIS SECTION IF YOU WANT TO TRAIN THE MODEL AGAIN ##

# Learning
# loading the dataset
csv_data_file = 'train/puzzle_train_dataset.csv'

# cleaning data and generating the training set
cleaner = Cleaner()
feed, target = cleaner.generate_feed_n_target(csv_data_file)

trainer = Trainer(FEED_SIZE, OUTPUT_SIZE, FC1_UNITS, FC2_UNITS, LR, CHECKPOINT)
trainer.train(feed, target, BATCH_SIZE, EPOCHS)

###############################################################


# Testing
# loading the test dataset
csv_test_file = 'test/puzzle_test_dataset.csv'

# cleaning data and generating the test set
cleaner = Cleaner()
ids, feed = cleaner.generate_feed(csv_test_file)

tester = Tester(FEED_SIZE, OUTPUT_SIZE, FC1_UNITS, FC2_UNITS, CHECKPOINT)
predictions = tester.test(feed)

# saving results
results = np.concatenate( (ids.reshape(-1, 1), predictions), axis = 1)

print('Results')
print(results)
np.savetxt('predictions.csv', results, delimiter=',', fmt='%s', header='ids,predictions')