from pre_processing_data import Cleaner
from train import Trainer


# Hyperparameter configration
FEED_SIZE = 44
FC1_UNITS = 128
FC2_UNITS = 64
OUTPUT_SIZE = 1
LR = 1e-4
BATCH_SIZE = 128
EPOCHS = 100
CHECKPOINT = './checkpoint.pth'

# Learning
# loading the dataset
csv_data_file = 'base.csv'

cleaner = Cleaner()
feed, target = cleaner.generate_feed_n_target(csv_data_file)

trainer = Trainer(FEED_SIZE, OUTPUT_SIZE, FC1_UNITS, FC2_UNITS, LR, CHECKPOINT)
# With Logistic Regression
trainer.train_with_logistic_regression(feed, target)

# With NN
trainer.train_with_nn(feed, target, BATCH_SIZE, EPOCHS)
