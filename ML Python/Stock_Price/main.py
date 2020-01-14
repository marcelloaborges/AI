from pre_processing_data import Cleaner
from train import Trainer

# Learning
# loading the dataset
# csv_data_file = 'PETR4.SA.csv'
csv_data_file = 'MGLU3.SA.csv'

cleaner = Cleaner()
feed, target = cleaner.generate_feed_n_target(csv_data_file)

trainer = Trainer(2)
# With Logistic Regression
trainer.train_with_polynomial_regression(feed, target)

# With NN
# trainer.train_with_nn(feed, target, BATCH_SIZE, EPOCHS)
