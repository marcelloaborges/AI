import math
from textwrap import wrap
import os

import numpy as np
import matplotlib.pyplot as plt

import torch

from pre_processing_data import DataGenerator
from train import Trainer


movies_csv = './ml-1m/movies.csv'
ratings_csv = './ml-1m/ratings.csv'

data_generator = DataGenerator()
(n_users, n_movies), (feed_train, feed_test, feed_target, feed_val), _, minmax = data_generator.generate_feed_n_test(ratings_csv)

print(f'Embeddings: {n_users} users, {n_movies} movies')
print(f'Feed shape: {feed_train.shape}')
print(f'Target shape: {feed_target.shape}')


DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
N_USERS = n_users
N_MOVIES = n_movies
N_FACTORS = 150
HIDDEN = [500, 500, 500]
EMBEDDING_DROPOUT = 0.05
DROPOUTS = [0.5, 0.5, 0.25]
CHECKPOINT = './checkpoint.pth'
LR = 1e-3
WEIGHT_DECAY = 1e-5
BATCH_SIZE = 2048
EPOCHS = 10

trainner = Trainer(DEVICE, N_USERS, N_MOVIES, N_FACTORS, HIDDEN, EMBEDDING_DROPOUT, DROPOUTS, LR, WEIGHT_DECAY, CHECKPOINT)
trainner.train( feed_train, feed_test, feed_target, feed_val, minmax, BATCH_SIZE, EPOCHS )