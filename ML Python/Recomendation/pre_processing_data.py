import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class DataGenerator:

    def __init__(self):
        self.ENCODING = 'ISO-8859-1'
        self.SEP = ';'

        self.USER_ID = 'user_id'
        self.MOVIE_ID = 'movie_id'
        self.RATING = 'rating'
        self.TIMESTAMP = 'timestamp'

        self.COLUMNS_LIST = ["user_id", "movie_id", "rating", "timestamp"]

    def generate_feed_n_test(self, ratings_csv, test_size=0.2, random_state=1):
        ratings = pd.read_csv(ratings_csv, sep = self.SEP, encoding = self.ENCODING, names = self.COLUMNS_LIST)
                
        ratings.groupby(self.USER_ID)[self.RATING].count()

        unique_users = ratings.user_id.unique()
        user_to_index = {old: new for new, old in enumerate(unique_users)}
        new_users = ratings.user_id.map(user_to_index).values.reshape(-1, 1)

        unique_movies = ratings.movie_id.unique()
        movie_to_index = {old: new for new, old in enumerate(unique_movies)}
        new_movies = ratings.movie_id.map(movie_to_index).values.reshape(-1, 1)

        n_users = unique_users.shape[0]
        n_movies = unique_movies.shape[0]

        feed = np.concatenate( (new_users, new_movies), axis=1 )
        target = ratings[self.RATING].values        


        feed_train, feed_test, feed_target, feed_val = train_test_split(feed, target, test_size=test_size, random_state=random_state)
        minmax = ratings.rating.min().astype(float), ratings.rating.max().astype(float)


        return (n_users, n_movies), (feed_train, feed_test, feed_target, feed_val), (user_to_index, movie_to_index), minmax