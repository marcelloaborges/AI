import numpy as np
import pandas as pd
import math

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler

class Cleaner:

    def generate_feed_n_target(self, csv_file):
        # columns dict        
        #  0  Date
        #  1  Open        # Scalar
        #  2  High        # Scalar
        #  3  Low         # Scalar
        #  4  Close       # Scalar
        #  5  Adj Close   # Scalar
        #  6  Volume      # Scalar        

        print('Cleaning test data...')        

        # load data from csv
        df = pd.read_csv(csv_file, delimiter=',')

        # drop null registers
        df = df.dropna()

        # filter usefull columns        
        data = df.iloc[ :, [ 1, 2, 3, 4, 5, 6 ] ]

        OPEN = 0
        HIGH = 1
        LOW = 2
        CLOSE = 3
        ADJ_CLOSE = 4
        VOLUME = 5                
        
        print('Normalizing features...')

        # scalar columns normalization
        scalar_data = self._generate_scalar_features(data, [ OPEN, HIGH, LOW, ADJ_CLOSE, VOLUME ] )

        # neural model feed and target
        feed = scalar_data
        target = data.iloc[ :, [CLOSE] ].values

        print('')

        return feed, target

    def _fill_na_with_value(self, data, column, value):
        data[column].fillna(value, inplace=True)

    def _generate_one_hot_vector_from_categorical_label(self, data, categorical_index):
        categorical_data = data.iloc[ :, categorical_index ]

        labelEncoder = LabelEncoder()
        categorical_data = labelEncoder.fit_transform( categorical_data )

        oneHotEncoder = OneHotEncoder()
        categorical_data = oneHotEncoder.fit_transform( categorical_data.reshape(-1,1) ).toarray()

        # dummy variable trap avoiding
        categorical_data = np.delete( categorical_data, np.s_[0], axis=1)

        return categorical_data

    def _generate_one_hot_vector_from_categorical_int(self, data, categorical_index):
        categorical_data = np.array( data.iloc[ :, categorical_index ].values ).reshape(-1,1)

        oneHotEncoder = OneHotEncoder()
        categorical_data = oneHotEncoder.fit_transform( categorical_data ).toarray()

        # dummy variable trap avoiding
        categorical_data = np.delete( categorical_data, np.s_[0], axis=1)

        return categorical_data

    def _generate_scalar_features(self, data, features_index):
        scalar_data = data.iloc[ :, features_index ]

        # normalization
        sc = StandardScaler()
        scalar_data = sc.fit_transform( scalar_data )

        return scalar_data


