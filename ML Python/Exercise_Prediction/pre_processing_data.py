import numpy as np
import pandas as pd
import math

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler

class Cleaner:

    def generate_feed(self, csv_file):

        # columns dict
        #  0  ids
        #  3  score_3
        #  5  score_5
        #  7  risk_rate
        #  8  amount_borrowed
        #  9 borrowed_in_months
        #  10 credit_limit
        #  12 income
        #  14 gender
        #  21 ok_since
        #  22 n_bankruptcies
        #  23 update
        #  24 n_accounts
        #  25 n_issues

        print('Cleaning test data...')
        print('')

        # load data from csv
        df = pd.read_csv(csv_file)
        
        # filter usefull columns        
        feed = df.iloc[ :,  [ 0, 3, 5, 7, 8, 9, 10, 12, 14, 21, 22, 23, 24, 25 ] ]

        # ids
        ids_values = feed[ 'ids' ].values

        print('')

        print('Dealing with missing data...')
        # default value for empty gender
        self._fill_na_with_value( feed, 'gender', 'o' )
        self._fill_na_with_value( feed, 'score_3', feed['score_3'].mean() )
        self._fill_na_with_value( feed, 'risk_rate', feed['risk_rate'].mean() )
        self._fill_na_with_value( feed, 'amount_borrowed', feed['amount_borrowed'].mean() )
        self._fill_na_with_value( feed, 'borrowed_in_months', feed['borrowed_in_months'].mean() )
        self._fill_na_with_value( feed, 'income', feed['income'].mean() )
        self._fill_na_with_value( feed, 'n_accounts', feed['n_accounts'].mean() )
        

        # regression for empty rows in numeric columns
        ids_column = 'ids'
        ids_column_index = 0         
        base_columns_for_regression = [ 1, 2, 3, 4, 5, 7, 12 ]        
        
        credit_limit_update = self._fill_na_with_regression( feed, ids_column, ids_column_index, 'credit_limit', 6, base_columns_for_regression )
        ok_since_update = self._fill_na_with_regression( feed, ids_column, ids_column_index, 'ok_since', 9, base_columns_for_regression )
        n_bankruptcies_update = self._fill_na_with_regression( feed, ids_column, ids_column_index, 'n_bankruptcies', 10, base_columns_for_regression )
        n_defaulted_loans_update = self._fill_na_with_regression( feed, ids_column, ids_column_index, 'n_defaulted_loans', 11, base_columns_for_regression )                        
        n_issues_update = self._fill_na_with_regression( feed, ids_column, ids_column_index, 'n_issues', 13, base_columns_for_regression )

        feed.set_index(ids_column, inplace=True)
        feed.update( credit_limit_update.set_index(ids_column) )
        feed.update( ok_since_update.set_index(ids_column) )
        feed.update( n_bankruptcies_update.set_index(ids_column) )
        feed.update( n_defaulted_loans_update.set_index(ids_column) )                        
        feed.update( n_issues_update.set_index(ids_column) )

        # adjust the dtype
        feed['credit_limit'] = feed['credit_limit'].astype(float)
        feed['ok_since'] = feed['ok_since'].astype(float)
        feed['n_bankruptcies'] = feed['n_bankruptcies'].astype(float)
        feed['n_defaulted_loans'] = feed['n_defaulted_loans'].astype(float)                        
        feed['n_issues'] = feed['n_issues'].astype(float)

        print('')

        print('Normalizing features...')
        # categorical features to one hot vector
        categorical_feed = self._generate_one_hot_vector_from_categorical(feed, 7)

        # scalar features normalization
        numeric_features_index = [ 0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12 ]
        scalar_feed = self._generate_scalar_features(feed, numeric_features_index)

        # generate neural model feed and target
        feed = np.concatenate( (categorical_feed, scalar_feed), axis = 1 )      

        return ids_values, feed

    def generate_feed_n_target(self, csv_file):

        # columns dict
        #  0  ids
        #  4  score_3
        #  6  score_5
        #  8  risk_rate
        #  9  amount_borrowed
        #  10 borrowed_in_months
        #  11 credit_limit
        #  13 income
        #  15 gender
        #  22 ok_since
        #  23 n_bankruptcies
        #  24 update
        #  25 n_accounts
        #  26 n_issues

        print('Cleaning training data...')
        print('')

        # load data from csv
        df = pd.read_csv(csv_file)

        print('Removing unlabeled rows...')
        df = self._remove_unlabeled_rows(df, 'default')

        # filter usefull columns
        feed = df.iloc[ :, [ 0, 4, 6, 8, 9, 10, 11, 13, 15, 22, 23, 24, 25, 26 ] ]

        print('')

        print('Dealing with missing data...')
        # default value for empty gender
        self._fill_na_with_value( feed, 'gender', 'o' )

        # regression for empty rows in numeric columns
        ids_column = 'ids'
        ids_column_index = 0         
        base_columns_for_regression = [ 1, 2, 3, 4, 5, 7, 12 ]        
        
        credit_limit_update = self._fill_na_with_regression( feed, ids_column, ids_column_index, 'credit_limit', 6, base_columns_for_regression )
        ok_since_update = self._fill_na_with_regression( feed, ids_column, ids_column_index, 'ok_since', 9, base_columns_for_regression )
        n_bankruptcies_update = self._fill_na_with_regression( feed, ids_column, ids_column_index, 'n_bankruptcies', 10, base_columns_for_regression )
        n_defaulted_loans_update = self._fill_na_with_regression( feed, ids_column, ids_column_index, 'n_defaulted_loans', 11, base_columns_for_regression )                        
        n_issues_update = self._fill_na_with_regression( feed, ids_column, ids_column_index, 'n_issues', 13, base_columns_for_regression )

        feed.set_index(ids_column, inplace=True)
        feed.update( credit_limit_update.set_index(ids_column) )
        feed.update( ok_since_update.set_index(ids_column) )
        feed.update( n_bankruptcies_update.set_index(ids_column) )
        feed.update( n_defaulted_loans_update.set_index(ids_column) )                        
        feed.update( n_issues_update.set_index(ids_column) )

        # adjust the dtype
        feed['credit_limit'] = feed['credit_limit'].astype(float)
        feed['ok_since'] = feed['ok_since'].astype(float)
        feed['n_bankruptcies'] = feed['n_bankruptcies'].astype(float)
        feed['n_defaulted_loans'] = feed['n_defaulted_loans'].astype(float)                        
        feed['n_issues'] = feed['n_issues'].astype(float)

        print('')

        print('Normalizing features...')
        # categorical features to one hot vector
        categorical_feed = self._generate_one_hot_vector_from_categorical(feed, 7)

        # scalar features normalization
        numeric_features_index = [ 0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12 ]
        scalar_feed = self._generate_scalar_features(feed, numeric_features_index)

        # generate neural model feed and target
        feed = np.concatenate( (categorical_feed, scalar_feed), axis = 1 )
        target = df.iloc[ :, [1] ].values

        print('')

        return feed, target

    def _remove_unlabeled_rows(self, df, label_column):
        # removing unlabeled rows
        non_labeled_index = df[ df[label_column].isnull() ].index
        df = df.drop( non_labeled_index )

        return df
    
    def _fill_na_with_value(self, feed, column, value):
        feed[column].fillna(value, inplace=True)
    
    def _fill_na_with_regression(self,
        feed,
        ids_column,
        ids_column_index,
        column,
        column_index,
        base_columns_index_for_regression,
        degree=3):

        # feed and target for regression
        update_feed = feed[ feed[column].notnull() ].values[ :, base_columns_index_for_regression ]
        update_target = feed[ feed[column].notnull() ].values[ :, column_index ]

        # polynomial features conversor
        feature_poly = PolynomialFeatures(degree = 3)
        update_poly = feature_poly.fit_transform(update_feed)

        # regressor
        regressor_update = LinearRegression()
        regressor_update.fit(update_poly, update_target)

        # prediction        
        update_test_feed_ids = feed[ feed[column].isnull() ].values[ :, [ ids_column_index ] ]        
        update_test_feed = feed[ feed[column].isnull() ].values[ :, base_columns_index_for_regression ]
        update_test_pred = regressor_update.predict( feature_poly.fit_transform(update_test_feed) ).reshape(-1, 1)
        update_test_pred_ids = np.concatenate( (update_test_feed_ids, update_test_pred), axis = 1 )        

        # generate a auxiliar dataframe for update the source dataframe
        update_test_pred_ids = pd.DataFrame( update_test_pred_ids, columns=[ ids_column, column ] )

        return update_test_pred_ids

    def _generate_one_hot_vector_from_categorical(self, feed, categorical_index):
        categorical_feed = feed.iloc[ :, categorical_index ]

        labelEncoder = LabelEncoder()
        categorical_feed = labelEncoder.fit_transform( categorical_feed )

        oneHotEncoder = OneHotEncoder()
        categorical_feed = oneHotEncoder.fit_transform( categorical_feed.reshape(-1,1) ).toarray()

        # dummy variable trap avoiding
        categorical_feed = np.delete( categorical_feed, np.s_[1], axis=1)

        return categorical_feed

    def _generate_scalar_features(self, feed, features_index):
        scalar_feed = feed.iloc[ :, features_index ]

        # normalization
        sc = StandardScaler()
        scalar_feed = sc.fit_transform( scalar_feed )

        return scalar_feed
