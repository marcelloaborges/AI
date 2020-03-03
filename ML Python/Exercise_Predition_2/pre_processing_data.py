import numpy as np
import pandas as pd
import math

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler

class Cleaner:

    def generate_feed_n_target(self, csv_file):
        # columns dict
        #  2  Age
        #  3  Job               # Categorical
        #  4  Marital           # Categorical
        #  5  Education         # Categorical
        #  6  Default           # Categorical
        #  7  Balance           
        #  8  CarLoan           # Categorical
        #  9  Communication     # Categorical
        #  10 LastContactDay    
        #  11 LastContactMonth  # Categorical
        #  12 NoOfContacts
        #  13 PrevAttempts
        #  14 Outcome           # Categorical
        #  15 CallStart         # Time
        #  16 CallEnd           # Time
        #  17 CarInsurance      # Target Y/N Prob        

        print('Cleaning test data...')
        print('')

        # load data from csv
        df = pd.read_csv(csv_file, delimiter=';')

        # filter usefull columns        
        data = df.iloc[ :, [ 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17 ] ]

        AGE = 0
        JOB = 1
        MARITAL = 2
        EDUCATION = 3
        DEFAULT = 4
        BALANCE = 5
        CAR_LOAN = 6
        COMMUNICATION = 7
        LAST_CONTACT_DAY = 8
        LAST_CONTACT_MONTH = 9
        N_Of_CONTATCS = 10
        PREV_ATTEMPTS = 11
        OUTCOME = 12
        CALL_START = 13
        CALL_END = 14
        CAR_INSURANCE = 15
        CALL_TIME = 16

        print('Dealing with missing data...')

        # default value for empty categorical columns
        self._fill_na_with_value( data, 'Job',              'o' )
        self._fill_na_with_value( data, 'Marital',          'o' )
        self._fill_na_with_value( data, 'Education',        'o' )
        self._fill_na_with_value( data, 'Default',          2.0 )
        self._fill_na_with_value( data, 'CarLoan',          2.0 )
        self._fill_na_with_value( data, 'Communication',    'o' )
        self._fill_na_with_value( data, 'LastContactMonth', 'o' )
        self._fill_na_with_value( data, 'Outcome',          'o' )

        # scalar columns empty values filled with mean
        self._fill_na_with_value( data, 'Age',              data['Age'].mean() )
        self._fill_na_with_value( data, 'Balance',          data['Balance'].mean() )
        self._fill_na_with_value( data, 'LastContactDay',   data['LastContactDay'].mean() )
        self._fill_na_with_value( data, 'NoOfContacts',     data['NoOfContacts'].mean() )
        self._fill_na_with_value( data, 'PrevAttempts',     data['PrevAttempts'].mean() )

        # adding the CALL TIME to the data
        data['CallStart'] = pd.to_timedelta( data['CallStart'] )
        data['CallEnd'] = pd.to_timedelta( data['CallEnd'] )
        data['CallTime'] = (data.iloc[ :, CALL_END ].values - data.iloc[ :, CALL_START ].values).astype(int)

        print('')
        print('Normalizing features...')
        
        # categorical columns to one hot encoder
        job_data = self._generate_one_hot_vector_from_categorical_label(              data, JOB )
        marital_data = self._generate_one_hot_vector_from_categorical_label(          data, MARITAL )
        education_data = self._generate_one_hot_vector_from_categorical_label(        data, EDUCATION )                
        default_data = self._generate_one_hot_vector_from_categorical_int(            data, DEFAULT )
        carloan_data = self._generate_one_hot_vector_from_categorical_int(            data, CAR_LOAN )
        communication_data = self._generate_one_hot_vector_from_categorical_label(    data, COMMUNICATION )
        lastcontactmonth_data = self._generate_one_hot_vector_from_categorical_label( data, LAST_CONTACT_MONTH )
        outcome_data = self._generate_one_hot_vector_from_categorical_label(          data, OUTCOME )

        # scalar columns normalization
        scalar_data = self._generate_scalar_features(data, [ AGE, BALANCE, LAST_CONTACT_DAY, N_Of_CONTATCS, PREV_ATTEMPTS, CALL_TIME ] )

        # neural model feed and target
        feed = np.concatenate( (job_data, marital_data, education_data, default_data, carloan_data, communication_data, lastcontactmonth_data, outcome_data, scalar_data ), axis=1 )        
        target = data.iloc[ :, [CAR_INSURANCE] ].values

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


