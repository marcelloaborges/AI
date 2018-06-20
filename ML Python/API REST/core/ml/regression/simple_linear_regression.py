import numpy as np #neural's models
import matplotlib.pyplot as plt #plots
import pandas as pd #manage datasets
from io import StringIO

class SimpleLinearRegression():

    def Run(self, csv):
        dataset = pd.read_csv(StringIO(csv))
        x = dataset.iloc[:, 0:1].values
        y = dataset.iloc[:, 1].values

        #split base into train and test
        from sklearn.cross_validation import train_test_split as tts
        x_train, x_test, y_train, y_test = tts(x, y, test_size = 0.2, random_state = 0)

        #fit the regression
        from sklearn.linear_model import LinearRegression
        regressor = LinearRegression()
        regressor.fit(x_train, y_train)

        #regression
        y_pred = regressor.predict(x_test)

        result = []
        for i in range(0, len(y_pred)):
            result.append(
                {
                    'Expected': y_test[i],
                    'Preditect': y_pred[i],
                }
            )

        return result