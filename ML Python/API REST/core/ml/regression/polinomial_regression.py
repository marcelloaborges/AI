import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from io import StringIO

class PolynomialRegression():

    def Run(self, csv):
        dataset = pd.read_csv(StringIO(csv))
        x = dataset.iloc[:, 0:1].values
        y = dataset.iloc[:, 1].values

        from sklearn.cross_validation import train_test_split as tts
        x_train, x_test, y_train, y_test = tts(x, y, test_size = 0.2, random_state = 0)

        from sklearn.linear_model import LinearRegression
        from sklearn.preprocessing import PolynomialFeatures

        feature_poly = PolynomialFeatures(degree = 4)
        x_poly = feature_poly.fit_transform(x_train)

        pr = LinearRegression()
        pr.fit(x_poly, y_train)      

        y_pred = pr.predict(feature_poly.fit_transform(x_test))

        result = []
        for i in range(0, len(y_pred)):
            result.append(
                {
                    'Expected': x_test.tolist()[i][0],
                    'Preditect': y_pred[i],
                }
            )

        return result