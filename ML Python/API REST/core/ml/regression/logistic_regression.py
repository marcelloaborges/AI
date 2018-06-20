import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from io import StringIO

class LogisticRegression():

    def Run(self, csv):
        dataset = pd.read_csv(StringIO(csv))

        x = dataset.iloc[:, [0,1]].values
        y = dataset.iloc[:, 2].values

        from sklearn.cross_validation import train_test_split as tts
        x_train, x_test, y_train, y_test = tts(x, y, test_size = 0.1, random_state = 0)

        from sklearn.preprocessing import StandardScaler
        sc_x = StandardScaler()

        x_train = sc_x.fit_transform(x_train)
        x_test = sc_x.fit_transform(x_test)


        from sklearn.linear_model import LogisticRegression
        #FOR LINEAR LOGISTIC REGRESSION => ONLY TWO OUTPUTS (SIGMOID)
        llr = LogisticRegression(random_state = 0)

        llr.fit(x_train, y_train)

        #COMPARING RESULTS
        from sklearn import metrics

        print("Multinomial Logistic regression Train Accuracy :: ", metrics.accuracy_score(y_train, mlr.predict(x_train)))
        print("Multinomial Logistic regression Test Accuracy :: ", metrics.accuracy_score(y_test, mlr.predict(x_test)))

        return [0]
