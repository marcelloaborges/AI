import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from io import StringIO

class LogisticRegression:

    def Run(self, csv):
        dataset = pd.read_csv(StringIO(csv))

        x = dataset.iloc[:, [0,1]].values
        y = dataset.iloc[:, 2].values

        from sklearn.cross_validation import train_test_split as tts
        x_train, x_test, y_train, y_test = tts(x, y, test_size = 0.2, random_state = 0)

        from sklearn.preprocessing import StandardScaler
        sc_x = StandardScaler()

        x_train_sc = sc_x.fit_transform(x_train)
        x_test_sc = sc_x.fit_transform(x_test)

        from sklearn.linear_model import LogisticRegression
        #FOR LINEAR LOGISTIC REGRESSION => ONLY TWO OUTPUTS (SIGMOID)
        llr = LogisticRegression(random_state = 0)

        llr.fit(x_train_sc, y_train)

        y_pred = llr.predict(x_test_sc)

        result = []
        for i in range(0, len(y_pred)):
            result.append(
                {
                    'Age': x_test.tolist()[i][0],
                    'Salary': x_test.tolist()[i][1],
                    'Expected': y_test.tolist()[i],
                    'Preditect': y_pred.tolist()[i],
                }
            )

        print(result)

        return result
