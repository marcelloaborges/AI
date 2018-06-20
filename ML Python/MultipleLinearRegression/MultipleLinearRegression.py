import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('C:\\Dev\\Learning\\Base\\SuperDataScience\\Multiple_Linear_Regression\\50_Startups.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder_x = LabelEncoder()
x[:, 3] = labelEncoder_x.fit_transform(x[:, 3])
oneHotEncoder = OneHotEncoder(categorical_features = [3])
x = oneHotEncoder.fit_transform(x).toarray()

#removing the first column to avoid the dummy variable trap
x = x[:, 1:]

from sklearn.cross_validation import train_test_split as tts
x_train, x_test, y_train, y_test = tts(x, y, test_size=0.2, random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)

print('All variables')
for i in range(0, len(y_pred)):
    print('exp:' + str(y_test[i]) + ' | pred: ' + str(y_pred[i]))

#backward elimination
import statsmodels.formula.api as sm

#adding the constant column for the regression (x0)
x = np.append(arr = np.ones((50, 1)).astype(int), values = x, axis = 1)

def backwardElimination(x, sl):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
    #print(regressor_OLS.summary())
    return x

SL = 0.05
#x_opt = x[:, [0, 1, 2, 3 , 4, 5]]
x_Modeled = backwardElimination(x, SL)

x_train, x_test, y_train, y_test = tts(x_Modeled, y, test_size=0.2, random_state=0)
regressor.fit(x_train, y_train)
y_pred = regressor.predict(x_test)

print('Opt variables')
for i in range(0, len(y_pred)):
    print('exp:' + str(y_test[i]) + ' | pred: ' + str(y_pred[i]))

