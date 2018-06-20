import numpy as np #neural's models
#import tensorflow as tf
import matplotlib.pyplot as plt #plots
import pandas as pd #manage datasets

#from mnist import MNIST
#from keras.datasets import mnist
#Training Data.
#[x_train, y_train], [x_test, y_test] = mnist.load_data[]

#data pre-processing
#from sklearn.preprocessing import Imputer as imp
#applying the mean strategy to the dataset for missing values in columns
#imputer = imp[missing_values = 'NaN', strategy = 'mean', axis = 0]

dataset = pd.read_csv('C:\\Dev\\Learning\\Base\\SuperDataScience\\Simple_Linear_Regression\\Salary_Data.csv')
x = dataset.iloc[:, :-1].values
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

for i in range(0, y_test.size):
  print (str(x_test[i]) + ":" + str(y_test[i]) + " => " + str(y_pred[i]))  

#plot results
plt.scatter(x_train, y_train, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

plt.scatter(x_test, y_test, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

