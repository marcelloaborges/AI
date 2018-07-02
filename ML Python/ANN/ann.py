#EXAMPLE OF CLASSIFICATION MODEL WITH KERAS

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import keras

dataset = pd.read_csv('C:\Dev\Learning\Base\SuperDataScience\Artificial_Neural_Networks\Churn_Modelling.csv')
x = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

#categorical variable Geografy
labelencoder_x_1 = LabelEncoder()
x[:, 1] = labelencoder_x_1.fit_transform(x[:, 1])

#categorical variable Gender
labelencoder_x_2 = LabelEncoder()
x[:, 2] = labelencoder_x_2.fit_transform(x[:, 2])

#dummy variable trap avoiding
onehotencoder = OneHotEncoder(categorical_features = [1])
x = onehotencoder.fit_transform(x).toarray()
x = x[:, 1:]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


#ANN BUILDING AND PROCESSING
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()

#H1/INPUT
classifier.add(Dense(output_dim = 6, init = 'uniform',  activation = 'relu', input_dim = 11))
#H2
classifier.add(Dense(output_dim = 6, init = 'uniform',  activation = 'relu'))

#OUTPUT
classifier.add(Dense(output_dim = 1, init = 'uniform',  activation = 'sigmoid'))

#CALC
#OPTIMIZER 
#LOSS => LOSS FUNCTION => FOR BINARY OUTPUT(SIGMOID) USUALLY WE USE 'BINARY CROSS ENTROPY'
#LOSS => LOSS FUNCTION => FOR N CLASSES OUTPUT(SOFTMAX) USUALLY WE USE 'CROSS ENTROPY'
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

classifier.fit(x_train, y_train, batch_size = 10, epochs = 100)


y_pred = classifier.predict(x_test)
y_pred = (y_pred > 0.5)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

print(cm)