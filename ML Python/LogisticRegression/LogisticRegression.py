import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('C:\\Dev\\Learning\\Base\\SuperDataScience\\Logistic_Regression\\Social_Network_Ads.csv')

x = dataset.iloc[:, [2,3]].values
y = dataset.iloc[:, 4].values

from sklearn.cross_validation import train_test_split as tts
x_train, x_test, y_train, y_test = tts(x, y, test_size = 0.1, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()

x_train = sc_x.fit_transform(x_train)
x_test = sc_x.fit_transform(x_test)


from sklearn.linear_model import LogisticRegression
#FOR LINEAR LOGISTIC REGRESSION => ONLY TWO OUTPUTS (SIGMOID)
llr = LogisticRegression(random_state = 0)
#FOR MULTINOMIAL LOGISTIC REGRESSION => MORE THAN 2 OUTPUTS (SOFTMAX)
mlr = LogisticRegression(multi_class = 'multinomial', solver = 'newton-cg')

llr.fit(x_train, y_train)
mlr.fit(x_train, y_train)


#TEXT PRINT FOR PRE RESULTS
#y_pred = llr.predict(x_test)

#for i in range(0, len(y_test)):
#    print('Exp:' + str(y_test[i]) + ' => Pred:' + str(y_pred[i]) + ' : ' + str(y_test[i] == y_pred[i]))

#from sklearn.metrics import confusion_matrix
#cm = confusion_matrix(y_test, y_pred)

#print(cm)


#COMPARING RESULTS
from sklearn import metrics

print("Logistic regression Train Accuracy :: ", metrics.accuracy_score(y_train, llr.predict(x_train)))
print("Logistic regression Test Accuracy :: ", metrics.accuracy_score(y_test, llr.predict(x_test)))
 
print("Multinomial Logistic regression Train Accuracy :: ", metrics.accuracy_score(y_train, mlr.predict(x_train)))
print("Multinomial Logistic regression Test Accuracy :: ", metrics.accuracy_score(y_test, mlr.predict(x_test)))

#GRAFIC EXAMPLE
from matplotlib.colors import ListedColormap

x_set, y_set = x_train, y_train
x1, x2 = np.meshgrid(
    np.arange(start = x_set[:, 0].min() - 1, stop = x_set[:, 0].max() + 1, step = 0.01),
    np.arange(start = x_set[:, 1].min() - 1, stop = x_set[:, 1].max() + 1, step = 0.01)
)
plt.contour(x1, x2, llr.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape), 
    alpha = 0.75, cmap = ListedColormap(('red', 'green')))

plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())

for i, j in enumerate(np.unique(y_set)):
    plt.scatter(
        x_set[y_set == j, 0], x_set[y_set == j, 1],
        c = ListedColormap(('red', 'green'))(i), label = j
    )
plt.title('Logistic Regression (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
