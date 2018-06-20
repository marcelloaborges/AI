import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('C:\\Dev\\Learning\\Base\\SuperDataScience\\Polynomial_Regression\\Position_Salaries.csv')

x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

y_pred = regressor.predict(6.5)

x_grid = np.arange(min(x), max(x), 0.1)
x_grid = np.reshape((len(x_grid), 1))

plt.scatter(x, y, color = 'red')
plt.plot(x_grid, regressor.predict(x_grid), color='blue')
plt.title('Truth of Bluff (Regression Model)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
