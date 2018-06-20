import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('C:\\Dev\\Learning\\Base\\SuperDataScience\\Polynomial_Regression\\Position_Salaries.csv')

x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values


from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(x, y)

"""
plt.scatter(x, y, color = 'red')
plt.plot(x, lr.predict(x), color='blue')
plt.title('Truth of Bluff (Linear Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()"""


from sklearn.preprocessing import PolynomialFeatures
feature_poly = PolynomialFeatures(degree = 4)
x_poly = feature_poly.fit_transform(x)

pr = LinearRegression()
pr.fit(x_poly, y)

x_grid = np.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape(len(x_grid), 1)

plt.scatter(x, y, color = 'red')
plt.plot(x_grid, pr.predict(feature_poly.fit_transform(x_grid)), color='blue')
plt.title('Truth of Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

print(lr.predict(6.5))
print(pr.predict(feature_poly.fit_transform(6.5)))