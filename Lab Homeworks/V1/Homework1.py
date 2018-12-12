
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
poly_reg.fit(X_poly, y)
lin_reg_poly = LinearRegression()
lin_reg_poly .fit(X_poly, y)

# Fitting Decision Tree Regression to the dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X, y)

# Fitting Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestRegressor
regressorRF = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressorRF.fit(X, y)

# Visualising the results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.plot(X_grid, lin_reg_poly .predict(poly_reg.fit_transform(X_grid)), color = 'green')
plt.plot(X_grid, regressor.predict(X_grid), color = 'purple')
plt.plot(X_grid, regressorRF.predict(X_grid), color = 'orange')
plt.title('Expected Salaries')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Getting the results 
lin_reg.predict(6.5)
lin_reg_poly.predict(poly_reg.fit_transform(6.5))
regressor.predict(6.5)
regressorRF.predict(6.5)
