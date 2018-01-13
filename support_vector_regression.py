# Regression Template

# Multiple Linear Regression

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the datasets

datasets = pd.read_csv('Position_Salaries.csv')
X = datasets.iloc[:, 1:2].values
Y = datasets.iloc[:, 2].values

"""# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
"""
Y = Y.reshape(-1,1)

# Feature Scaling

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_Y = StandardScaler()
X = sc_X.fit_transform(X)
Y = sc_Y.fit_transform(Y)


# Fitting the SVR model to the dataset

from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X,Y)

# Predicting a new result with the Polynomial Regression 

Y_Pred = sc_Y.inverse_transform(regressor.predict(sc_X.transform(np.array([6.5]).reshape(-1,1))))

# Visualising the Regression results

plt.scatter(X,Y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Regression Results')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

