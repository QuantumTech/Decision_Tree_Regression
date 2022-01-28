# Decision Tree Regression

# CART = Classification and regression tree

# Regression tree is more complex than classification trees

# The regression tree splits the data up

# Information entropy (Research this)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 1) Data preprocessing

# IMPORTING THE LIBRARIES

# 'np' is the numpy shortcut!
# 'plt' is the matplotlib shortcut!
# 'pd' is the pandas shortcut!

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 1) Data preprocessing

# IMPORTING THE DATASET

# Data set is creating the data frame of the 'Position_Salaries.csv' file
# Features (independent variables) = The columns the predict the dependent variable
# Dependent variable = The last column
# 'X' = The matrix of features (country, age, salary)
# 'Y' = Dependent variable vector (purchased (last column))
# '.iloc' = locate indexes[rows, columns]
# ':' = all rows (all range)
# ':-1' = Take all the columns except the last one
# '.values' = taking all the values

# 'X = dataset.iloc[:, 1:-1].values' (This will take the values from the second column (index 1) to the second from last on)

# 'y = dataset.iloc[:, -1].values' (NOTICE! .iloc[all the rows, only the last column])

# 'X' (This contains the 'x' axis in a vertical 2D array)
# 'y' (This contains the 'y' axis in a 1D horizontal vector)

# Note to self: Only the CSV file and column/row values need to be changed here!

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# TRAINING THE DECISION TREE REGRESSION MODEL ON THE WHOLE DATASET

# Reminder: 'regressor' is an object of the DecisionTreeRegressor

# If I had split the data then I would have to use 'X_train' and 'y_train' in the '.fit' method

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X, y)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# PREDICTING A NEW RESULT

regressor.predict([[6.5]])

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# VISUALISING THE DECISION TREE REGRESSION RESULTS (HIGHER RESOLUTION)

# Side note: This step is only relevent to data with one feature, any more than that would require higher dimensions!

X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'orange')
plt.plot(X_grid, regressor.predict(X_grid), color = 'green')
plt.title('Truth or Bluff (Decision Tree Regression)', color = 'blue')
plt.xlabel('Position level', color = 'red')
plt.ylabel('Salary', color = 'violet')
plt.show()