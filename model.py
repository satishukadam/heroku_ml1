# Import the libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.externals import joblib

from warnings import filterwarnings
filterwarnings('ignore')

# Read the data
df = pd.read_csv('D:/Study/DataScience/Flask/Application1/heightweight.csv')


# Define feature and target variables
X = df.iloc[:, 0].values
y = df.iloc[:, -1].values

X = X.reshape(-1, 1)


# Split the data into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


# Create linear regression model
lr = LinearRegression()

lr.fit(X_train, y_train)


# Predict the values and find RMSE
y_pred = lr.predict(X_test)

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)


# Save the model as pickle
joblib.dump(lr, open('D:/Study/DataScience/Flask/Application1/models/regression_model.pkl', 'wb'))

