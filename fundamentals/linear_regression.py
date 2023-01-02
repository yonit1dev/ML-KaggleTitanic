import numpy as np
from pandas import read_csv
from sklearn.linear_model import LinearRegression

# Relationship between gross revenue in a month vs amount of money spent on advertising that month
# Predicts the gross revenue
# Implemented manually and using scikit-learn


# Common utilities


def read_data(x_header, y_header):
    path = input("Path to CSV File: ")
    frame = read_csv(path)
    return frame[x_header].values, frame[y_header].values


def read_input_amount():
    return float(input("Enter amount: "))


x_train, y_train = read_data('Amount(Thousands birr)',
                             'Gross Revenue(Millions birr)')


# Numpy implementation
slope = np.cov(x_train, y_train, ddof=1)[0][1] / np.var(x_train, ddof=1)
intercept = y_train.mean() - (slope * x_train.mean())

np_amount = read_input_amount()
print(f"Slope: {slope}, Intercept: {intercept}")
np_revenue = (slope * np_amount) + intercept

print(f'Predicted Gross Revenue - Numpy Model: {round(np_revenue, 3)}')

# Scikit-Learn Implementation

regressor = LinearRegression()
regressor.fit(x_train.reshape(-1, 1), y_train.reshape(-1, 1))

sci_amount = read_input_amount()
sci_revenue = regressor.predict(np.array(sci_amount).reshape(-1, 1))[0][0]

print(f'Predicted Gross Revenue - Scikit Model: {round(sci_revenue, 3)}')
