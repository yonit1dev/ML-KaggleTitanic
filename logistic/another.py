import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

from scipy.special import expit

frame = pd.read_csv("./data/heart.csv")

# Picked MaxHR feature for checking correlation
feature = ["MaxHR"]

x = frame[feature]
y = frame["HeartDisease"]

X = x.astype('float32')
scaler = StandardScaler().fit(X)
x_scaled = scaler.transform(X)


# splitting data to training and test set
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, random_state=1, test_size=0.25)
log_reg = LogisticRegression()
log_reg.fit(x_train, y_train)

# Prediction
y_pred = log_reg.predict(x_test)

loss = expit(x_test * (-log_reg.coef_) + log_reg.intercept_).ravel()

plt.scatter(x_test, loss, label="Logistic Regression Model", color="black", linewidth=3)

# plot test and prediction
plt.scatter(x_test, y_test, color="red")
plt.scatter(x_test, y_pred, color="green")

# Visualize using confusion matrix
cm = confusion_matrix(y_test, y_pred)
cm_display = ConfusionMatrixDisplay(cm).plot()

plt.show()

