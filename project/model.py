import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pylab as plot

# Reading the training data
frame = pd.read_csv("./data/train.csv")

# Check for the right amount of columns and rows
print(frame.shape == (891, 12))

# Cleansing some data
frame['Age'] = frame['Age'].fillna(frame['Age'].mean())
frame = frame.drop(columns="Cabin", axis=1)


frame['Succumb'] = 1 - frame['Survived']
