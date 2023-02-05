# Assignment 1 Part Two - Implementation of K-Means Clustering based on a given dataset.
# Dataset can be found in data folder.

# a. Using python and relevant libraries, implement K-means clustering on these 48 observations with cluster numbers k=2, k=3, and k=4.
# b. For all values of k, calculate centroids and print; plot the scatter plot of the clusters; centroids should also be plotted with a different color or shape.
# c. Which value of k do you think is the best value? Provide your reason for your answer.

import numpy as np
from pandas import read_csv
import matplotlib.pyplot as plt
# from sklearn.cluster import KMeans
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import silhouette_score
# from sklearn.datasets import make_blobs


def read_data(x, y):
    path = "./data/points.csv"
    frame = read_csv(path)
    return frame[x].values, frame[y].values


def plot_data(x, y):
    plt.scatter(x, y, color="blue", marker="*")
    plt.xlabel("x - value")
    plt.ylabel("y - value")

    plt.title("ML - Assignment Sample Data for K-Means Clustering")

    plt.show()


x, y = read_data('X', 'Y')
plot_data(x, y)


# Soluton of part 'a'

# Using k = 2
# Using k = 3
# Using k = 4
