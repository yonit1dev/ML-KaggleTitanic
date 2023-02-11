import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

frame = pd.read_csv("../data/points.csv")

# choose k = 2
km = KMeans(n_clusters=2, n_init="auto")
y_predicted = km.fit_predict(frame[['X', 'Y']])
centroids = km.cluster_centers_

print(centroids)

# frame['cluster'] = y_predicted


# frame_c1 = frame[frame.cluster == 0]
# frame_c2 = frame[frame.cluster == 1]

# plt.scatter(centroids[0][0], centroids[0][1], color="black")
# plt.scatter(centroids[1][0], centroids[1][1], color="yellow")
# plt.scatter(frame_c1.X, frame_c1['Y'], color="red", marker="*")
# plt.scatter(frame_c2.X, frame_c2['Y'], color="green", marker="*")

# plt.xlabel('x-values')
# plt.ylabel('y-values')
# plt.title("ML Assignment 1 - Part Two")

plt.show()
