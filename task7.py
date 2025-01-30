import math
import numpy
import numpy as np
import PIL
from matplotlib import pyplot as plt
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import sklearn.metrics as metrics
from scipy.optimize import linear_sum_assignment

def get_initial_centroids(data_set):
    centroids = np.zeros((len(data_set.target_names),64))
    for i in range(len(data_set.target_names)):
        centroid = np.zeros(64)
        for j in range(len(data_set)):
            if(data_set.target[j]==i):
                centroid = np.array(data_set.data[j])
                break

        centroids[i,:] = centroid[:]
    return centroids

digits = load_digits()
print(digits.target_names)

centroids = get_initial_centroids(digits)
print(centroids.shape)

pca = PCA(n_components=np.unique(digits.target_names).size).fit(digits.data)

# k_means = KMeans(init=centroids,n_clusters=np.unique(digits.target_names).size).fit(digits.data)
# k_means_predict = KMeans(init=centroids, n_clusters=np.unique(digits.target_names).size).fit_predict(digits.data)

k_means = KMeans(n_clusters=np.unique(digits.target_names).size).fit(digits.data)
k_means_predict = KMeans(n_clusters=np.unique(digits.target_names).size).fit_predict(digits.data)


inertia = k_means.inertia_
print(f"Внутрикластерное расстояние: {inertia}")

centroids = k_means.cluster_centers_
inter_cluster_distances = 0
# Проходим по всем парам кластеров
for i in range(np.unique(digits.target_names).size):
    for j in range(i + 1, np.unique(digits.target_names).size):
        # Вычисляем евклидово расстояние между центроидами i и j
        distance = np.linalg.norm(centroids[i] - centroids[j])
        inter_cluster_distances += distance
print("Межкластерные расстояния:")
print(inter_cluster_distances)

conf_matrix = metrics.confusion_matrix(digits.target, k_means.labels_)

# Majority voting
cluster_to_class = np.argmax(conf_matrix, axis=0)

mapped_labels = np.array([cluster_to_class[label] for label in k_means.labels_])

# #Hungarian algorithm
# row_ind, col_ind = linear_sum_assignment(-conf_matrix)
#
# mapped_labels = np.zeros_like(k_means.labels_)
# for cluster, true_class in zip(col_ind, row_ind):
#     mapped_labels[k_means.labels_ == cluster] = true_class

corrected_conf_matrix = metrics.confusion_matrix(digits.target, mapped_labels)

print("Матрица:\n",corrected_conf_matrix)