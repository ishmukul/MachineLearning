# Clustering on faithful dataset.
# Dataset is visible as two big clusters.
# Idea is to use blackbox algorithms sklearn and compare results with algorithm written from scratch
#
# This file contains algorithm from scratch.

# Import functions
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler

from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_samples, silhouette_score

from timeit import default_timer as timer


# Define a plotting function. This data is a two column data, hence, a plotting function works nice on this. If
# columns are more than 2, visualization is difficult.
class Plot:
    def __init__(self, dpi=75, edge_color="k"):
        '''
        Initial parameters
        :param dpi: resolution of figure
        :param edge_color: Edge color of symbols
        '''
        self.dpi = dpi
        self.edge_color = edge_color

    def plot_fig(self, x, filename=""):
        '''
        Figure plotting
        :param x: Data
        :param filename: filename to save
        :return: None
        '''
        plt.figure(figsize=plt.figaspect(0.8), dpi=self.dpi)
        x1 = x[x.columns[0]]
        x2 = x[x.columns[1]]
        plt.scatter(x1, x2, c="g", edgecolor=self.edge_color)
        plt.xlabel('eruptions')
        plt.ylabel('waiting')
        if filename:
            plt.savefig(filename)

    def plot_cluster(self, x, label, centroid, filename="", scale=True):
        '''
        Plotting scaled data
        :param x: Data
        :param label: X and Y labels
        :param centroid: Coordinates of Cluster Centroids for plotting
        :param filename: Filename to save
        :param scale: Label names if data was scaled with MinMaxScaler
        :return: None
        '''
        plt.figure(figsize=plt.figaspect(0.8), dpi=self.dpi)
        x1 = x[x.columns[0]]
        x2 = x[x.columns[1]]
        plt.scatter(x1, x2, c=label.astype(np.float), edgecolor=self.edge_color)
        plt.scatter(centroid[:, 0], centroid[:, 1], marker="x", s=150, c="r")

        if scale:
            plt.xlabel('Eruptions (Normalized)')
            plt.ylabel('Waiting time (Normalized)')
        else:
            plt.xlabel('Eruptions')
            plt.ylabel('Waiting time')

        if filename:
            plt.savefig(filename)

    def plot_scree(self, inertia, filename=""):
        '''
        Plotting scree plot
        :param inertia: Inertia
        :param filename: Filename to save
        :return: None
        '''
        plt.figure(figsize=plt.figaspect(0.8), dpi=self.dpi)
        x2 = inertia
        x1 = range(1, len(inertia) + 1)
        plt.plot(x1, x2, '-o')
        plt.xlabel('Number of clusters')
        plt.ylabel('Inertia')
        plt.title("Scree plot for K-Means algorithm")

        if filename:
            plt.savefig(filename)


# Starting from scratch
# k_means
class KMeansScratch:
    def __init__(self, k=2, max_iterations=5):
        self.k = k
        self.max_iterations = max_iterations

    # Fitting function
    def fit(self, x):
        self.x = x

        # n_observations and m_features definition
        n, m = self.x.shape

        #  Select random centroids.
        # Two approaches:
        # 1) use random number generator
        self.centroids = np.zeros(self.k * m).reshape(self.k, m)
        for i in range(self.k):
            for j in range(m):
                self.centroids[i, j] = np.random.rand()
        # 2) Select random observation as centroids
        # self.centroids = (self.x.sample(n=self.k)).to_numpy()

        # Next step is to assign all the points to the closest cluster centroid and recompute centroids of newly formed
        # clusters. Repeat this step until convergence occurs.

        # Loop over max iterations
        for i in range(self.max_iterations):
            # Create a copy of original X vector
            self.xd = self.x.copy()
            # Calculate distance between each point and cluster center. Loop over number of clusters.
            for index in range(self.k):
                ed = np.sqrt(np.sum((self.x - self.centroids[index, :]) ** 2, axis=1))
                # Add distance column to the copy of X.
                self.xd[index] = ed

            # Create a small dataframe of distances and find minimum in each row. This column vector will be our labels.
            dd = km.xd.iloc[:, m:]
            self.label_ = dd.idxmin(axis=1)
            self.xd["Label"] = self.label_

            # Sort centroids by newly assigned labels and take mean of the X columns. This will give column wise
            # minimum and the new centroid.
            self.centroids = self.xd.groupby(["Label"]).mean().iloc[:, :m]  # m is number of features
            self.centroids = self.centroids.to_numpy()

            # Following lines are for generating plots for each step of calculation. Plots saved to kmeans_scratch
            # folder
            FName = "cluster_steps/cluster_kmeans_k%d_Step%d.png" % (self.k, i)
            plot.plot_cluster(self.xd, self.label_, self.centroids, filename=FName)
        return self


# Plotting options
plt.close('all')
# Instantiate Plt class
plot = Plot()

# Import dataset
data = pd.read_csv("data/faithful.csv")
X = data.iloc[:, range(data.shape[1])]

# Scaling features.
# MinMax scaler form sklearn will be used.
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(data=X_scaled)
X_scaled.columns = X.columns

# Plot original data for visualization. It is a two column data, so easy to visualize
FileName = "plots/data_raw.png"
plot.plot_fig(X, FileName)

# First, let us check for best value of number of clusters to be searched.
# Scree plot is an ideal choice for this.
# Scree plot is a plot of Inertia vs number of clusters

start_time = timer()
print("============================")
print("sklearn KMeans algorithm:")

inertia_km = []
for i in range(1, 9):
    kmeans_sklearn = KMeans(n_clusters=i, random_state=1).fit(X_scaled)
    # print(kmeans.labels_)  # Diagnostic
    labels_km = kmeans_sklearn.labels_
    centroids_km = kmeans_sklearn.cluster_centers_
    inertia_km.append(kmeans_sklearn.inertia_)
    print("Number of clusters searched = ", centroids_km.shape[0])
    print("Cluster distribution = ")
    print(kmeans_sklearn.cluster_centers_)
    print(pd.Series(labels_km).value_counts())
    # Plot and save figure
    FileName = ("plots/cluster_kmeans_k%d.png" % i)
    plot.plot_cluster(X_scaled, labels_km, centroids_km, filename=FileName)

FileName = "plots/kmeans_scree_plot.png"
plot.plot_scree(inertia_km, filename=FileName)
end_time = timer()
print("Time taken for cluster classification using sklearn.KMeans = %0.3f s.\n" % (end_time - start_time))

# Clustering using sklearn KMeans algorithm
# KMeans algorithm
start_time = timer()
print("============================")
print("sklearn KMeans algorithm:")
kmeans_sklearn = KMeans(n_clusters=2, random_state=1).fit(X_scaled)
# print(kmeans.labels_)  # Diagnostic
labels_km = kmeans_sklearn.labels_
centroids_km = kmeans_sklearn.cluster_centers_
print("Number of clusters searched = ", centroids_km.shape[0])
print("Cluster distribution = ")
print(kmeans_sklearn.cluster_centers_)
print(pd.Series(labels_km).value_counts())
# Plot ans save figure
FileName = 'plots/cluster_' + 'kmeans_sklearn_scaled.png'
plot.plot_cluster(X_scaled, labels_km, centroids_km, FileName)
end_time = timer()
print("Time taken for cluster classification using sklearn.KMeans = %0.3f s.\n" % (end_time - start_time))

# Clustering using Kmeans from scratch
start_time = timer()
print("============================")
print("KMeans algorithm (Scratch):")
km = KMeansScratch(k=2, max_iterations=5)
km.fit(X_scaled)
print("Number of clusters searched = ", km.centroids.shape[0])
print("Cluster Centroids")
print(km.centroids)
labels = km.label_
print("\nCluster distribution")
print(labels.value_counts())
FileName = 'plots/cluster_' + 'kmeans_scaled.png'
plot.plot_cluster(X_scaled, labels, km.centroids, filename=FileName)
end_time = timer()
print("Time taken for cluster classification using KMeans = %0.3f s.\n" % (end_time - start_time))
