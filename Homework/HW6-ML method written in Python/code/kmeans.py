import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from Data import Data
import math


"""
CS383: Hw6
Instructor: Ian Gemp
TAs: Scott Jordan, Yash Chandak
University of Massachusetts, Amherst

README:

Feel free to make use of the function/libraries imported
You are NOT allowed to import anything else.

Following is a skeleton code which follows a Scikit style API.
Make necessary changes, where required, to get it correctly running.

Note: Running this empty template code might throw some error because 
currently some return values are not as per the required API. You need to
change them.

Good Luck!
"""


# ================= Helper Function for Plotting ====================
def plot_centroids(centroids, marker='o', scale=1):
    for c, centroid in enumerate(centroids):
        plt.plot([centroid[0]], [centroid[1]], color=cm(1. * c / K), marker=marker,
                 markerfacecolor='w', markersize=10*scale, zorder=scale)

# ===================================================================


class kmeans():
    def __init__(self, K):
        self.K = K      # Set the number of clusters

        np.random.seed(1234)
        self.centroids = np.random.rand(K, 2) # Initialize the position for those cluster centroids
        #print(self.centroids)
        # Plot initial centroids with 'o'
        plot_centroids(self.centroids, marker='o')

    def fit(self, X, iterations=10):
        """
        :param X: Input data, shape: (N,2)
        :param iterations: Maximum number of iterations, Integer scalar
        :return: None
        """
        self.C = -np.ones(np.shape(X)[0],dtype=int) # Initializing which center does each sample belong to

        # WRITE the required CODE for learning HERE
        #print(self.C)
        for itr in range(iterations):
            self.C = self.assign_clusters(X, self.C, self.centroids)
            self.centroids = self.update_centroids(X, self.C, self.centroids)

        return 0


    def update_centroids(self, X, C, centroids):
        """
        :param X: Input data, shape: (N,2)
        :param C: Assigned clusters for each sample, shape: (N,)
        :param centroids: Current centroid positions, shape: (K,2)
        :return: Update positions of centroids, shape: (K,2)

        Recompute centroids
        """
        # WRITE the required CODE HERE and return the computed values
        #C and K have the same length

        centrals = np.zeros((K, 2))
        means = [np.array([[0,0]])] * K
        for i in range(len(C)):
            means[int(C[i])] = np.append(means[int(C[i])], [X[i]], axis = 0)

        for mean_index in range(len(means)):
            temp = np.delete(means[mean_index], 0, axis = 0)
            x = np.mean(temp[:, 0])
            y = np.mean(temp[:, 1])
            centrals[mean_index] = [x, y]
        return centrals



    def assign_clusters(self, X, C, centroids):
        """
        :param X: Input data, shape: (N,2)
        :param C: Assigned clusters for each sample, shape: (N,)
        :param centroids: Current centroid positions, shape: (K,2)
        :return: New assigned clusters for each sample, shape: (N,)

        Assign data points to clusters
        """
        # WRITE the required CODE HERE and return the computed values
        new_clusters = np.zeros(len(X))

        for x in range(len(X)):
            new_centroid = 0
            dist = self.calculateDistance(X[x], centroids[0])
            for i in range(len(centroids)):
                temp = self.calculateDistance(X[x], centroids[i])
                if temp < dist:
                    new_centroid = i
                    dist = temp
            new_clusters[x] = new_centroid

        return new_clusters


    def calculateDistance(self, p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        return dist


    def get_clusters(self):
        """
        *********** DO NOT EDIT *******
        :return: assigned clusters and centroid locaitons, shape: (N,), shape: (K,2)
        """
        return self.C, self.centroids

if __name__ == '__main__':
    cm = plt.get_cmap('gist_rainbow') # Color map for plotting

    # Get data
    data = Data()
    X = data.get_kmeans_data()

    # Compute clusters for different number of centroids
    for K in [3]:
        # K-means clustering
        model = kmeans(K)
        model.fit(X)
        C, centroids = model.get_clusters()
        temp = model.update_centroids(X,C,centroids)
        # Plot final computed centroids with '*'
        plot_centroids(centroids, marker='*', scale=2)
        print(temp)
        # Plot the sample points, with their color representing the cluster they belong to
        for i, x in enumerate(X):
            plt.plot([x[0]], [x[1]], 'o', color=cm(1. * C[i] / K), zorder=0)
        plt.axis('square')
        plt.axis([0, 1, 0, 1])
        plt.savefig('figures/Q3_' + str(K) + '.png')
        plt.close()