import numpy as np
import random
# Data structure of the K-Means algorithm
class Kmeans:
    def __init__(self, X, K):
        # Data - this is a numpy matrix of data points
        self.X = X
        # Number of clusters to be split into
        self.K = K
        # Initialize centroids - choose K random non-repeating elements from X
        unique_list = list(np.unique(self.X, axis = 0))
        random_sample = random.sample(unique_list, self.K)
        self.centroids = np.array(random_sample)

    def closest_centroids(self):
        # Let N be the number of data points, returns an array A of length N where
        # A[i] is the index of the centroid that X[i] is closest to.
        
        # Computing the distance between 
        # f1 = self.X
        # f2 = self.centroids

        # dist = sqrt(a-b)
        # where a = ||self.X||^2 + ||self.centroids||^2^T 
        # and b= 2*f1* f2^T
         
        data_sqrd = np.sum((self.X)**2, axis=1, keepdims= True)   #||self.X||^2 
        c_sqrd = np.sum((self.centroids)**2, axis=1) #||self.centroids||^2 
        a = data_sqrd + np.transpose(c_sqrd)  #a = ||self.X||^2 + ||self.centroids||^2 

        
        f1,f2 = self.X, self.centroids
        b = 2 * np.matmul(f1,np.transpose(f2)) #b= 2*f1* f2^T

        dist = np.reshape(a-b,(-1,self.K)) #reshape towards one shape dimension and cetroids
        return np.argmin(dist, axis = 1) #return closest centroid 

    def run(self, max_iter):
        # Performs the K-means algorithm on X
        # If the points do not converge, the algorithm will terminate after max_iter number of iterations.
        for i in range(max_iter):
            centroid_indices = self.closest_centroids() # calculate closest centroids
            # Computes the centroids based on the current cluster
            avgs = np.array([np.mean(self.X[centroid_indices == it], 
                axis=0) 
                for it in range(self.K)])
            if np.all(avgs == self.centroids):
                # A convergence occurred!
                print("Converged at Iteration: ", i)
                break
            else:
                # Otherwise, update the current centroids to the new averages
                self.centroids = avgs
        
        return (self.centroids, centroid_indices)