import time
import numpy as np

def pca_naive(X, K):
    """
    PCA -- naive version

    Inputs:
    - X: (float) A numpy array of shape (N, D) where N is the number of samples,
         D is the number of features
    - K: (int) indicates the number of features you are going to keep after
         dimensionality reduction

    Returns a tuple of:
    - P: (float) A numpy array of shape (K, D), representing the top K
         principal components
    - T: (float) A numpy vector of length K, showing the score of each
         component vector
    """

    ###############################################
    #TODO: Implement PCA by extracting eigenvector#
    ###############################################
    # X = X_patch
    X = (X - np.mean(X, axis = 0))/ np.std(X, axis = 0)
    cov = np.cov(X.T)
    T, P = np.linalg.eig(cov)
    sortedOrder = np.argsort(T)[::-1]
    T = T[sortedOrder[np.arange(0,K)]]
    P = P[:,sortedOrder[np.arange(0,K)]]
    P = P.T
    # ev = T
    # eig = P


    ###############################################
    #              End of your code               #
    ###############################################
    
    return (P, T)
