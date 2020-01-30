# -*- coding: utf-8 -*-
import numpy as np
from scipy.spatial.distance import cdist
from scipy.cluster.vq import kmeans2


def llc(X, C=None, k=None, beta=1e-6, **kwargs):
    """
    Implements Approximate Locally Linear Coding
    Inputs:
        X: (N x d) numpy array
        C: (Default: None)
            integer: number of anchor points (kmeans used to obtain codebook)
            OR
            (c x d) array of anchor points (codebook)
        k: (Default: None) Number of nearest neighbors for sparsity. If k > c or k<1, then k is changed to c
        beta: regularization parameter (lambda in the paper)
    Outputs:
        (G,C,distortion)
            G: Gamma coefficients (N x c) numpy array
            C: Codebook (c x d)
            Y: The transformed points (N x d) Y = G*C
    """
    if type(C) == type(0):
        C, _ = kmeans2(X, C, **kwargs)

    assert X.shape[1] == C.shape[1]
    N, d = X.shape
    c, _ = C.shape
    if k is None or k < 1 or k > c:
        print("Warning: k set to ", c)
        k = c
    D = cdist(X, C, 'euclidean')
    I = np.zeros((N, k), dtype=int)
    for i in range(N):
        d = D[i, :]
        idx = np.argsort(d)
        I[i, :] = idx[:k]

    II = np.eye(k)
    G = np.zeros((N, c))  # Gammas
    ones = np.ones((k, 1))
    for i in range(N):
        idx = I[i, :]
        z = C[idx, :] - np.tile(X[i, :], (k, 1))  # shift ith pt to origin
        Z = np.dot(z, z.T)  # local covariance
        Z = Z + II * beta * np.trace(Z) # regularlization (K>D)
        w = np.linalg.solve(Z, ones)  # np.dot(np.linalg.inv(Z), ones)
        w = w / np.sum(w)  # enforce sum(w)=1
        G[i, idx] = w.ravel()
    Y = np.dot(G, C)
    return Y

