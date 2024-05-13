import numpy as np
from scipy.stats import bernoulli
import args
import pickle

def create_simuB(N, K, zeta=1.0, seed=None):
# N = args.num_points
# K = args.num_clusters
# D = args.hidden2_dim
# K = 3
    print(N, K, zeta, seed)
    if seed is not None:
        np.random.seed(seed)

    Pi1 = np.zeros((K, K))
    Pi2 = np.zeros((K, K))
    Pi3 = np.zeros((K, K))

    a1 = 0.25
    b1 = 0.01 + (1 - zeta) * (a1 - 0.01)
    a2 = 0.1
    b2 = 0.03 + (1 - zeta) * (a2 - 0.03)
    a3 = 0.15
    b3 = 0.02 + (1 - zeta) * (a3 - 0.02)

    Pi1[0, 0] = Pi1[1, 2] = Pi1[2, 1] = b1
    Pi1[0, 1] = Pi1[0, 2] = Pi1[1, 0] = Pi1[1, 1] = Pi1[2, 0] = Pi1[2, 2] = a1
    Pi2[0, 0] = Pi2[1, 2] = Pi2[2, 1] = b2
    Pi2[0, 1] = Pi2[0, 2] = Pi2[1, 0] = Pi2[1, 1] = Pi2[2, 0] = Pi2[2, 2] = a2
    Pi3[0, 0] = Pi3[1, 2] = Pi3[2, 1] = b3
    Pi3[0, 1] = Pi3[0, 2] = Pi3[1, 0] = Pi3[1, 1] = Pi3[2, 0] = Pi3[2, 2] = a3

    Rho = [0.1, 0.45, 0.45]
    c = np.random.multinomial(1, Rho, size=N)
    c = np.argmax(c, axis=1)

    A1 = np.zeros((N, N))
    A2 = np.zeros((N, N))
    A3 = np.zeros((N, N))
    for i in range(N-1):
        for j in range(i+1, N):
            prob1 = Pi1[c[i], c[j]]
            A1[i,j] = A1[j,i] = bernoulli.rvs(prob1, loc=0, size=1)

            prob2 = Pi2[c[i], c[j]]
            A2[i, j] = A2[j, i] = bernoulli.rvs(prob2, loc=0, size=1)

            prob3 = Pi3[c[i], c[j]]
            A3[i, j] = A3[j, i] = bernoulli.rvs(prob3, loc=0, size=1)

    adj_matrices = [A1, A2, A3]  # A3, A2,
    print("sparsity A1:", np.sum(A1) / (N * N))
    print("sparsity A2:", np.sum(A2) / (N * N))
    print("sparsity A3:", np.sum(A3) / (N * N))

    return adj_matrices, c

# A, Label = create_simuB(30, 3, 0.9, 1)