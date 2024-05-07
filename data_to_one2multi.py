import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.special import expit
from scipy.stats import bernoulli
from scipy.io import savemat


def create_simuA_and_save(N, K, zeta=0.95, seed=None):
    if seed is not None:
        np.random.seed(seed)

    # Data generation logic (same as before)
    mu1 = [0, 0]
    mu2 = [zeta * 1.5, zeta * 1.5]
    mu3 = [-1.5 * zeta, zeta * 1.5]
    sigma1 = [[0.1, 0], [0, 0.1]]
    sigma2 = [[0.3, 0], [0, 0.3]]
    sigma3 = [[0.1, 0], [0, 0.1]]
    x1 = np.random.multivariate_normal(mu1, sigma1, N // K)
    x2 = np.random.multivariate_normal(mu2, sigma2, N // K)
    x3 = np.random.multivariate_normal(mu3, sigma3, N - 2 * (N // K))

    positions = np.concatenate((x1, x2, x3), axis=0)
    labels = np.concatenate((np.repeat(0, N // K), np.repeat(1, N // K), np.repeat(2, N - 2 * (N // K))), axis=0)
    # Convert labels to one-hot encoding
    labels_one_hot = np.eye(K)[labels]  # Method 1: Using np.eye and indexing

    dst = pdist(positions, 'euclidean')
    dst = squareform(dst)

    alpha1, gamma1 = -1.5, 0.1
    alpha2, gamma2 = -0.2, 0.5
    alpha3, gamma3 = 0.2, 1

    A1, A2, A3 = [np.zeros((N, N)) for _ in range(3)]
    for i in range(N - 1):
        for j in range(i + 1, N):
            A1[i, j] = A1[j, i] = bernoulli.rvs(expit(alpha1 - gamma1 * dst[i, j]))
            A2[i, j] = A2[j, i] = bernoulli.rvs(expit(alpha2 - gamma2 * dst[i, j]))
            A3[i, j] = A3[j, i] = bernoulli.rvs(expit(alpha3 - gamma3 * dst[i, j]))

    # Index calculations
    train_idx = np.arange(int(0.8 * N)).reshape(1,-1)
    val_idx = np.arange(int(0.8 * N), int(0.9 * N)).reshape(1,-1)
    test_idx = np.arange(int(0.9 * N), N).reshape(1,-1)

    # Dictionary for saving
    data_dict = {
        "A1": A1,
        "A2": A2,
        "A3": A3,
        "features": np.eye(N),
        "label": labels_one_hot,
        "test_idx": test_idx,
        "train_idx": train_idx,
        "val_idx": val_idx
    }

    # Save to .mat file
    savemat("simuA.mat", data_dict)
    print("Data saved to network_data.mat")

# Usage
create_simuA_and_save(600, 3, 0.95, seed=42)
