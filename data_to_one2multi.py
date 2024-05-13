import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.special import expit
from scipy.stats import bernoulli
from scipy.io import savemat
import os


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
    # Directory to save output files
    output_dir = "one2multi_data"
    os.makedirs(output_dir, exist_ok=True)

    savemat(f"{output_dir}/simuA_{seed}_{zeta}.mat", data_dict)
    print("Data saved to one2multi_data.mat")

# # Usage
# zetas = [0.3, 0.2]  # [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.95]
# seeds = np.arange(0, 25).tolist()  # [0]
#
# for zeta in zetas:
#     for seed in seeds:
#         create_simuA_and_save(600, 3, zeta, seed)


def create_simuB_and_save(N, K, zeta=1.0, seed=None):
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
    # Convert labels to one-hot encoding
    labels_one_hot = np.eye(K)[c]  # Method 1: Using np.eye and indexing

    A1 = np.zeros((N, N))
    A2 = np.zeros((N, N))
    A3 = np.zeros((N, N))
    for i in range(N - 1):
        for j in range(i + 1, N):
            prob1 = Pi1[c[i], c[j]]
            A1[i, j] = A1[j, i] = bernoulli.rvs(prob1, loc=0, size=1)

            prob2 = Pi2[c[i], c[j]]
            A2[i, j] = A2[j, i] = bernoulli.rvs(prob2, loc=0, size=1)

            prob3 = Pi3[c[i], c[j]]
            A3[i, j] = A3[j, i] = bernoulli.rvs(prob3, loc=0, size=1)

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
    # Directory to save output files
    output_dir = "one2multi_data"
    os.makedirs(output_dir, exist_ok=True)

    savemat(f"{output_dir}/simuB_{seed}_{zeta}.mat", data_dict)
    print("Data saved to one2multi_data.mat")

# # Usage
# zetas = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]  # [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.95]
# seeds = np.arange(0, 25).tolist()  # [0]
#
# for zeta in zetas:
#     for seed in seeds:
#         create_simuB_and_save(600, 3, zeta, seed)


def create_simuC_and_save(N, K, seed=None):

    if seed is not None:
        np.random.seed(seed)

    x = np.random.uniform(-1,1,N//K)
    c = np.random.multinomial(1, [0.5,0.5], size=N//K)
    c = np.argmax(c, axis=1)
    y = np.sqrt(1 - x**2) + np.random.normal(0,0.1,N//K)
    y[c==1] = -y[c==1]

    x2 = np.random.uniform(-5,5,N//K)
    c = np.random.multinomial(1, [0.5,0.5], size=N//K)
    c = np.argmax(c, axis=1)
    y2 = np.sqrt(25-x2**2) + np.random.normal(0,0.1,N//K)
    y2[c==1] = -y2[c==1]

    x3 = np.random.uniform(-10,10,N-2*(N//K))
    c = np.random.multinomial(1, [0.5,0.5], size=N-2*(N//K))
    c = np.argmax(c, axis=1)
    y3 = np.sqrt(100-x3**2) + np.random.normal(0,0.1,N-2*(N//K))
    y3[c==1] = -y3[c==1]

    import matplotlib.pyplot as plt
    f, ax = plt.subplots(1, figsize=(8, 8))
    ax.scatter(x, y, color='#7294d4')
    ax.scatter(x2, y2, color='#fdc765')
    ax.scatter(x3, y3, color='#869f82')
    ax.set_title("Original Embeddings of Scenario C", fontsize=18)
    # f.savefig("C:/Users/Dingge/Desktop/results/emb_orig_C.pdf", bbox_inches='tight')

    K1 = np.concatenate((x.reshape(-1,1),y.reshape(-1,1)), axis=1)
    K2 = np.concatenate((x2.reshape(-1,1),y2.reshape(-1,1)), axis=1)
    K3 = np.concatenate((x3.reshape(-1,1),y3.reshape(-1,1)), axis=1)

    C= np.concatenate((K1,K2,K3), axis=0)
    # np.savetxt('emb_3clusters.txt', K)

    Label1 = np.repeat(0, N//K)
    Label2 = np.repeat(1, N//K)
    Label3 = np.repeat(2, N-2*(N//K))
    Label = np.concatenate((Label1,Label2,Label3), axis=0)
    labels_one_hot = np.eye(K)[Label]  # Method 1: Using np.eye and indexing


    dst = pdist(C, 'euclidean')
    dst = squareform(dst)

    alpha1 = -1.5
    gamma1 = 0.1  # -1.5-0.1d
    alpha2 = -0.2
    gamma2 = 0.5  # -0.2-0.5d
    # same as a single layer in deepLPM
    alpha3 = 0.2
    gamma3 = 1  # 0.2-d

    A1 = np.zeros((N, N))
    A2 = np.zeros((N, N))
    A3 = np.zeros((N, N))

    for i in range(N - 1):
        for j in range(i + 1, N):
            prob1 = expit(alpha1 - gamma1 * dst[i, j])
            A1[i, j] = A1[j, i] = bernoulli.rvs(prob1, loc=0, size=1)

            prob2 = expit(alpha2 - gamma2 * dst[i, j])
            A2[i, j] = A2[j, i] = bernoulli.rvs(prob2, loc=0, size=1)

            prob3 = expit(alpha3 - gamma3 * dst[i, j])
            A3[i, j] = A3[j, i] = bernoulli.rvs(prob3, loc=0, size=1)

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
    # Directory to save output files
    output_dir = "one2multi_data"
    os.makedirs(output_dir, exist_ok=True)

    savemat(f"{output_dir}/simuC_{seed}.mat", data_dict)
    print("Data saved to one2multi_data.mat")

# Usage
seeds = np.arange(0, 25).tolist()

for seed in seeds:
    create_simuC_and_save(600, 3, seed)