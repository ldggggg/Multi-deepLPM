import numpy as np
import scipy.io as sio
import args


def create_ACM():
    data = sio.loadmat("D:/Work/Benchmark/WWW2020-O2MAC-master/data/ACM3025.mat")
    labels = np.argmax(data['label'], axis=1)
    A1, A2 = data['PAP'], data['PLP']
    adj_matrices = [A1, A2]
    feat_matrix = data['feature'].astype(float)

    N = args.num_points
    print(args.dataset)
    print(N)
    print("sparsity A1:", np.sum(A1) / (N * N))
    print("sparsity A2:", np.sum(A2) / (N * N))

    return adj_matrices, labels

# create_ACM()