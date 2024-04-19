import numpy as np
from scipy.spatial.distance import pdist, squareform
import args
import pickle
import matplotlib.pyplot as plt
from scipy.special import expit
from scipy.stats import bernoulli
import networkx as nx

def create_simuA(N, K):
# N = args.num_points
# K = args.num_clusters
# D = args.hidden2_dim
    np.random.seed(42)

    delta = 0.95
    mu1 = [0, 0]
    mu2 = [delta * 1.5, delta * 1.5]
    mu3 = [-1.5 * delta, delta * 1.5]
    z_mu = np.concatenate((mu1,mu2,mu3), axis=0)
    # mu4 = [3, 0]
    # mu5 = [-3, 0]
    sigma1 = [[0.1, 0],[0, 0.1]]
    sigma2 = [[0.3, 0],[0, 0.3]]
    sigma3 = [[0.1, 0],[0, 0.1]]
    z_log_sigma = np.concatenate((sigma1,sigma2,sigma3), axis=0)
    # sigma4 = [[0.1, 0],[0, 0.1]]
    # sigma5 = [[0.2, 0],[0, 0.2]]
    x1 = np.random.multivariate_normal(mu1, sigma1, N//K)
    x2 = np.random.multivariate_normal(mu2, sigma2, N//K)
    x3 = np.random.multivariate_normal(mu3, sigma3, N-2*(N//K))
    # x4 = np.random.multivariate_normal(mu4, sigma4, 120)
    # x5 = np.random.multivariate_normal(mu5, sigma5, 120)


    # f, ax = plt.subplots(1,figsize=(8,8))
    # ax.scatter(x1[:,0], x1[:,1], color = '#7294d4')
    # ax.scatter(x2[:,0], x2[:,1], color = '#fdc765')
    # ax.scatter(x3[:,0], x3[:,1], color = '#869f82')
    # # ax.scatter(x4[:,0], x4[:,1], color = 'y')
    # # ax.scatter(x5[:,0], x5[:,1], color = 'purple')
    # ax.set_title("Original Embeddings of Scenario A (Delta=0.5)", fontsize=18)
    # plt.show()

    positions = np.concatenate((x1,x2,x3), axis=0)
    # np.savetxt('emb_3clusters.txt', X)
    # np.savetxt('mu_3clusters.txt', z_mu)
    # np.savetxt('cov_3clusters.txt', z_log_sigma)
    Label1 = np.repeat(0, N//K)
    Label2 = np.repeat(1, N//K)
    Label3 = np.repeat(2, N-2*(N//K))
    # Label4 = np.repeat(3, 120)
    # Label5 = np.repeat(4, 120)
    Label = np.concatenate((Label1,Label2,Label3), axis=0)


    dst = pdist(positions, 'euclidean')
    dst = squareform(dst)

    alpha1 = -3.5
    gamma1 = 0.1  # -3.5-0.1d
    alpha2 = -0.2
    gamma2 = 0.5  # -0.2-0.5d
    # same as a single layer in deepLPM
    alpha3 = 0.2
    gamma3 = 1  # 0.2-d

    A1 = np.zeros((N, N))
    A2 = np.zeros((N, N))
    A3 = np.zeros((N, N))

    for i in range(N-1):
        for j in range(i+1, N):
            prob1 = expit(alpha1 - gamma1 * dst[i,j])
            A1[i,j] = A1[j,i] = bernoulli.rvs(prob1, loc=0, size=1)

            prob2 = expit(alpha2 - gamma2 * dst[i,j])
            A2[i,j] = A2[j,i] = bernoulli.rvs(prob2, loc=0, size=1)

            prob3 = expit(alpha3 - gamma3 * dst[i, j])
            A3[i, j] = A3[j, i] = bernoulli.rvs(prob3, loc=0, size=1)


    # # test multi-layer
    # adj_matrices = [A1, A2, A3]
    # test single layer
    adj_matrices = [A2, A3]
    print("sparsity A1:", np.sum(A1) / (N * N))
    print("sparsity A2:", np.sum(A2) / (N * N))
    print("sparsity A3:", np.sum(A3) / (N * N))

    # # Generate a graph for visualization
    # G = nx.Graph()
    #
    # # Add nodes with positions
    # for i, pos in enumerate(positions):
    #     G.add_node(i, pos=pos)
    #
    # def add_edges_from_matrix(G, A, color):
    #     N = len(positions)
    #     for i in range(N):
    #         for j in range(i + 1, N):
    #             if A[i, j] == 1:
    #                 G.add_edge(i, j, color=color)
    #
    # # Assuming A1, A2, A3 are your adjacency matrices
    # add_edges_from_matrix(G, A1, 'blue')
    # add_edges_from_matrix(G, A2, 'red')
    # add_edges_from_matrix(G, A3, 'green')
    #
    # # Setup for subplot visualization
    # fig, axs = plt.subplots(1, 4, figsize=(20, 5))  # 4 subplots: 3 individual layers + 1 combined
    #
    # # Function to draw a specific layer
    # def draw_layer(ax, color, title):
    #     pos = nx.get_node_attributes(G, 'pos')
    #     edges = [(u, v) for u, v, d in G.edges(data=True) if d['color'] == color]
    #     nx.draw_networkx_nodes(G, pos, node_size=50, node_color='gray', ax=ax)
    #     nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=color, width=2, ax=ax)
    #     ax.set_title(title)
    #
    # # Draw each layer on its own subplot
    # draw_layer(axs[0], 'blue', 'Layer 1 (Blue)')
    # draw_layer(axs[1], 'red', 'Layer 2 (Red)')
    # draw_layer(axs[2], 'green', 'Layer 3 (Green)')
    #
    # # Draw all layers combined on the last subplot
    # pos = nx.get_node_attributes(G, 'pos')
    # nx.draw_networkx_nodes(G, pos, node_size=50, node_color='gray', alpha=0.8, ax=axs[3])
    # colors = nx.get_edge_attributes(G, 'color').values()
    # nx.draw_networkx_edges(G, pos, edge_color=list(colors), width=2, ax=axs[3])
    # axs[3].set_title('All Layers Combined')
    #
    # plt.tight_layout()
    # plt.show()


    # np.savetxt('adj_simuA_test.txt', A)
    # np.savetxt('label_simuA_test.txt', Label)
    # f.savefig("C:/Users/Dingge/Desktop/results/emb_orig_A.pdf", bbox_inches='tight')

    return adj_matrices, Label

# A, Label = create_simu(args.num_points, args.num_clusters)
# A, Label = create_simuA(100, 3)

############## loading and manipulatind docs and vocabulary  ###############
# dct = pickle.load(open('dizionario_2texts.pkl', 'rb'))
# dctn = dct.token2id
# V = len(dctn)
#
# with open('sim_data_docs_deeplsm_2texts', 'rb') as fp:
#     docs = pickle.load(fp)
# # num version of docs
# ndocs = []
# for doc in range(len(docs)):
#     tmp = []
#     for word in docs[doc]:
#         tmp.append(dctn[word])
#     ndocs.append(tmp)
# # complete dtm for row
# cdtm = []
# for idx in range(len(ndocs)):
#     cdtm.append(np.bincount(ndocs[idx], minlength=V))
# cdtm = np.asarray(cdtm, dtype='float32')
#
# edges = np.zeros((args.num_points, args.num_points, V))
# clr = np.where(A == 1)[0]
# clc = np.where(A == 1)[1]
# for i in range(len(clr)):
#     edges[clr[i],clc[i],:] = edges[clc[i],clr[i],:] = cdtm[args.num_points*clr[i]+clc[i],:]
#
# with open('edges_simu_3clusters_2texts_delta0.4', 'wb') as fp:
#     pickle.dump(edges, fp)
#
# # with open('edges_simu_3clusters_2texts', 'rb') as fp:
# #     edges_test = pickle.load(fp)
# # sum = np.sum(edges_test, axis=1)
