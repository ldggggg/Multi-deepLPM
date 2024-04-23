import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import itertools
import os
import numpy as np
import args
from sklearn.mixture import GaussianMixture
from sklearn.metrics.cluster import adjusted_rand_score
import matplotlib.pyplot as plt
from input_data import load_data
from sklearn.cluster import KMeans


os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
device = torch.device('cuda:0')

def glorot_init(input_dim, output_dim):
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = torch.rand(input_dim, output_dim, dtype = torch.float32) * 2 * init_range - init_range
    initial = initial.to(device)
    return nn.Parameter(initial)

def he_init(input_dim, output_dim):
    std = np.sqrt(2. / (input_dim + output_dim))
    initial = torch.randn(input_dim, output_dim, dtype=torch.float32) * std
    return nn.Parameter(initial.to(device))


# Graph convolutional layers
class GCN(nn.Module):
    def __init__(self, input_dim, output_dim, adj, activation=F.relu, **kwargs):
        super(GCN, self).__init__(**kwargs)
        self.weight = he_init(input_dim, output_dim)
        self.adj = adj
        self.activation = activation

    def forward(self, inputs):
        # print("X dim:", inputs.shape)
        x = torch.mm(inputs, self.weight)
        # print("XW dim:", x.shape)
        x = torch.mm(self.adj, x)
        # print("AXW dim:", x.shape)
        outputs = self.activation(x)
        return outputs


class MultiEncoder(nn.Module):
    def __init__(self, adj_matrices):
        """
        Initialize the multi-encoder model.

        :param adj_matrices: List of adjacency matrices for each encoder layer.
        :param input_dim: The input dimension.
        :param hidden_dim: The hidden dimension for GCN layers inside each encoder.
        :param output_dims: List of output dimensions for each encoder, defining the dimensionality of means and logstds.
        :param device: Device to run the model on ('cpu' or 'cuda').
        """
        super(MultiEncoder, self).__init__()
        self.encoders = nn.ModuleList()

        # Create an encoder for each adjacency matrix
        for i, adj in enumerate(adj_matrices):
            encoder = nn.ModuleDict({
                'base_gcn': GCN(args.input_dim, args.hidden_dim, adj),
                'gcn_mean': GCN(args.hidden_dim, args.output_dims[i], adj, activation=lambda x: x),
                'gcn_logstd': GCN(args.hidden_dim, args.output_dims[i], adj, activation=lambda x: x)
            })
            self.encoders.append(encoder)

        # Define MLP for aggregated mean and logstd
        total_output_dim = sum(args.output_dims)
        print("total_output_dim:", total_output_dim)
        self.mlp_mean = nn.Sequential(nn.Linear(total_output_dim, args.emb_dim), nn.ReLU(), nn.Linear(
            args.emb_dim, args.emb_dim))
        self.mlp_logstd = nn.Sequential(nn.Linear(total_output_dim, args.emb_dim), nn.ReLU(), nn.Linear(
            args.emb_dim, 1))

    def forward(self, X):
        self.all_means = []
        self.all_logstds = []

        for encoder in self.encoders:
            hidden = encoder['base_gcn'](X)
            mean = encoder['gcn_mean'](hidden)
            logstd = encoder['gcn_logstd'](hidden)
            self.all_means.append(mean)
            self.all_logstds.append(logstd)

        concatenated_means = torch.cat(self.all_means, dim=1)
        concatenated_logstds = torch.cat(self.all_logstds, dim=1)

        self.aggregated_mean = self.mlp_mean(concatenated_means)
        self.aggregated_logstd = self.mlp_logstd(concatenated_logstds)

        # self.aggregated_mean = torch.stack(self.all_means).mean(dim=0)
        # self.aggregated_logstd = torch.stack(self.all_logstds).mean(dim=0)

        gaussian_noise = torch.randn(X.size(0), args.emb_dim).to(device)
        sampled_z = gaussian_noise * torch.exp(self.aggregated_logstd / 2) + self.aggregated_mean

        return self.aggregated_mean, self.aggregated_logstd, sampled_z


class MultiDecoder(nn.Module):
    def __init__(self):
        super(MultiDecoder, self).__init__()
        self.alphas = nn.Parameter(torch.randn(args.num_layers))
        self.betas = nn.Parameter(torch.randn(args.num_layers, args.feature_dim))
        self.gammas = nn.Parameter(torch.randn(args.num_layers))

    def forward(self, Z, Y_list):
        # Y_list contains the edge features for each layer: [Y^(1), Y^(2), ..., Y^(L)]
        A_pred_list = []

        for l in range(args.num_layers):
            Y = Y_list[l]
            alpha = self.alphas[l]
            beta = self.betas[l]
            gamma = self.gammas[l]

            inner_product = torch.matmul(Z, Z.T)
            tnp = torch.sum(Z ** 2, axis=1).reshape(-1, 1).expand(size=inner_product.shape)

            # Assuming Y is a feature matrix where Y_ij is a row vector for pair (i, j)
            Y_beta = torch.matmul(Y, beta.unsqueeze(-1)).squeeze(-1)  # Linear combination of features and beta parameters

            # Compute the squared Euclidean distance and scale by gamma
            distance_term = gamma * (tnp - 2 * inner_product + tnp.T)

            # The final prediction for the adjacency matrix for this layer
            A_pred = torch.sigmoid(alpha + Y_beta - distance_term)
            A_pred_list.append(A_pred)

            # A_pred = torch.sigmoid(alpha + Y_beta - distance_term)
            # # print("Original A_pred min/max:", torch.min(A_pred).item(), torch.max(A_pred).item())
            #
            # # Manually clamping values using torch.where
            # A_pred_clamped = torch.where(A_pred > 1 - 1e-8, torch.full_like(A_pred, 1 - 1e-8), A_pred)
            # A_pred_clamped = torch.where(A_pred_clamped < 1e-8, torch.full_like(A_pred_clamped, 1e-8), A_pred_clamped)
            # # print("Manually clamped A_pred min/max:", torch.min(A_pred_clamped).item(),
            # #       torch.max(A_pred_clamped).item())
            # A_pred_list.append(A_pred_clamped)

        return A_pred_list


class MultiLPM(nn.Module):
    def __init__(self, adj_matrices):
        super(MultiLPM, self).__init__()
        self.adj_matrices = adj_matrices
        self.encoder = MultiEncoder(adj_matrices)
        self.decoder = MultiDecoder()

        # N * K, assuming N is the number of points (nodes) and K is the number of clusters
        self.delta = nn.Parameter(torch.FloatTensor(adj_matrices[0].size(0), args.num_clusters).fill_(0.1),
                                  requires_grad=False)

        # K
        self.pi_k = nn.Parameter(torch.FloatTensor(args.num_clusters).fill_(1) / args.num_clusters, requires_grad=False)

        # K * P, assuming P is the output dimension (size) of the embeddings
        self.mu_k = nn.Parameter(torch.FloatTensor(
            np.random.multivariate_normal(np.zeros(args.emb_dim), np.eye(args.emb_dim), args.num_clusters)),
                                 requires_grad=False)

        # K
        self.log_cov_k = nn.Parameter(torch.FloatTensor(args.num_clusters, 1).fill_(0.1), requires_grad=False)

    def pretrain(self, X, adj_labels, Y_list, labels):
        # Define an optimizer for the pretraining phase
        optimizer = Adam(itertools.chain(self.encoder.parameters(), self.decoder.parameters()), lr=args.pretrain_lr)

        for epoch in range(args.pretrain_epochs):
            # Forward pass to get the embeddings from the MultiEncoder
            z_mu, z_log_sigma, z = self.encoder(X)

            # Perform reconstruction using the Decoder
            A_pred_list = self.decoder(z, Y_list)  # Modify to add any other required arguments

            # Calculate the reconstruction loss for each layer and sum them up
            loss_list = []
            for A_pred, adj_label in zip(A_pred_list, adj_labels):
                # Ensure adj_label is on the same device as A_pred
                adj_label = adj_label.to(A_pred.device)
                loss = F.binary_cross_entropy(A_pred.view(-1), adj_label.to_dense().view(-1))
                kl_divergence = 0.5 / A_pred.size(0) * (
                            1 + 2 * z_log_sigma - z_mu ** 2 - torch.exp(z_log_sigma) ** 2).sum(1).mean()
                loss -= kl_divergence  # to train cora, we need to add the kl divergence
                loss_list.append(loss)
            # print(loss_list)

            # Take the average of the losses from all layers
            loss_total = sum(loss_list) / len(A_pred_list)

            # Backpropagation
            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()

            if (epoch + 1) % 1 == 0:
                print(f"Pretrain Epoch: {epoch + 1}/{args.pretrain_epochs}, Loss: {loss_total.item()}")

        # After pretraining, detach the embeddings and run KMeans clustering
        z_detached = z.detach().cpu().numpy()
        kmeans = KMeans(n_clusters=args.num_clusters).fit(z_detached)
        labelk = kmeans.labels_
        print("pretraining ARI Kmeans:", adjusted_rand_score(labels, labelk))

        # Initialize cluster parameters based on KMeans results
        self.delta.fill_(1e-8)
        seq = np.arange(0, len(self.delta))
        positions = np.vstack((seq, labelk))
        self.delta[positions] = 1.
        # print(self.delta)

        # self.mu_k.data = torch.from_numpy(kmeans.cluster_centers_).float().to(device)

        print('Pretraining completed.')

    # Functions for the initialization of cluster parameters
    def update_delta(self, mu_phi, log_cov_phi, pi_k, mu_k, log_cov_k, P):
        det = 1e-8
        KL = torch.zeros((args.num_points, args.num_clusters), dtype = torch.float32)  # N * K
        KL = KL.to(device)
        for k in range(args.num_clusters):
            log_cov_K = torch.ones_like(log_cov_phi) * log_cov_k[k]
            mu_K = torch.ones((args.num_points, mu_k.shape[1])).to(device) * mu_k[k]
            temp = P * (log_cov_K - log_cov_phi - 1) \
                   + P * torch.exp(log_cov_phi) / torch.exp(log_cov_K) \
                   + torch.norm(mu_K - mu_phi, dim=1, keepdim=True) ** 2 / torch.exp(log_cov_K)
            KL[:, k] = 0.5 * temp.squeeze()

        denominator = torch.sum(pi_k.unsqueeze(0) * torch.exp(-KL), axis=1, dtype = torch.float32)
        for k in range(args.num_clusters):
            self.delta.data[:, k] = pi_k[k] * torch.exp(-KL[:, k]) / denominator + det

    def update_others(self, mu_phi, log_cov_phi, delta, P):
        N_k = torch.sum(delta, axis=0, dtype = torch.float32)

        self.pi_k.data = N_k / args.num_points

        for k in range(args.num_clusters):
            delta_k = delta[:, k]  # N * 1
            self.mu_k.data[k] = torch.sum(mu_phi * delta_k.unsqueeze(1), axis=0, dtype = torch.float32) / N_k[k]
            mu_k = self.mu_k

            diff = P * torch.exp(log_cov_phi) + torch.sum((mu_k[k].unsqueeze(0) - mu_phi) ** 2, axis=1, dtype = torch.float32).unsqueeze(1)
            cov_k = torch.sum(delta_k.unsqueeze(1) * diff, axis=0, dtype = torch.float32) / (P * N_k[k])
            self.log_cov_k.data[k] = torch.log(cov_k + 1e-8)


# #######################################  Test #########################################
# # Assuming device is defined as "cpu" or "cuda"
# device = torch.device('cpu')
#
# # Mock the args you would get from args.py
# class Args:
#     num_points = 100  # Number of nodes
#     num_clusters = 3
#     input_dim = 10
#     hidden_dim = 64
#     emb_dim = 16  # Embedding dimension, adjust if needed
#     output_dims = [emb_dim] * 3  # Assuming three layers with the same output dimension
#     num_layers = 3
#     feature_dim = 10
#     pretrain_lr = 0.001
#     pretrain_epochs = 10
#
# args = Args()
#
# # Initialize mock adjacency matrices for testing
# adj_matrices = [torch.eye(args.num_points) for _ in range(args.num_layers)]
#
# # Create a MultiLPM instance
# model = MultiLPM(adj_matrices)
# model.to(device)
#
# # Define a mock dataset for pretraining
# X = torch.rand(args.num_points, args.input_dim).to(device)
# adj_labels = [torch.randint(0, 2, (args.num_points, args.num_points)).float().to(device) for _ in range(args.num_layers)]
# Y_list = [torch.rand(args.num_points, args.feature_dim).to(device) for _ in range(args.num_layers)]
#
# # Test the pretrain function
# model.pretrain(X, adj_labels, Y_list)
#
# # Test the update_delta function with dummy data
# mu_phi = torch.randn(args.num_points, args.emb_dim)
# log_cov_phi = torch.randn(args.num_points, 1)
# pi_k = torch.randn(args.num_clusters)
# mu_k = torch.randn(args.num_clusters, args.emb_dim)
# log_cov_k = torch.randn(args.num_clusters, 1)
# P = args.emb_dim
# model.update_delta(mu_phi, log_cov_phi, pi_k, mu_k, log_cov_k, P)
#
# # Test the update_others function with dummy data
# delta = torch.randn(args.num_points, args.num_clusters)
# model.update_others(mu_phi, log_cov_phi, delta, P)