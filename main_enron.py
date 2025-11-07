### to load real-world data
from enron import load_enron
import model
import torch
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
import os
import time
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import adjusted_rand_score
import matplotlib.pyplot as plt
from preprocessing import *
import args
import pdb
import scipy.io as sio
### to create simu data
from testA import create_simuA
# from testB import create_simuB
# from testC import create_simuC

# Train on CPU or GPU
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
device = torch.device('cuda:0')  # GPU
# print(torch.cuda.is_available())
# print(device)
# os.environ['CUDA_VISIBLE_DEVICES'] = ""  # CPU


def get_acc(adj_recs, adj_labels):
    acc = 0.0
    for adj_rec, adj_label in zip(adj_recs, adj_labels):
        labels_all = adj_label.to_dense().view(-1).long()
        preds_all = (adj_rec > 0.5).view(-1).long()
        accuracy = (preds_all == labels_all).sum().float() / labels_all.size(0)
        acc += accuracy
        acc /= args.num_layers
    return acc


def Multi_ELBO_Loss(delta, pi_k, mu_k, log_cov_k, mu_phi, log_cov_phi, A_pred_list, adj_labels, P):
    # Multi-layer graph reconstruction loss
    det = 1e-8
    Loss1 = 0.0
    for A_pred, adj_label in zip(A_pred_list, adj_labels):
        # Convert adj_label to dense format if necessary
        if adj_label.is_sparse:
            adj_label_dense = adj_label.to_dense()
        else:
            adj_label_dense = adj_label

        # To prevent loss1=NaN
        if torch.any(A_pred <= 0) or torch.any(A_pred >= 1):
            print("A_pred contains values out of expected range!")

        # Graph reconstruction loss for this layer
        OO = adj_label_dense * torch.log((A_pred / (1. - A_pred + det)) + det) + torch.log(
            (1. - A_pred + det))
        # OO = adj_label_dense * (torch.log((A_pred / (1. - A_pred)) + 1e-8)) + torch.log((1. - A_pred) + 1e-8)
        OO.fill_diagonal_(0.0)
        OO = OO.to(device)

        if torch.isnan(OO).any():
            print("NaN values detected in OO matrix")

        loss_layer = -torch.sum(OO)
        # print(loss_layer)

        # Sum up the loss from each layer
        Loss1 += loss_layer

        # Loss1 += F.binary_cross_entropy(A_pred.view(-1), adj_label_dense.view(-1))
    Loss1 /= len(A_pred_list)

    # KL divergence
    KL = torch.zeros((args.num_points, args.num_clusters))  # N * K
    KL = KL.to(device)
    for k in range(args.num_clusters):
        log_cov_K = torch.ones_like(log_cov_phi) * log_cov_k[k]
        mu_K = torch.ones((args.num_points, mu_k.shape[1])).to(device) * mu_k[k]
        temp = P*(log_cov_K-log_cov_phi-1) \
                  + P*torch.exp(log_cov_phi + det)/torch.exp(log_cov_K + det) \
                  + torch.norm(mu_K-mu_phi,dim=1,keepdim=True)**2/torch.exp(log_cov_K + det)
        KL[:, k] = 0.5*temp.squeeze() + det
        if torch.isnan(KL).any():
            print("NaN values detected in KL matrix in training")

    Loss2 = torch.sum(delta * KL)

    Loss3 = torch.sum(delta * (torch.log(pi_k.unsqueeze(0) + det) - torch.log(delta + det)))

    Loss = Loss1 + Loss2 - Loss3

    return Loss, Loss1, Loss2, -Loss3


##################### Load data ########################
### to load real-world data
if args.dataset == 'enron':
    adj_matrices, labels = load_enron()
    feat_matrix = np.eye(args.num_points)
    cov_matrices = [np.zeros(args.num_points), np.zeros(args.num_points)]
elif args.dataset == "ACM":
    data = sio.loadmat("D:/Work/Benchmark/WWW2020-O2MAC-master/data/ACM3025.mat")
    feat_matrix = data['feature'].astype(float)
    # print("feat_matrix:", feat_matrix.shape)
    cov_matrices = [np.zeros(args.num_points), np.zeros(args.num_points)]
### to create simu data
elif args.dataset == 'simuA' or 'simuB' or 'simuC':
    adj_matrices, labels = create_simuA(args.num_points, args.num_clusters, 0.95, 42)
    feat_matrix = np.eye(args.num_points)
    cov_matrices = [np.zeros(args.num_points), np.zeros(args.num_points), np.zeros(args.num_points)]


# Initialize lists to store the processed matrices
processed_adj_norms = []
processed_adj_labels = []
processed_edges = []

# Process each pair of adjacency and covariate matrix and feature matrix
features = sp.csr_matrix(feat_matrix)
features = sparse_to_tuple(features.tocoo())
features = torch.sparse.FloatTensor(torch.LongTensor(features[0].astype(float).T),
                            torch.FloatTensor(features[1]),
                            torch.Size(features[2]))
features = features.to(device)

for adj, cov in zip(adj_matrices, cov_matrices):
    # Convert adjacency matrix to scipy CSR format
    adj_csr = sp.csr_matrix(adj)

    # Normalize the adjacency matrix
    adj_norm = preprocess_graph(adj_csr)
    adj_norm = torch.sparse.FloatTensor(torch.LongTensor(adj_norm[0].astype(float).T),
                                        torch.FloatTensor(adj_norm[1]),
                                        torch.Size(adj_norm[2]))
    adj_norm = adj_norm.to(device)
    processed_adj_norms.append(adj_norm)

    # Create adjacency label for loss calculation
    adj_label = adj_csr + sp.eye(adj_csr.shape[0])
    adj_label = sparse_to_tuple(adj_label)
    adj_label = torch.sparse.FloatTensor(torch.LongTensor(adj_label[0].astype(float).T),
                                         torch.FloatTensor(adj_label[1]),
                                         torch.Size(adj_label[2]))
    adj_label = adj_label.to(device)
    processed_adj_labels.append(adj_label)

    # Convert feature matrix (if cov is not None) and adjacency matrix
    if cov is not None:
        edges = sp.csr_matrix(cov)  # Using covariates as features
        edges = sparse_to_tuple(edges.tocoo())
        edges = torch.sparse.FloatTensor(torch.LongTensor(edges[0].astype(float).T),
                                            torch.FloatTensor(edges[1]),
                                            torch.Size(edges[2]))
        edges = edges.to(device)
        processed_edges.append(edges)


################################ Model ##################################
# init model and optimizer
model = getattr(model, args.model)(processed_adj_norms)
model.to(device)  # to GPU
model.pretrain(features, processed_adj_labels, processed_edges, labels)  # pretraining
optimizer = Adam(model.parameters(), lr=args.train_lr)  # , weight_decay=0.01

# store loss
store_loss = torch.zeros(args.train_epochs).to(device)
store_loss1 = torch.zeros(args.train_epochs).to(device)
store_loss2 = torch.zeros(args.train_epochs).to(device)
store_loss3 = torch.zeros(args.train_epochs).to(device)
store_ari = []  # for simu data who have the true labels

#################################### train model #####################################
begin = time.time()
for epoch in range(args.train_epochs):
    t = time.time()
    mu_phi, log_cov_phi, z = model.encoder(features)

    A_pred_list = model.decoder(z, processed_edges)

    if epoch < 1 or (epoch + 1) % 1 == 0:
        # update pi_k, mu_k and log_cov_k
        delta = model.delta
        model.update_others(mu_phi.detach().clone(),
                            log_cov_phi.detach().clone(),
                            delta, args.emb_dim)

        # update delta
        pi_k = model.pi_k
        log_cov_k = model.log_cov_k
        mu_k = model.mu_k
        model.update_delta(mu_phi.detach().clone(),
                           log_cov_phi.detach().clone(),
                           pi_k, mu_k, log_cov_k, args.emb_dim)

    pi_k = model.pi_k                    # pi_k should be a copy of model.pi_k
    log_cov_k = model.log_cov_k
    mu_k = model.mu_k
    delta = model.delta
    loss, loss1, loss2, loss3 = Multi_ELBO_Loss(delta, pi_k, mu_k, log_cov_k, mu_phi, log_cov_phi,
                                                A_pred_list, processed_adj_labels, args.emb_dim)

    if torch.isnan(loss).any() or torch.isnan(loss1).any() or torch.isnan(loss2).any() or torch.isnan(loss3).any():
        print("NaN detected!")
        pdb.set_trace()  # Activate Python Debugger

    if epoch > 1 or (epoch + 1) % 1 == 0:
        # calculate of ELBO loss
        optimizer.zero_grad()
        # update of GCN
        loss.backward()
        # clip of gradient
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    train_acc = get_acc(A_pred_list, processed_adj_labels)

    if (epoch + 1) % 1 == 0:
        # eva(labels, torch.argmax(delta, axis=1).cpu().numpy(), epoch)
        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(loss.item()),
              "train_loss1=", "{:.5f}".format(loss1.item()), "train_loss2=", "{:.5f}".format(loss2.item()),
              "train_loss3=", "{:.5f}".format(loss3.item()),
              "train_acc=", "{:.5f}".format(train_acc),
              "time=", "{:.5f}".format(time.time() - t))

    # save train loss for visu
    store_loss[epoch] = torch.Tensor.item(loss)
    store_loss1[epoch] = torch.Tensor.item(loss1)
    store_loss2[epoch] = torch.Tensor.item(loss2)
    store_loss3[epoch] = torch.Tensor.item(loss3)
    # save ARI
    store_ari.append(adjusted_rand_score(labels, torch.argmax(delta, axis=1).cpu().numpy()))  # for simu data who have the true labels

    if torch.isnan(loss).any() or torch.isnan(loss1).any() or torch.isnan(loss2).any() or torch.isnan(loss3).any():
        break  # Optionally stop training if a NaN is found

end = time.time()
print('training time ......................:', end-begin)

################################# plots to show results ###################################
# plot train loss
f, ax = plt.subplots(1, figsize=(15, 10))
plt.subplot(231)
plt.plot(store_loss1.cpu().data.numpy(), color='red')
plt.title("Reconstruction loss1")

plt.subplot(232)
plt.plot(store_loss2.cpu().data.numpy(), color='red')
plt.title("KL loss2")

plt.subplot(233)
plt.plot(store_loss3.cpu().data.numpy(), color='red')
plt.title("Cluster loss3")

plt.subplot(212)
plt.plot(store_loss.cpu().data.numpy(), color='red')
plt.title("Training loss in total")

plt.show()

# plot ARI
# for simu data who have the true labels
if args.dataset != 'enron':
    f, ax = plt.subplots(1, figsize=(15, 10))
    ax.plot(store_ari, color='blue')
    ax.set_title("ARI")
    plt.show()
print("ARI_delta:", max(store_ari))

# ARI with kmeans
kmeans = KMeans(n_clusters=args.num_clusters).fit(model.encoder.aggregated_mean.cpu().data.numpy())
labelk = kmeans.labels_
print("ARI_kmeans_embedding:", adjusted_rand_score(labels, labelk))


# num_runs = 1
# loss_values = []
#
# for run in range(num_runs):
#     print(args.num_points, args.num_clusters)
#     adj_matrices, labels = load_enron()
#     embs, min_loss = run_training(model, adj_matrices, labels)
#     loss_values.append(min_loss.cpu().data.numpy())


###################### plot 1 ################################
### used for simu data who has true labels
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# Alternatively, using PCA for dimensionality reduction
# pca = PCA(n_components=2)
# embeddings_2d = pca.fit_transform(embeddings)
embeddings = z.cpu().data.numpy()
statuses = np.array(labels)

# Apply t-SNE
tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=42)
embeddings_2d = tsne.fit_transform(embeddings)

np.savetxt('pos' + str(args.num_clusters) + '.csv', embeddings_2d, delimiter=',')

# Plotting
plt.figure(figsize=(10, 8))
for status in np.unique(statuses):
    idx = statuses == status
    plt.scatter(embeddings_2d[idx, 0], embeddings_2d[idx, 1], label=status, alpha=0.8, s=50)

plt.title('Visualizing Node Embeddings with Node Status')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.legend(title='Status')
plt.grid(True)
plt.show()


###################### plot 2 ################################
### used for real-world datasets
# Determine the most probable cluster for each node
cluster_labels = np.argmax(model.delta.cpu().data.numpy(), axis=1)

# Create a color map based on number of clusters
colors = plt.cm.get_cmap('viridis', args.num_clusters)

# Plotting
plt.figure(figsize=(10, 8))
scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=cluster_labels, cmap=colors, edgecolor='k', alpha=0.7)
plt.colorbar(scatter, label='Cluster Label')
plt.title('Visualizing Node Embeddings with Cluster Labels')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.show()


###################### plot 3 ################################
### used for real-world datasets
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# Convert categorical status labels to numeric if necessary
encoder = LabelEncoder()
numeric_statuses = encoder.fit_transform(statuses)

# Compute the confusion matrix
cm = confusion_matrix(numeric_statuses, cluster_labels)

# Plot the confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(cm[:,0:5], annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix of Cluster Labels vs Node Status')
plt.xlabel('Predicted Labels')
plt.ylabel('True Statuses')
plt.show()