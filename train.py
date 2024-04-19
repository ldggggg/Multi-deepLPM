import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import scipy.sparse as sp
import numpy as np
import os
import time
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import adjusted_rand_score
import matplotlib.pyplot as plt

from input_data import load_data
from preprocessing import *
import model
import args
from simu import generate_matrices
from evaluation import eva
from testA import create_simuA

# Train on CPU or GPU
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
device = torch.device('cuda:0')  # GPU
# print(torch.cuda.is_available())
# print(device)
# os.environ['CUDA_VISIBLE_DEVICES'] = ""  # CPU


##################### Load data ########################
if args.dataset == 'simu':
    adj_matrices, cov_matrices, feat_matrix = generate_matrices()
    print('Generated adjacency and covariate matrices.')

elif args.dataset == 'simuA':
    adj_matrices, labels = create_simuA(args.num_points, args.num_clusters)
    feat_matrix = np.eye(args.num_points)
    # cov_matrices = [np.zeros(args.num_points), np.zeros(args.num_points), np.zeros(args.num_points)]
    cov_matrices = [np.zeros(args.num_points), np.zeros(args.num_points)]

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
    print(adj.shape)
    print(adj_csr.shape)

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


def get_acc(adj_recs, adj_labels):
    acc = 0.0
    for adj_rec, adj_label in zip(adj_recs, adj_labels):
        labels_all = adj_label.to_dense().view(-1).long()
        preds_all = (adj_rec > 0.5).view(-1).long()
        accuracy = (preds_all == labels_all).sum().float() / labels_all.size(0)
        acc += accuracy
        acc /= args.num_layers
    return acc

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
store_ari = []

def Multi_ELBO_Loss(delta, pi_k, mu_k, log_cov_k, mu_phi, log_cov_phi, A_pred_list, adj_labels, P):
    # Multi-layer graph reconstruction loss
    Loss1 = 0.0
    for A_pred, adj_label in zip(A_pred_list, adj_labels):
        # Convert adj_label to dense format if necessary
        if adj_label.is_sparse:
            adj_label_dense = adj_label.to_dense()
        else:
            adj_label_dense = adj_label

        # Graph reconstruction loss for this layer
        OO = adj_label_dense * (torch.log((A_pred / (1. - A_pred)) + 1e-16)) + torch.log((1. - A_pred) + 1e-16)
        OO.fill_diagonal_(0)
        OO = OO.to(device)
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
                  + P*torch.exp(log_cov_phi)/torch.exp(log_cov_K) \
                  + torch.norm(mu_K-mu_phi,dim=1,keepdim=True)**2/torch.exp(log_cov_K)
        KL[:, k] = 0.5*temp.squeeze()

    Loss2 = torch.sum(delta * KL)

    Loss3 = torch.sum(delta * (torch.log(pi_k.unsqueeze(0)) - torch.log(delta)))

    Loss = Loss1 + Loss2 - Loss3

    return Loss, Loss1, Loss2, -Loss3


##################### Visualisation of learned embeddings by PCA ####################
from sklearn.decomposition import PCA
def visu():
    if args.dataset == 'eveques':
        labelC = []
        delta = model.delta.cpu().data.numpy()
        labels = np.argmax(delta, axis=1)
        for idx in range(len(labels)):
            if labels[idx] == 0:
                labelC.append('lightblue')
            elif labels[idx] == 1:
                labelC.append('lightgreen')
            elif labels[idx] == 2:
                labelC.append('yellow')
            elif labels[idx] == 3:
                labelC.append('purple')
            elif labels[idx] == 4:
                labelC.append('blue')
            elif labels[idx] == 5:
                labelC.append('orange')
            elif labels[idx] == 6:
                labelC.append('cyan')
            elif labels[idx] == 7:
                labelC.append('red')
            else:
                labelC.append('green')

    pca = PCA(n_components=2, svd_solver='full')
    out = pca.fit_transform(model.encoder.mean.cpu().data.numpy())
    mean = pca.fit_transform(model.mu_k.cpu().data.numpy())
    f, ax = plt.subplots(1, figsize=(15, 10))
    ax.scatter(out[:, 0], out[:, 1], color=labelC)
    # ax.scatter(mean[:, 0], mean[:, 1], color='black', s=50)
    ax.set_title('PCA result of embeddings of DeepLPM (K='+str(args.num_clusters)+')', fontsize=18)
    plt.show()
    # f.savefig("C:/Users/Dingge/Desktop/results/emb_ARVGA.pdf", bbox_inches='tight')


#################################### train model ################################################
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

    if epoch > 1:    
        # calculate of ELBO loss
        optimizer.zero_grad()
        # update of GCN
        loss.backward()
        optimizer.step()

    train_acc = get_acc(A_pred_list, processed_adj_labels)

    if (epoch + 1) % 1 == 0:
        # eva(labels, torch.argmax(delta, axis=1).cpu().numpy(), epoch)
        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(loss.item()),
              "train_loss1=", "{:.5f}".format(loss1.item()), "train_loss2=", "{:.5f}".format(loss2.item()),
              "train_loss3=", "{:.5f}".format(loss3.item()),
              "train_acc=", "{:.5f}".format(train_acc),
              "time=", "{:.5f}".format(time.time() - t))
        # pred = []
        # for i in range(args.num_points):
        #     if i not in delete:
        #         pred.append(torch.argmax(delta, axis=1).cpu().numpy()[i])

    # if (epoch + 1) % 2000 == 0:
    #     visu()
        # f, ax = plt.subplots(1, figsize=(10, 15))
        # ax.scatter(model.encoder.mean.cpu().data.numpy()[:, 0], model.encoder.mean.cpu().data.numpy()[:, 1], color=labelC)
        # ax.scatter(model.mu_k.cpu().data.numpy()[:, 0], model.mu_k.cpu().data.numpy()[:, 1], color='black', s=50)
        # ax.set_title("Embeddings after training!")
        # plt.show()

    # if epoch== 3:
    #     temp_pathout = "C:/Users/Dingge/Documents/GitHub/deepLPM/temp/"
    #     import pathlib
    #     PathTemp = pathlib.Path(temp_pathout)
    #     if not PathTemp.exists():
    #         PathTemp.mkdir(parents=True,exist_ok=True)
    #
    #     torch.save(log_cov_k,f"{temp_pathout}/log_cov_k.pt")
    #     torch.save(log_cov_phi, f"{temp_pathout}/log_cov_phi.pt")
    #     torch.save(mu_k, f"{temp_pathout}/mu_k.pt")
    #     torch.save(mu_phi, f"{temp_pathout}/mu_phi.pt")


    store_loss[epoch] = torch.Tensor.item(loss)  # save train loss for visu
    store_loss1[epoch] = torch.Tensor.item(loss1)
    store_loss2[epoch] = torch.Tensor.item(loss2)
    store_loss3[epoch] = torch.Tensor.item(loss3)

    # store_ari[epoch] = torch.tensor(adjusted_rand_index(labels, torch.argmax(delta, axis=1)))  # save ARI
    if args.dataset == 'eveques':
        print('Unsupervised data without true labels (no ARI) !')
    else:
        store_ari.append(adjusted_rand_score(labels, torch.argmax(delta, axis=1).cpu().numpy()))

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

# print('Min loss:', torch.min(store_loss), 'K='+str(args.num_clusters), 'P='+str(args.emb_dim))

# import json
# # Instead of the previous print statement
# print(json.dumps({
#     "min_loss": torch.min(store_loss).item(),
#     "num_clusters": args.num_clusters,
#     "use_nodes": args.use_nodes,
#     "use_edges": args.use_edges
# }))
#
# if args.dataset != 'eveques':
#     print('Max ARI:', max(store_ari))
# print('alpha, beta:', model.alpha, model.beta)

# plot ARI
if args.dataset != 'eveques':
    f, ax = plt.subplots(1, figsize=(15, 10))
    ax.plot(store_ari, color='blue')
    ax.set_title("ARI")
    plt.show()


# ARI with kmeans
kmeans = KMeans(n_clusters=args.num_clusters).fit(model.encoder.aggregated_mean.cpu().data.numpy())
labelk = kmeans.labels_
print("ARI_embedding:", adjusted_rand_score(labels, labelk))

########################### save data for visualisation in R ##################################
# import csv
# file = open('cora_data_A_k='+str(args.num_clusters)+'_p=16_'+str(args.use_nodes)+str(args.use_edges)+'.csv', "w")
# writer = csv.writer(file)
# mean = model.encoder.mean.cpu().data.numpy()
# pred_labels = torch.argmax(delta, axis=1).cpu().numpy()
# for w in range(args.num_points):
#     writer.writerow([w, mean[w][0],mean[w][1],mean[w][2],mean[w][3],mean[w][4],mean[w][5],mean[w][6],mean[w][7],
#                      mean[w][8],mean[w][9],mean[w][10],mean[w][11],mean[w][12],mean[w][13],mean[w][14],mean[w][15], pred_labels[w]])  # mean[w][8],mean[w][9],mean[w][10],mean[w][11],mean[w][12],mean[w][13],mean[w][14],mean[w][15]
# file.close()
#
# from sklearn.decomposition import PCA
# pca = PCA(n_components=2, svd_solver='full')
# out = pca.fit_transform(mean)
# np.savetxt('cora_pos_A_k='+str(args.num_clusters)+'_p=16_'+str(args.use_nodes)+str(args.use_edges)+'.txt', out)
#
# np.savetxt('cora_cl_A_k='+str(args.num_clusters)+'_p=16_'+str(args.use_nodes)+str(args.use_edges)+'.txt', pred_labels)
#
# np.savetxt('cora_mu_k='+str(args.num_clusters)+'_p=16_'+str(args.use_nodes)+str(args.use_edges)+'.txt', model.mu_k.cpu().data.numpy())