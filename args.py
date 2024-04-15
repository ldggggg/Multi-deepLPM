### CONFIGS ###
# Data selected from:
# simulated data: 'simuA': LPCM; 'simuB': SBM; 'simuC': circle data;
# real data: 'eveques'; 'cora'.
dataset = 'simuA'
delta = 0.6
model = 'deepLPM'

input_dim = 1000  # dim of X
hidden_dim = 128
output_dims = [16, 32, 64]
emb_dim = 16
num_layers = 3
feature_dim = 500  # dim of Y
num_cluster = 3

pretrain_epochs = 20
pretrain_lr = 1e-3


if dataset == 'simuA':  # or 'simuA', 'simuC'
    use_nodes = False
    use_edges = False
    nb_of_edges = 0  # covariates dimension D

    num_points = 300  # number of nodes N
    input_dim = 300  # node features dimension (identity matrix in our model)
    hidden1_dim = 64  # hidden layer dimension
    hidden2_dim = 16
    num_clusters = 6

    num_epoch = 600  # 600  # training epochs
    learning_rate = 5e-3  # 2e-3 or 5e-3 (B: delta<0.4)
    pre_lr = 0.1  # 0.1 (B, C) or 0.2 (A, B: delta<0.4)
    pre_epoch = 100  # 100  # pretraining epochs: 100(B: delta<0.6) or 70 (B: delta<0.8, C) or 50

elif dataset == 'eveques':
    use_nodes = False
    use_edges = True
    nb_of_edges = 3  # D

    num_points = 1287  # N
    if use_nodes == True:
        input_dim = 10
    else:
        input_dim = 1287
    hidden1_dim = 64
    hidden2_dim = 4
    num_clusters = 2

    num_epoch = 800
    learning_rate = 2e-3
    pre_lr = 1e-3
    pre_epoch = 100

elif dataset == 'cora':
    use_nodes = True
    use_edges = False
    nb_of_edges = 49  # D

    num_points = 2708  # N
    input_dim = 1433  # dictionary size
    hidden1_dim = 64
    hidden2_dim = 4
    num_clusters = 2

    num_epoch = 2000
    learning_rate = 8e-3
    pre_lr = 0.01  # 0.005
    pre_epoch = 50