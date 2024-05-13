### CONFIGS ###
dataset = 'simuB'
model = 'MultiLPM'

if dataset == 'ACM':
    num_clusters = 3  # K
    num_points = 3025  # N
    feature_dim = 3025  # dim of Y
    input_dim = 1870  # dim of X
    hidden_dim = 64
    output_dims = [16, 16]
    emb_dim = 16  # P
    num_layers = 2  # L

    pretrain_epochs = 20
    pretrain_lr = 1e-2

    train_epochs = 600
    train_lr = 5e-3


elif dataset == 'simuA' or 'simuB' or 'simuC':
    num_clusters = 3  # K
    num_points = 600  # N
    feature_dim = 600  # dim of Y
    input_dim = 600  # dim of X
    hidden_dim = 64
    output_dims = [16, 16, 16]  # [16, 16, 16]
    emb_dim = 16  # P
    num_layers = 3  # 3  # L

    pretrain_epochs = 50  # 50 for simuA
    pretrain_lr = 1e-2  # 1e-2 for simuA

    train_epochs = 600  # 1000
    train_lr = 5e-3