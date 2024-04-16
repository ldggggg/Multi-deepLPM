### CONFIGS ###
dataset = 'simu'
delta = 0.6
model = 'MultiLPM'

input_dim = 1000  # dim of X
hidden_dim = 128
output_dims = [16, 32, 64]
emb_dim = 16
num_layers = 3
feature_dim = 500  # dim of Y
num_cluster = 3

pretrain_epochs = 20
pretrain_lr = 1e-3