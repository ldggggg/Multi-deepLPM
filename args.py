### CONFIGS ###
dataset = 'simuA'
delta = 0.6
model = 'MultiLPM'

num_clusters = 3  # K
num_points = 100  # N
feature_dim = 100  # dim of Y
input_dim = 100  # dim of X
hidden_dim = 32
output_dims = [16, 16, 16]
emb_dim = 16  # P
num_layers = 3  # L

pretrain_epochs = 100
pretrain_lr = 2e-3

train_epochs = 200
train_lr = 5e-3