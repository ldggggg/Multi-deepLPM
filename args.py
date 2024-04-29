### CONFIGS ###
dataset = 'simuB'
model = 'MultiLPM'

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