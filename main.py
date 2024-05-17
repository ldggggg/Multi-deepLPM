import csv
import numpy as np
# from testA import create_simuA
# from testB import create_simuB
from testC import create_simuC
# from testACM import create_ACM

from train import run_training # Adjust this import based on your actual module and function
import model
import args
import time


zeta_values = [0.9]  # [0.95, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]  # [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
seeds = [1]  # np.arange(0, 25).tolist()  # [42, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
num_runs = 1  # 5
results = []


for zeta in zeta_values:
    for seed in seeds:
        ari_values = []
        loss_values = []
        for run in range(num_runs):
            # adj_matrices, labels = create_ACM()
            print(args.num_points, args.num_clusters, zeta, seed)
            # adj_matrices, labels = create_simuA(N=args.num_points, K=args.num_clusters, zeta=zeta, seed=seed)  # Adjust N and K as per your setup  # zeta=zeta, (for simuA and B)
            adj_matrices, labels = create_simuC(N=args.num_points, K=args.num_clusters, seed=seed)
            begin = time.time()
            max_ari, min_loss = run_training(model, adj_matrices, labels)
            end = time.time()
            ari_values.append(max_ari)
            loss_values.append(min_loss.cpu().data.numpy())

        # average_ari = np.mean(ari_values)
        min_loss_index = np.argmin(loss_values)
        max_ari = np.max(ari_values)
        ari_min_loss = np.array(ari_values)[min_loss_index]
        results.append({'zeta': zeta, 'seed': seed, 'max_ari': max_ari, 'ari_min_loss':ari_min_loss, 'all_aris': ari_values, 'all_loss': loss_values})

print('computation time:', end-begin)

# # Writing results to a CSV file
# with open('experiment_results_A.csv', 'w', newline='') as file:
#     fieldnames = ['zeta', 'seed', 'max_ari', 'ari_min_loss', 'all_aris', 'all_loss']
#     writer = csv.DictWriter(file, fieldnames=fieldnames)
#     writer.writeheader()
#     for result in results:
#         writer.writerow({'zeta': result['zeta'], 'seed': result['seed'], 'max_ari': result['max_ari'], 'ari_min_loss':result['ari_min_loss'],
#                          'all_aris': "|".join(map(str, result['all_aris'])), 'all_loss': "|".join(map(str, result['all_loss']))})

print("Results saved to experiment_results_A.csv")