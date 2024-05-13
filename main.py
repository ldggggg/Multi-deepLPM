import csv
import numpy as np
# from testA import create_simuA
from testB import create_simuB
# from testC import create_simuC
# from testACM import create_ACM

from train import run_training # Adjust this import based on your actual module and function
import model
import args

zeta_values = [0.6]  # [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.95]
seeds = [9]  # np.arange(0, 25).tolist()  # [42, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]  # [42, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
num_runs = 1  # 5
results = []

for zeta in zeta_values:
    for seed in seeds:
        ari_values = []
        loss_values = []
        for run in range(num_runs):
            # adj_matrices, labels = create_ACM()
            print(args.num_points, args.num_clusters, zeta, seed)
            adj_matrices, labels = create_simuB(N=args.num_points, K=args.num_clusters, zeta=zeta, seed=seed)  # Adjust N and K as per your setup  # zeta=zeta, (for simuA and B)
            max_ari, min_loss = run_training(model, adj_matrices, labels)
            ari_values.append(max_ari)
            loss_values.append(min_loss)

        # average_ari = np.mean(ari_values)
        max_ari = np.max(ari_values)
        results.append({'zeta': zeta, 'seed': seed, 'max_ari': max_ari, 'all_aris': ari_values, 'all_loss': loss_values})

# Writing results to a CSV file
with open('experiment_results.csv', 'w', newline='') as file:
    fieldnames = ['zeta', 'seed', 'max_ari', 'all_aris', 'all_loss']
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    for result in results:
        writer.writerow({'zeta': result['zeta'], 'seed': result['seed'], 'max_ari': result['max_ari'],
                         'all_aris': "|".join(map(str, result['all_aris'])), 'all_loss': "|".join(map(str, result['all_loss']))})

print("Results saved to experiment_results.csv")