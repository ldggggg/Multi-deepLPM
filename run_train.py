import subprocess
import re
import os
import json

def set_args(num_clusters_value, hidden2_dim_value):
    # Read the original args.py file
    with open('args.py', 'r') as f:
        content = f.readlines()

    # Modify the values of num_clusters and hidden2_dim
    with open('args.py', 'w') as f:
        for line in content:
            if 'num_clusters =' in line:
                f.write(f"num_clusters = {num_clusters_value}\n")
            elif 'hidden2_dim =' in line:
                f.write(f"hidden2_dim = {hidden2_dim_value}\n")
            else:
                f.write(line)


def extract_min_loss(output):
    try:
        data = json.loads(output)
        return data['min_loss']
    except json.JSONDecodeError:
        return None


# Parameters
num_clusters_values = [2, 3, 4, 5, 6]
hidden2_dims = [2, 4, 8, 16, 32]

results = {}

for num_clusters in num_clusters_values:
    for hidden2_dim in hidden2_dims:
        # Update the args.py file
        set_args(num_clusters, hidden2_dim)

        # Execute train.py five times and save the min loss
        losses = []
        for _ in range(5):
            result = subprocess.run(['python', 'train.py'], capture_output=True, text=True)
            output = result.stdout
            print(output)
            min_loss = extract_min_loss(output)
            if min_loss is not None:
                losses.append(min_loss)

        # Compute the average of the 5 minimum losses
        average_loss = sum(losses) / len(losses) if losses else None
        key = (num_clusters, hidden2_dim)
        results[key] = average_loss

# Print results
for (num_clusters, hidden2_dim), avg_loss in results.items():
    print(f"Num Clusters: {num_clusters}, Hidden2_dim: {hidden2_dim}, Average Min Loss: {avg_loss:.4f}")