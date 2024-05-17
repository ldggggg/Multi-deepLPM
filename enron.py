import pandas as pd
import numpy as np

def to_adjacency_matrix(connection_matrix):
    # Convert the connection matrix to a boolean array, where True represents a connection
    adjacency_matrix = np.array(connection_matrix) != 0

    # Convert boolean values to integers (True to 1, False to 0)
    return adjacency_matrix.astype(int)


def load_enron():
    print('loading enron data...............................')
    # Load the data frame
    P = pd.read_csv("D:/Enron-TO-CC-SepDec2001.Rdata/P_process_data.csv")

    # Load the adjacency matrices
    adj_matrix_to = pd.read_csv("D:/Enron-TO-CC-SepDec2001.Rdata/adj_matrix_to.csv", header=None)
    adj_matrix_cc = pd.read_csv("D:/Enron-TO-CC-SepDec2001.Rdata/adj_matrix_cc.csv", header=None)

    # Adjust the row indices to start from 1 instead of 0
    adj_matrix_to.index -= 1
    adj_matrix_cc.index -= 1

    adj_to = adj_matrix_to.iloc[1:]
    adj_cc = adj_matrix_cc.iloc[1:]

    # Convert the data to integers
    adj_to = adj_to.apply(pd.to_numeric)
    adj_cc = adj_cc.apply(pd.to_numeric)

    adj_to = adj_to.to_numpy()
    adj_cc = adj_cc.to_numpy()

    # Convert to adjacency matrix
    adj_matrix_1 = to_adjacency_matrix(adj_cc)
    adj_matrix_2 = to_adjacency_matrix(adj_to)

    adj_matrices = [adj_matrix_1, adj_matrix_2]

    status = P['status'].tolist()

    return adj_matrices, status

# # Function to create graph from adjacency matrix and color nodes
# import networkx as nx
# import matplotlib.pyplot as plt

# def create_graph(adj_matrix, node_info):
#     G = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph())
#
#     # Mapping of index to node id and adding attributes like 'status'
#     id_map = node_info['eid'].to_list()  # Adjust column name if different
#     attr = node_info.set_index('eid').to_dict('index')
#     nx.set_node_attributes(G, {idx: attr[id_map[idx]] for idx in G.nodes()})
#
#     # Extracting status for color mapping
#     status_color = {status: idx for idx, status in enumerate(set(node_info['status']))}
#     colors = [status_color[data['status']] for _, data in G.nodes(data=True)]
#
#     return G, colors
#
#
# # Create graph for 'TO' network
# G_to, colors_to = create_graph(adj_to, P)
# # Create graph for 'CC' network
# G_cc, colors_cc = create_graph(adj_cc, P)
#
#
# # Function to draw the graph
# def draw_graph(G, colors, title):
#     plt.figure(figsize=(12, 8))
#     pos = nx.spring_layout(G)  # For better node distribution
#     nx.draw(G, pos, node_color=colors, with_labels=True, cmap=plt.cm.Set1, node_size=50)
#     plt.title(title)
#     plt.show()
#
#
# # Draw the graphs
# draw_graph(G_to, colors_to, 'TO Network')
# draw_graph(G_cc, colors_cc, 'CC Network')
