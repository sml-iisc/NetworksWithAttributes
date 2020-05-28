import os
import scipy.sparse


def read_sinanet_bins(path):
    """
    Reads the Sinanet network and puts each feature in bins
    :param path: Path to the folder containing dataset files ./Sinanet/
    :return: adj_list, adj_mat, node_features, ground_truth
        adj_list: List of lists where the i'th list contains the neighbors of ith node
        adj_mat: Sparse adjacency matrix for the network
        node_features: (num_nodes, num_features) Sparse feature matrix, each example is a row. Features are binary
        ground_truth: (num_nodes, 1) Ground truth community labels for all nodes. In case of multiple ground truth
                      communities for each node, only the last one is kept
    """
    # Prepare the file names
    comm_file = os.path.join(path, 'clusters.txt')
    features_file = os.path.join(path, 'content.txt')
    edges_file = os.path.join(path, 'edge.txt')

    # Get the number of nodes and number of features
    with open(features_file) as f:
        lines = list(f)
        num_nodes = len(lines)
        num_features = len(lines[0].strip().split(' '))

    # Get the edges
    adj_list = [[] for _ in range(num_nodes)]
    coords_row = []
    coords_col = []
    data = []
    with open(edges_file) as f:
        for line in f:
            n1, n2 = line.strip().split('\t')
            n1 = int(n1) - 1
            n2 = int(n2) - 1
            coords_row += [n1, n2]
            coords_col += [n2, n1]
            data += [1]
            data += [1]
            adj_list[n1] += [n2]
            adj_list[n2] += [n1]
    adj_mat = scipy.sparse.csr_matrix((data, (coords_row, coords_col)), shape=(num_nodes, num_nodes))

    # Get the circles
    ground_truth = [0] * num_nodes
    with open(comm_file) as f:
        circle = 0
        for line in f:
            nodes = line.strip().split('\t')
            for node in nodes:
                if int(node) == 0:
                    break
                ground_truth[int(node) - 1] = circle
            circle += 1

    # Get the features
    coords_row = []
    coords_col = []
    data = []
    count = 0
    with open(features_file) as f:
        for line in f:
            tokens = line.strip().split(' ')
            for feat in range(num_features):
                coords_row += [count]
                coords_col += [feat*10 + int(float(tokens[feat]) * 10)]
                data += [1]
            count += 1
    node_features = scipy.sparse.csr_matrix((data, (coords_row, coords_col)), shape=(num_nodes, num_features * 10))

    return adj_list, adj_mat, node_features, ground_truth


def read_sinanet(path, threshold=0.1):
    """
    Reads the Sinanet network
    :param path: Path to the folder containing dataset files ./Sinanet/
    :param threshold: Value of threshold for finding binary features
    :return: adj_list, adj_mat, node_features, ground_truth
        adj_list: List of lists where the i'th list contains the neighbors of ith node
        adj_mat: Sparse adjacency matrix for the network
        node_features: (num_nodes, num_features) Sparse feature matrix, each example is a row. Features are binary
        ground_truth: (num_nodes, 1) Ground truth community labels for all nodes. In case of multiple ground truth
                      communities for each node, only the last one is kept
    """
    # Prepare the file names
    comm_file = os.path.join(path, 'clusters.txt')
    features_file = os.path.join(path, 'content.txt')
    edges_file = os.path.join(path, 'edge.txt')

    # Get the number of nodes and number of features
    with open(features_file) as f:
        lines = list(f)
        num_nodes = len(lines)
        num_features = len(lines[0].strip().split(' '))

    # Get the edges
    adj_list = [[] for _ in range(num_nodes)]
    coords_row = []
    coords_col = []
    data = []
    with open(edges_file) as f:
        for line in f:
            n1, n2 = line.strip().split('\t')
            n1 = int(n1) - 1
            n2 = int(n2) - 1
            coords_row += [n1, n2]
            coords_col += [n2, n1]
            data += [1]
            data += [1]
            adj_list[n1] += [n2]
            adj_list[n2] += [n1]
    adj_mat = scipy.sparse.csr_matrix((data, (coords_row, coords_col)), shape=(num_nodes, num_nodes))

    # Get the circles
    ground_truth = [0] * num_nodes
    with open(comm_file) as f:
        circle = 0
        for line in f:
            nodes = line.strip().split('\t')
            for node in nodes:
                if int(node) == 0:
                    break
                ground_truth[int(node) - 1] = circle
            circle += 1

    # Get the features
    coords_row = []
    coords_col = []
    data = []
    count = 0
    with open(features_file) as f:
        for line in f:
            tokens = line.strip().split(' ')
            for feat in range(num_features):
                if float(tokens[feat]) >= threshold:
                    coords_row += [count]
                    coords_col += [feat]
                    data += [1]
            count += 1
    node_features = scipy.sparse.csr_matrix((data, (coords_row, coords_col)), shape=(num_nodes, num_features))

    return adj_list, adj_mat, node_features, ground_truth


if __name__ == '__main__':
    _, _, feats, gt = read_sinanet_bins('.')
#    import numpy as np
#    print((np.sum(feats, axis=0) >= 5).sum()
