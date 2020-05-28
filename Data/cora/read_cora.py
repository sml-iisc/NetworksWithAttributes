import numpy as np
import os
from scipy.sparse import csr_matrix


def read_cora(data_path='./'):
    """
    :param data_path: Path to the root folder that contains the Cora dataset
    returns: adj_mat, node_features, ground_truth
        adj_mat: (N x N) binary sparse matrix
        node_features: (N x M) binary sparse matrix
        ground_truth: List of labels (N elements)
    """
    feat_file = os.path.join(data_path, 'cora.content')
    net_file = os.path.join(data_path, 'cora.cites')

    # Get the node IDs, features and ground truth
    node_to_idx = dict()
    idx_to_node = dict()
    ground_truth = []
    rows = []
    cols = []
    data = []
    idx = 0
    label_dict = {'Case_Based':0, 'Genetic_Algorithms':1, 'Neural_Networks':2, 'Probabilistic_Methods':3,
                  'Reinforcement_Learning':4, 'Rule_Learning':5, 'Theory':6}

    with open(feat_file, 'r') as f:
        for line in f:
            tokens = line.strip().split('\t')

            nodeid = int(tokens[0])
            node_to_idx[nodeid] = idx
            idx_to_node[idx] = nodeid

            ground_truth += [label_dict[tokens[-1]]]

            feat_no = 0
            for val in tokens[1:-1]:
                if int(val) > 0:
                    rows += [idx]
                    cols += [feat_no]
                    data += [int(val)]
                feat_no += 1

            idx += 1

        num_nodes = idx
        node_features = csr_matrix((data, (rows, cols)), shape=(num_nodes, feat_no))

    rows = []
    cols = []
    data = []
    with open(net_file, 'r') as f:
        for line in f:
            id1, id2 = line.strip().split('\t')
            id1 = node_to_idx[int(id1)]
            id2 = node_to_idx[int(id2)]
            rows += [id1]
            cols += [id2]
            data += [1]

        adj_mat = csr_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes))

    return adj_mat, node_features, ground_truth



if __name__ == '__main__':
    A, Y, Z = read_cora()
    print(A.shape)
    print(Y.shape)
    print(len(Z))
    print(A)
