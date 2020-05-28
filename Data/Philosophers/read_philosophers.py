import numpy as np
from scipy.sparse import csr_matrix
import os


def read_philosophers(home_dir='./', threshold=5, return_feat_names=False):
    """
    :param home_dir: Path to the directory containing the dataset
    :param threshold: Threshold to be used for eliminating features
    :param return_feat_names: Whether to return names of different features or not
    :return: adj_mat, node_features, ground_truth
        adj_mat: N x N binary adjacency matrix
        node_features: N x M binary node features
        names: Names of all philosophers
        feat_names: Names of all features
    """
    # Read the data
    adj_mat = csr_matrix(np.load(os.path.join(home_dir, 'adj_mat.npy')))
    node_features = csr_matrix(np.load(os.path.join(home_dir, 'node_features.npy')))
    names = [0] * node_features.shape[0]
    for line in open(os.path.join(home_dir, 'node_names.txt')):
        if len(line.strip()) > 0:
            name, idx = line.strip().split('\t')
            names[int(idx)] = name

    feat_names = [0] * node_features.shape[1]
    if return_feat_names:
        # Read the feature names
        with open(os.path.join(home_dir, 'feat_names.txt')) as f:
            for line in f:
                idx, name = line.strip().split('\t')
                feat_names[int(idx)] = name

    # Remove extra nodes
    idx = np.nonzero(adj_mat.sum(axis=1))
    adj_mat = adj_mat[np.ix_(idx[0], idx[0])]
    node_features = node_features[idx[0], :]
    names = [names[i] for i in idx[0].tolist()]

    # Remove nodes that everyone connects to
    idx = np.nonzero(adj_mat.sum(axis=0) <= adj_mat.shape[0] * 0.7)
    adj_mat = adj_mat[np.ix_(idx[1], idx[1])]
    node_features = node_features[idx[1], :]
    names = [names[i] for i in idx[1].tolist()]

    # Remove nodes that connect to everyone
    idx = np.nonzero(adj_mat.sum(axis=1) <= adj_mat.shape[0] * 0.7)
    adj_mat = adj_mat[np.ix_(idx[0], idx[0])]
    node_features = node_features[idx[0], :]
    names = [names[i] for i in idx[0].tolist()]

    # Remove extra features
    idx = np.nonzero(node_features.sum(axis=0) > threshold)
    node_features = node_features[:, idx[1]]
    if return_feat_names:
        feat_names = [feat_names[i] for i in idx[1].tolist()]

    idx = np.nonzero(node_features.sum(axis=0) <= node_features.shape[0] * 0.5)
    node_features = node_features[:, idx[1]]
    if return_feat_names:
        feat_names = [feat_names[i] for i in idx[1].tolist()]

    # Remove nodes that do not have any features left
    idx = np.nonzero(node_features.sum(axis=1) > 0)
    adj_mat = adj_mat[np.ix_(idx[0], idx[0])]
    node_features = node_features[idx[0], :]
    names = [names[i] for i in idx[0].tolist()]

    # Remove self edges
    adj_mat = adj_mat.todense()
    for i in range(adj_mat.shape[0]):
        adj_mat[i, i] = 0
    adj_mat = csr_matrix(adj_mat)

    if return_feat_names:
        return adj_mat, node_features, names, feat_names

    return adj_mat, node_features, names


if __name__ == '__main__':
    adj_mat, node_features, names = read_philosophers()
    print(adj_mat.shape)
    print(node_features.shape)
    print(names)
