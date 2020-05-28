import numpy as np
from scipy.sparse import csr_matrix
import os


def el_reader(home_dir='./', bins=([[1, 1], [2, 2]],
                                   [[1, 1], [2, 2]], 
                                   [[1, 1], [2, 2], [3, 3]], 
                                   [[0, 5], [6, 10], [11, 15], [16, 20], [21, 25], [26, 30], [31, 35]],
                                   [[20, 30], [31, 40], [41, 50], [51, 60], [61, 70]],
                                   [[1, 1], [2, 2]],
                                   [[1, 1], [2, 2], [3, 3]]),
              net_type='friend', fetch36=False, ground_truth_col=1):
    """
        :param home_dir: Path to home directory containing the dataset files
        :param bins: List of bins to use for features. bins[i] is again a list for ith feature
        :param net_type: Type of network to fetch. (work, adv or friend)
        :param fetch36: If True, then subset of 36 nodes will be fetched
        :param ground_truth_col: Columns in attributes file that should be used as ground truth
        returns: adj_mat, node_attributes, ground_truth
            adj_mat: NxN binary adjacency matrix
            node_attributes: NxM binary node attributes. Note that ground truth is part of these attributes
            ground_truth: List of ground truth label for all nodes
    """
    subset = '36' if fetch36 else ''

    # Read the network
    with open(os.path.join(home_dir, 'EL' + net_type + subset + '.dat'), 'r') as net_file:
        adj_mat = csr_matrix(np.loadtxt(net_file))    
    
    # Read the attributes
    with open(os.path.join(home_dir, 'ELattr.dat'), 'r') as attr_file:
        attributes = np.loadtxt(attr_file)
        attributes = attributes[:, 1:]   # Remove seniority which is just serial number
        if fetch36:
            attributes = attributes[:36, :]
        
        # Get the number of features
        num_nodes = attributes.shape[0]
        num_features = 0
        for feat in bins:
            num_features += len(feat)
        
        # Binarize the attributes
        node_attributes = np.zeros((num_nodes, num_features))
        for node in range(num_nodes):
            idx = 0
            for j in range(7):
                for b in bins[j]:
                    if b[0] <= attributes[node, j] <= b[1]:
                        node_attributes[node, idx] = 1
                    idx += 1
        node_attributes = csr_matrix(node_attributes)               
        
        # Get the ground truth
        ground_truth = (attributes[:, ground_truth_col-1].astype(int) - 1).tolist()
    
    return adj_mat, node_attributes, ground_truth
        

if __name__ == '__main__':
    adj_mat, node_attributes, ground_truth = el_reader()

