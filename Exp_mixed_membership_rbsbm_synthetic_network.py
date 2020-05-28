import numpy as np
from model_mixed_membership_rbsbm import RBMMSBM
from torch.optim import Adam
from gen_network_mmrbsbm import gen_network_mmrbsbm_1
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import sys


"""
Arguments to be passed to the program

num_nodes : Number of nodes to be used for generating Synthetic Network
num_communities : Number of communities to be used for generating Synthetic Network
num_features : Number of features to be used for generating Synthetic Network
lr : Learning rate to be used by Adam optimizer
learning_rate : Learning rate to be used by the RBM
momentum_rate : momentum used to be by the RBM
chn_len : Length of the persistent chain used in CD algorithm pertaining to RBM
num_samp : Number of samples to be used to approximate the Expectation in CD algorithm pertaining to RBM
logfilepath : Path of the log file
"""

num_nodes = int(sys.argv[1])
num_communities = int(sys.argv[2])
num_features = int(sys.argv[3])
lr = float(sys.argv[4])
learning_rate = float(sys.argv[5])
momentum_rate = float(sys.argv[6])
chn_len = int(sys.argv[7])
num_samp = int(sys.argv[8])
logfilepath = sys.argv[9]

# Generate synthetic network
B = 0.5 * np.eye(num_communities) + 0.1 * (np.ones((num_communities, num_communities)) - np.eye(num_communities))
adj_mat, node_features, mm_vec, z_pq_1, z_pq_2, F = gen_network_mmrbsbm_1(num_features, num_communities, B, num_nodes)
adj_mat = adj_mat + adj_mat.T
for i in range(num_communities):
    z_pq_1[:,:,i] = z_pq_1[:,:,i] + z_pq_1[:,:,i].T
    z_pq_2[:,:,i] = z_pq_2[:,:,i] + z_pq_2[:,:,i].T

print('Number of nodes:', num_nodes)
print('Number of attributes:', num_features)
print('Number of edges:', adj_mat.sum())
print('P(edge):', adj_mat.sum() / (num_nodes * (num_nodes - 1)))
print('Number of communities:', num_communities)

num_trials = 1
Max_iters = 5

# Compute prior for B matrix
alphas = np.zeros((num_communities, num_communities))
betas = np.zeros((num_communities, num_communities)) + \
        3 * (np.ones((num_communities, num_communities)) - np.eye(num_communities))

dump_file = open(logfilepath + 'log_' + str(num_communities) + '_' + str(num_nodes) + '.txt', 'w')
# Start the training
plt.ion()
for trial in range(num_trials):
    # Set up the model
    model = RBMMSBM(num_nodes, num_communities, num_features, alphas, betas)
    optim = Adam(model.parameters(), lr=lr)

    # Train the model
    for i in range(Max_iters):
        # Run the E-Step
        elbo = model(torch.from_numpy(adj_mat).float(), torch.from_numpy(node_features).float())
        optim.zero_grad()
        elbo.backward()
        optim.step()

        # Run the M-step
        for _ in range(1):
            model.m_step(node_features, chain_length=chn_len, num_samples=num_samp, lr=learning_rate, momentum=momentum_rate, use_persistence=True)

        # Show the scores
        if i % 5 == 0:
            print('Trial:', trial + 1,
                  'Iter:', i + 1,
                  'ELBO:', elbo.item())
            dump_file.write('Trial:\t'+str(trial + 1)+'\tIter:\t'+str(i + 1)+'\tELBO:\t'+str(elbo.item())+'\n')
            mu = torch.exp(model.mu)
            mu_normalized = mu / (mu.sum(dim=1).unsqueeze(1).expand_as(mu))
            mu_normalized = mu_normalized.detach().numpy()
            sns.heatmap(np.concatenate([mu_normalized, mm_vec], axis=1))
            plt.draw()
            plt.pause(0.1)
            plt.clf()
dump_file.close()
