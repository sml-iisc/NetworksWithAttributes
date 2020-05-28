import numpy as np
import model_base_rbsbm as model
import gen_network as gn
from sklearn.metrics.cluster import normalized_mutual_info_score
import time
import sys

"""
Arguments to be passed to the program

learning_rate : Learning rate to be used by the RBM
momentum_rate : momentum used to be by the RBM
prior_rat : Controls the value of off-diagonal elements in prior B matrix which is set to 1/(1 + prior_rat)
lamda_frac : Lambda hyperparameter used in the simulated annealing step heuristic
chn_len : Length of the persistent chain used in CD algorithm pertaining to RBM
num_samp : Number of samples to be used to approximate the Expectation in CD algorithm pertaining to RBM
batch_par : Batch size hyperparamter. Dictates the number of latent factors (corresponding to nodes) to be updated in a single E-step
logfilepath : Path of the log file
"""

learning_rate = float(sys.argv[1])
momentum_rate = float(sys.argv[2])
prior_rat = float(sys.argv[3])
lamda_frac = float(sys.argv[4])
chn_len = int(sys.argv[5])
num_samp = int(sys.argv[6])
batch_par = int(sys.argv[7])
logfilepath = sys.argv[8]


def reflow_comm_numbers(comm):
    num_nodes = comm.shape[0]
    act_comm = dict()
    count = 0
    for i in range(num_nodes):
        if comm[i] in act_comm:
            comm[i] = act_comm[comm[i]]
        else:
            act_comm[comm[i]] = count
            comm[i] = count
            count += 1
    return comm


def gen_net(N, K, M, alphas=None, betas=None, p_pos=0.1, p_neg=0.1):
    """
    :param N: Number of nodes
    :param K: Number of communities
    :param M: Number of attributes
    :param alphas: Prior terms for computing B matrix
    :param betas: Prior terms for computing B matrix
    :param p_pos: Probability of a positive attribute
    :param p_neg: Probability of a negative attribute
    :return: adj_mat, node_features, ground_truth
    """
    # Set alphas and betas if None
    if alphas is None:
        alphas = np.ones((K, K))
    if betas is None:
        betas = (np.ones((K, K)) + 10 * (np.ones((K, K)) - np.eye(K))) * np.sqrt(N)

    # Initialize the block matrix
    B = alphas / (alphas + betas)

    # Initialize positive neutral and negative attributes for each community
    W = np.zeros((M, K))
    for m in range(M):
        for k in range(K):
            u = np.random.random()
            if u <= p_pos:
                W[m, k] = np.random.normal(loc=5, scale=0.1)
            elif p_pos < u <= p_pos + p_neg:
                W[m, k] = np.random.normal(loc=-5, scale=0.1)
            else:
                W[m, k] = np.random.normal(scale=0.1)

    W = 1 / (1 + np.exp(- W + 2))

    # Get the generated network
    adj_mat, node_attributes, ground_truth = gn.gen_network_sbm_loop(W, B, N, 10000)

    return adj_mat, node_attributes, ground_truth

Max_iters = 1001

for N in [100, 1000, 5000, 10000, 20000, 50000, 100000]:
    K = int(np.ceil(np.log2(N)))
    M = 100
    dump_file = open(logfilepath + str(N) + '.txt', 'w')
    for trial in range(1):
        # Get the network
        print('Generating network:')
        adj_mat, node_features, ground_truth = gen_net(N, K, M)
        ground_truth = reflow_comm_numbers(np.asarray(ground_truth))
        K = ground_truth.max() + 1
        print('Done')
        print('N:', N, '\nK:', K, '\nM:', M)
        print('Sparsity:', np.sum(adj_mat) / (N ** 2))
        dump_file.write('Trial:\t' + str(trial + 1) + '\tN:\t' + str(N) + '\tK:\t' + str(K) + '\tM:\t' + str(M) +
                        '\tSparsity:\t' + str(np.sum(adj_mat) / (N ** 2)) + '\n')

        # Prepare the prior terms for B matrix
        alphas = np.ones((K, K))
        betas = (np.ones((K, K)) + prior_rat * (np.ones((K, K)) - np.eye(K))) * np.sqrt(N)

        # Prepare the model
        batch_size = min(batch_par, N)
        rbmsbm = model.RBMSBM(N, K, M, alphas, betas)

        # Get the edges
        indices = np.nonzero(adj_mat)
        edges = []
        for i in range(indices[0].shape[0]):
            edges.append((indices[0][i], indices[1][i]))

        curr_time = time.time()
        # Train the model
        for i in range(1, Max_iters):

            # Show the scores
            if i % 10 == 0:
                time_diff = time.time() - curr_time
                communities = reflow_comm_numbers(np.argmax(rbmsbm.q, axis=1))
                nmi = normalized_mutual_info_score(ground_truth, communities, average_method='arithmetic')
                nc = communities.max() + 1
                print('Iter:', i, 'Num Communities:', nc)
                print('NMI:', nmi)
                print('\n')
                dump_file.write(
                    'Trial:\t' + str(trial + 1) + '\tIter:\t' + str(i) + '\tNMI:\t' + str(nmi) + '\tNumComm:\t' +
                    str(nc) + '\tTime:\t' + str(time_diff) + '\n')
                curr_time = time.time()

            # Run the E-Step
            rbmsbm.variational_e_step(edges, adj_mat, node_features,
                                      indices=np.random.randint(0, N, size=batch_size).tolist(),
                                      lamda=1 - lamda_frac * (1 - i/float(Max_iters)))

            # Run the M-step
            rbmsbm.variational_m_step(node_features, chain_length=chn_len, num_samples=num_samp, lr=learning_rate, momentum=momentum_rate,
                                      use_persistence=True)
    dump_file.close()
