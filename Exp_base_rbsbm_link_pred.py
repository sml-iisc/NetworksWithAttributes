import numpy as np
import sys
import model_base_rbsbm_link_pred as model3
from sklearn.metrics import f1_score, normalized_mutual_info_score, roc_auc_score
from scipy.sparse import csr_matrix
import seaborn as sn
import matplotlib.pyplot as plt

sys.path.append('../Data/FacebookEgo')
sys.path.append('../Data/Sinanet')
sys.path.append('../Data/cora')
sys.path.append('../Data/citeseer')
sys.path.append('../Data/LazegaLawyers')
sys.path.append('../Data/Philosophers')

import facebook_reader as fbr
import sinanet_reader as snr
import read_cora as rcr
import read_citeseer as rcs
import el_reader as elr
import read_philosophers as rph

"""
Arguments to be passed to the program

learning_rate : Learning rate to be used by the RBM
momentum_rate : momentum used to be by the RBM
prior_rat : Controls the value of off-diagonal elements in prior B matrix which is set to 1/(1 + prior_rat)
lamda_frac : Lambda hyperparameter used in the simulated annealing step heuristic
chn_len : Length of the persistent chain used in CD algorithm pertaining to RBM
num_samp : Number of samples to be used to approximate the Expectation in CD algorithm pertaining to RBM
batch_par : Batch size hyperparamter. Dictates the number of latent factors (corresponding to nodes) to be updated in a single E-step
dataset : Dataset to be used for the experiment. Specify one of cora, citeseer, Sinanet, Philosophers
"""


learning_rate = float(sys.argv[1])
momentum_rate = float(sys.argv[2])
prior_rat = float(sys.argv[3])
lamda_frac = float(sys.argv[4])
chn_len = int(sys.argv[5])
num_samp = int(sys.argv[6])
batch_par = int(sys.argv[7])
logfilepath = sys.argv[8]
dataset = sys.argv[9]


def relabel_communities(predicted, actual):
    num_communities = actual.max() + 1
    num_nodes = predicted.shape[0]
    comm_relabel = list(range(num_communities))
    for i in range(num_communities):
        temp = predicted == i
        best_comm = 0
        max_overlap = 0
        for j in range(num_communities):
            temp2 = actual == j
            if np.dot(temp, temp2) > max_overlap:
                max_overlap = np.dot(temp, temp2)
                best_comm = j
        comm_relabel[i] = best_comm

    for i in range(num_nodes):
        predicted[i] = comm_relabel[predicted[i]]

    return predicted


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


def evaluate_communities(predicted, actual, eval_fn=f1_score, param=None):
    """
    Follows method presented in CESNA paper to evaluate communities
    """
    # Reflow the numbers
    actual = reflow_comm_numbers(actual)
    predicted = reflow_comm_numbers(predicted)
    num_actual = np.max(actual) + 1
    num_pred = np.max(predicted) + 1

    # Compare actual to predicted
    actual_predicted = 0
    for i in range(num_actual):
        max_score = 0
        for j in range(num_pred):
            if param is None:
                score = eval_fn(actual == i, predicted == j)
            else:
                score = eval_fn(actual == i, predicted == j, average=param)
            if score > max_score:
                max_score = score
        actual_predicted += max_score
    actual_predicted /= (2 * num_actual)

    # Compare predicted to actual
    predicted_actual = 0
    for i in range(num_pred):
        max_score = 0
        for j in range(num_actual):
            if param is None:
                score = eval_fn(predicted == i, actual == j)
            else:
                score = eval_fn(predicted == i, actual == j, average=param)
            if score > max_score:
                max_score = score
        predicted_actual += max_score
    predicted_actual /= (2 * num_pred)

    return actual_predicted + predicted_actual


# Read the data
if(dataset == 'cora'):
    adj_mat, node_features, ground_truth = rcr.read_cora('./Data/cora')
elif(dataset == 'citeseer'):
    adj_mat, node_features, ground_truth = rcs.read_citeseer('./Data/citeseer')
elif(dataset == 'Sinanet'):
    adj_list, adj_mat, node_features, ground_truth = snr.read_sinanet_bins('./Data/Sinanet')
elif(dataset == 'Philosophers'):
    adj_mat, node_features, names = rph.read_philosophers('./Data/Philosophers', threshold=10)
else:
    adj_mat, node_features, ground_truth = rcr.read_cora('./Data/cora')

ground_truth = np.asarray(ground_truth)
adj_mat = adj_mat.todense()


# Some useful variables
num_nodes = adj_mat.shape[0]
num_communities = np.max(reflow_comm_numbers(ground_truth)) + 1
num_features = node_features.shape[1]
batch_size = min(batch_par, num_nodes)
num_trials = 50
Max_iters = 1000

print('Number of nodes:', num_nodes)
print('Number of attributes:', num_features)
print('Number of edges:', adj_mat.sum())
print('P(edge):', adj_mat.sum() / (num_nodes * (num_nodes - 1)))
print('Number of communities:', num_communities)

# Get the edges
indices = np.nonzero(adj_mat)
indices_negative = np.nonzero(np.ones(adj_mat.shape) - adj_mat)
num_missing = int(0.2 * indices[0].shape[0])
labels = np.concatenate([np.ones((1, num_missing)), np.zeros((1, num_missing))], axis=1).squeeze()

dump_file = open(logfilepath+'log.txt', 'w')

plt.ion()
for trial in range(num_trials):
    # Compute prior for B matrix
    alphas = np.ones((num_communities, num_communities))
    betas = np.ones((num_communities, num_communities)) + \
            10 * (np.ones((num_communities, num_communities)) - np.eye(num_communities))
    
    # Inject missing links such that edges and non-edges are equally likely to be missing
    positive = list(range(indices[0].shape[0]))
    np.random.shuffle(positive)
    missing_edges = [(indices[0][i], indices[1][i]) for i in positive[:num_missing]]

    negative = list(range(indices_negative[0].shape[0]))
    np.random.shuffle(negative)
    missing_edges += [(indices_negative[0][i], indices_negative[1][i]) for i in negative[:num_missing]]

    # Calculate the observed adjacency matrix
    A_unk = np.zeros((num_nodes, num_nodes))
    for i, j in missing_edges:
        A_unk[i, j] = 1
    A_obs = csr_matrix(np.multiply(adj_mat, (np.ones(A_unk.shape) - A_unk)))
    A_unk = csr_matrix(A_unk)

    # Get the edges
    indices_obs = np.nonzero(A_obs)
    edges = []
    for i in range(indices_obs[0].shape[0]):
        edges.append((indices_obs[0][i], indices_obs[1][i]))

    # Set up the model
    model = model3.RBMSBM(num_nodes, num_communities, num_features, alphas, betas)

    # Train the model
    for i in range(Max_iters):
        # Get the communities
        communities = np.argmax(model.q, axis=1)

        # Show the scores
        if i % 10 == 0 or i == 999:
            print('Trial:', trial + 1,'Iter:', i + 1,'NMI:', normalized_mutual_info_score(ground_truth, communities))
            dump_file.write(str(trial+1) + '\t' +str(i + 1) + '\t' + str(normalized_mutual_info_score(ground_truth, communities,average_method='arithmetic')) + '\n')

        # Run the E-Step
        model.variational_e_step(edges, missing_edges, A_obs, A_unk, node_features,
                                 indices=np.random.randint(0, num_nodes, size=batch_size).tolist(),
                                 lamda=1 - lamda_frac * (1 - i/float(Max_iters)))

        # Run the M-step
        for _ in range(1):
            model.variational_m_step(node_features, chain_length=chn_len, num_samples=num_samp, lr=learning_rate, momentum=momentum_rate,
                                 use_persistence=True)

    probs = model.predict(missing_edges)
    auc = roc_auc_score(labels, probs)
    print('Trial:', trial + 1,'AUC:', auc)
    dump_file.write(str(trial + 1) + '\tAUC\t' + str(auc) + '\n')

dump_file.close()
