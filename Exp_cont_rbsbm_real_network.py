import numpy as np
import sys
import model_cont_rbsbm as model_cont
from sklearn.metrics import f1_score,normalized_mutual_info_score
from sklearn.preprocessing import MinMaxScaler
import seaborn as sn
import matplotlib.pyplot as plt

sys.path.append('./Data/Sinanet')

import sinanet_cont_reader as scr


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


def relabel_communities(predicted, actual):
    num_communities = actual.max() + 1
    num_nodes = predicted.shape[0]
    comm_relabel = list(range(num_communities))
    for i in range(num_communities):
        temp = np.array(predicted == i,dtype=int)
        best_comm = 0
        max_overlap = 0
        for j in range(num_communities):
            temp2 = np.array(actual == j,dtype=int)
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

_,adj_mat,n_features,ground_truth = scr.read_sinanet('./Data/Sinanet')
ground_truth = np.asarray(ground_truth)
n_features = n_features.todense()
scaler = MinMaxScaler()
scaler.fit(n_features)
node_features = scaler.transform(n_features)

# Some useful variables
num_nodes = adj_mat.shape[0]
num_communities = np.max(reflow_comm_numbers(ground_truth)) + 1
num_features = node_features.shape[1]
batch_size = min(batch_par, num_nodes)

print('Number of nodes:', num_nodes)
print('Number of attributes:', num_features)
print('Number of edges:', adj_mat.sum())
print('P(edge):', adj_mat.sum() / (num_nodes * (num_nodes - 1)))
print('Number of communities:', num_communities)

# Get the edges
indices = np.nonzero(adj_mat)
edges = []
for i in range(indices[0].shape[0]):
    edges.append((indices[0][i], indices[1][i]))

# Compute prior for B matrix
alphas = np.ones((num_communities, num_communities))
betas = np.ones((num_communities, num_communities)) + \
        prior_rat * (np.ones((num_communities, num_communities)) - np.eye(num_communities))


Max_iters = 1001
num_trials = 50
dump_file = open(logfilepath+'log.txt','w')
plt.ion()

for trial in range(num_trials):
    # Set up the model
    model = model_cont.RBMSBM(num_nodes, num_communities, num_features, alphas, betas)

    # Train the model
    for i in range(Max_iters):
        # Get the communities
        communities = np.argmax(model.q, axis=1)
        communities = relabel_communities(communities,ground_truth)

        # Show the scores
        if i % 10 == 0:
            print('Iter:', i+1,' NMI:', normalized_mutual_info_score(ground_truth, communities,average_method='arithmetic'))
            print('\n')
            dump_file.write(str(trial+1) + '\t' +str(i + 1) + '\t' + str(normalized_mutual_info_score(ground_truth, communities,average_method='arithmetic')) + '\n')
            sn.heatmap(model.q, vmin=0, vmax=1)
            plt.draw()
            plt.pause(0.1)
            plt.clf()

        # Run the E-Step
        model.variational_e_step(edges, adj_mat.todense(), node_features,
                             indices=np.random.randint(0, num_nodes, size=batch_size).tolist(),
                             lamda=1 - lamda_frac * (1 - i/float(Max_iters)))
        # Run the M-step
        for _ in range(1):
            model.variational_m_step(node_features, chain_length=chn_len, num_samples=num_samp, lr=learning_rate, momentum=momentum_rate, use_persistence=True)

dump_file.close()
