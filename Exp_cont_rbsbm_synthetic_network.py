import numpy as np
import sys
import model_cont_rbsbm as model_cont
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score,normalized_mutual_info_score
from sklearn.preprocessing import MinMaxScaler
import time
np.set_printoptions(threshold=sys.maxsize)


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


#Generate Synthetic Network

def gen_net(M,K,N,mu_sig,sig):

    alphas = np.ones((K, K))
    betas = (np.ones((K, K)) + 5 * (np.ones((K, K)) - np.eye(K))) * np.sqrt(N)
    B = alphas / (alphas + betas)
    adj_mat = np.zeros((N,N))
    ground_truth = np.random.randint(low=0,high=K,size=N)

    for p in range(N):
        for q in range(N):
            prob_pq = B[ground_truth[p],ground_truth[q]]
            adj_mat[p,q] = np.random.binomial(size=1,n=1,p=prob_pq)

    mu = np.random.multivariate_normal(np.zeros((M)),np.eye(M)*mu_sig,size=K)
    sigma = np.eye(M) * sig
    features = np.zeros((N,M))
    for i in range(N):
        features[i,:] = np.random.multivariate_normal(mu[ground_truth[i],:],sigma)

    return adj_mat,features,ground_truth

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

Max_iters = 1001

for N in [100, 1000, 5000, 10000]:
    K = int(np.ceil(np.log10(N)))
    M = 25
    dump_file = open(logfilepath + str(N) + '.txt','w')
    batch_size = min(batch_par,N)
    mu_sig = 6 * int(np.ceil(np.log10(N)))
    sig = 1.5 * int(np.ceil(np.log10(N)))
    for trial in range(3):
        # Generate the network
        print('Generating network:')
        adj_mat, features, ground_truth = gen_net(M,K,N,mu_sig,sig)
        print('Done')
        print('N:', N, '\nK:', K, '\nM:', M)
        print('Sparsity:', np.sum(adj_mat) / (N ** 2))
        dump_file.write('Trial:\t' + str(trial + 1) + '\tN:\t' + str(N) + '\tK:\t' + str(K) + '\tM:\t' + str(M) + '\tSparsity:\t' + str(np.sum(adj_mat) / (N ** 2)) + '\n')
        #Preprocess
        #Normalize features
        scaler = MinMaxScaler()
        scaler.fit(features)
        feat_norm = scaler.transform(features)
        # Get the edges
        indices = np.nonzero(adj_mat)
        edges = []
        for i in range(indices[0].shape[0]):
            edges.append((indices[0][i], indices[1][i]))

        #Compute prior for B matrix
        alphas = np.ones((K,K))
        betas = (np.ones((K, K)) + prior_rat * (np.ones((K, K)) - np.eye(K))) * np.sqrt(N)

        model = model_cont.RBMSBM(N,K,M,alphas,betas)

        plt.ion()
        curr_time = time.time()
        # Train the Model
        for i in range(1,Max_iters):
           #Run the E-step
            model.variational_e_step(edges,adj_mat,feat_norm,
                    indices=np.random.randint(0, N, size=batch_size).tolist(),
                    lamda = 1 - lamda_frac * (1 - i/float(Max_iters)))
            #Run the M-step
            for _ in range(10):
                model.variational_m_step(feat_norm, chain_length=chn_len, num_samples=num_samp, lr=learning_rate, momentum=momentum_rate, use_persistence=True)

            #Get the communities
            communities = np.argmax(model.q,axis=1)
            communities = relabel_communities(communities,ground_truth)
            # Show the scores
            if(i%10 ==0):
                time_diff = time.time() - curr_time
                nmi = normalized_mutual_info_score(ground_truth, communities,average_method='arithmetic')
                f1 = f1_score(ground_truth,communities,average='micro')
                print('Iter:',i)
                print('NMI:',nmi)
                print('F1:',f1)
                print('\n')
                dump_file.write('Trial:\t'+str(trial+1)+'\tIter:\t'+str(i)+'\tF1:\t'+str(f1)+'\tNMI:\t'+str(nmi)+'\tTime:\t'+str(time_diff)+'\n')
                curr_time = time.time()
                sn.heatmap(model.q,vmin=0,vmax=1)
                plt.draw()
                plt.pause(0.1)
                plt.clf()

    dump_file.close()
