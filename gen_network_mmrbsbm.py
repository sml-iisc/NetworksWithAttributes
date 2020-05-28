import numpy as np
import scipy.stats as st


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x, axis=-1):
    x = x - np.expand_dims(x.max(axis=axis), axis=axis).repeat(x.shape[axis], axis)
    x = np.exp(x)
    return x / np.expand_dims(x.sum(axis), axis=axis).repeat(x.shape[axis], axis)


class exp_pdf(st.rv_continuous):
    """
    Implements a custom distribution used for sampling in RBM
    """
    def _pdf(self, x, beta):
        return np.exp(beta * x) * beta /(np.exp(beta * self.b) - np.exp(beta * self.a))

    def _argcheck(self, beta):
        return beta != 0

    def _cdf(self, x, beta):
        if x >= self.b:
            return 1.0
        elif x <= self.a:
            return 0.0
        else:
            return (np.exp(beta * x) - np.exp(beta * self.a))/(np.exp(beta * self.b) - np.exp(beta * self.a))

    def _ppf(self, x, beta):
        return np.log(np.exp(beta * self.a) + x * (np.exp(beta * self.b) - np.exp(beta * self.a)))/beta


def rbm_sample(M, K, W, b, c, chain_length=10, num_samples=1, start_with_z=True):
    """
    :param M: Number of attributes
    :param K: Number of communities
    :param W: (M, K) Weight matrix for RBM
    :param b: (M, 1) Bias terms for features
    :param c: (K, 1) Bias terms for communities
    :param chain_length: Number of steps to take for Gibbs sampling
    :param num_samples: Number of samples to generate
    :param start_with_z: Whether to start the Gibbs chain by sampling z
    returns: y, z
        y: (num_samples, M) Sampled y values
        z: (num_samples, K) Sampled z values
    """

    y = (np.random.random(size=(num_samples, M)) <= 0.5).astype(float)
    z = np.random.dirichlet(np.ones(K)/K, size=num_samples)

    for _ in range(chain_length):
        if start_with_z:
            for k in range(K):
                K_d = np.random.choice(np.delete(np.arange(K), k))
                betak = np.matmul(y, W[:, k]-W[:, K_d]) + c[k] - c[K_d]
                sumk =  z[:, k] + z[:, K_d]
                for i in range(num_samples):
                    dist = exp_pdf(a=0, b=sumk[i])
                    sampl = dist.rvs(beta=betak[i])
                    z[i, k] = sampl
                    z[i, K_d] = sumk[i] - sampl
            y = sigmoid(np.matmul(z, W.T) + b.repeat(num_samples, axis=1).T)  # num_samples x M
            y = (np.random.random(y.shape) <= y).astype(float)
        else:
            y = sigmoid(np.matmul(z, W.T) + b.repeat(num_samples, axis=1).T)  # num_samples x M
            y = (np.random.random(y.shape) <= y).astype(float)
            for k in range(K):
                K_d= np.random.choice(K)
                betak = np.matmul(y, W[:, k]-W[:, K_d]) + c[k] - c[K_d]
                sumk = z[:, k] + z[:, K_d]
                for i in range(num_samples):
                    dist = exp_pdf(a=0, b=sumk[i])
                    sampl = dist.rvs(beta=betak[i])
                    z[i, k] = sampl
                    z[i, K_d] = sumk[i] - sampl
    return y, z


# generates synthetic mixed membership random graphs
def gen_network_mmrbsbm(W, b, c, B, N=100):
    """
    Generates a synthetic random graph using RBMSBM model
    :param N: Number of nodes in the graph
    :param W: (M, K) Weight matrix for RBM
    :param b: (M, 1) Bias terms for features
    :param c: (K, 1) Bias terms for communities
    :param B: (K, K) Block matrix for SBM
    :return: adj_mat, features, mixed_membership_vec
        adj_mat: (N x N) adjacency matrix with no self loop
        features: (N x M) feature matrix, each row represents a node
        mixed_membership_vec: (N x K) mixed membership vectors
    """

    # Useful variables
    M = b.shape[0]
    K = c.shape[0]

    adj_mat = np.zeros((N, N))
    # Sample the nodes features are communities
    features, mixed_membership_vec = rbm_sample(M, K, W, b, c, chain_length=100, num_samples=N)

    zp_pq = np.zeros((N, N, K))
    zq_pq = np.zeros((N, N, K))
    for q in range(N):
        for p in range(q, N):
            zp_pq[p, q, :] = np.random.multinomial(1, mixed_membership_vec[p, :])
            zq_pq[p, q, :] = np.random.multinomial(1, mixed_membership_vec[q, :])
            prob_pq = np.matmul(zp_pq[p, q, :].reshape(1, -1), np.matmul(B, zq_pq[p, q, :].reshape(-1, 1)))
            adj_mat[p, q] = (np.random.rand() <= prob_pq).astype(float)

    return adj_mat, features, mixed_membership_vec, zp_pq, zq_pq


def gen_network_mmrbsbm_1(M, K, B, N=100):
    # Useful variables

    adj_mat = np.zeros((N, N))
    mixed_membership_vec = np.random.dirichlet(0.1 * np.ones(K), size=N)

    F = np.random.randn(K, M) > 0.1
    features = np.zeros((N, M))
    for i in range(N):
        for j in range(M):
            features[i, j] = np.dot(mixed_membership_vec[i, :] > 0.2, F[:, j])
    features = features.astype(float)

    zp_pq = np.zeros((N, N, K))
    zq_pq = np.zeros((N, N, K))
    for q in range(N):
        for p in range(q+1, N):
            zp_pq[p, q, :] = np.random.multinomial(1, mixed_membership_vec[p, :])
            zq_pq[p, q, :] = np.random.multinomial(1, mixed_membership_vec[q, :])
            prob_pq = np.matmul(zp_pq[p, q, :].reshape(1, -1), np.matmul(B, zq_pq[p, q, :].reshape(-1, 1)))
            adj_mat[p, q] = (np.random.binomial(n=1, p=prob_pq)).astype(float)

    return adj_mat, features, mixed_membership_vec, zp_pq, zq_pq, F
