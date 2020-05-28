import numpy as np
from scipy.special import digamma

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x, axis=-1, take_exp=True):
    if take_exp:
        x = x - np.expand_dims(x.max(axis=axis), axis=axis).repeat(x.shape[axis], axis)
        x = np.exp(x)
    return x / np.expand_dims(x.sum(axis), axis=axis).repeat(x.shape[axis], axis)


def xavier_init(shape0, shape1):
    return np.random.normal(loc=0, scale=2/(shape0 + shape1), size=(shape0, shape1))


class RBMSBM(object):
    """
    Implements the model that combines RBM and SBM for directed networks with a prior on B. Uses the standard
    implementation of RBM's contrastive divergence
    """
    def __init__(self, N, K, M, alphas, betas):
        """
        :param N: Number of nodes
        :param K: Number of communities
        :param M: Number of attributes
        :param alphas: (K, K) matrix of prior alphas for B
        :param betas: (K, K) matrix of prior betas for B
        """
        super(RBMSBM, self).__init__()
        self.N = N
        self.K = K
        self.M = M
        self.alphas = alphas
        self.betas = betas

        # Initialize the weights and biases for RBM
        self.W = xavier_init(self.M, self.K)
        self.b = np.zeros((self.M, 1))
        self.c = np.zeros((self.K, 1))

        # Initialize the parameters for posterior on block matrix
        self.alphas_post = alphas
        self.betas_post = betas

        # Initialize gradients
        self.grad_b = np.zeros(self.b.shape)
        self.grad_c = np.zeros(self.c.shape)
        self.grad_W = np.zeros(self.W.shape)

        # Initialize posterior for class membership
        self.q = np.ones((self.N, self.K))
        self.q = self.q / np.expand_dims(self.q.sum(axis=1), axis=1).repeat(repeats=self.K, axis=1)

        # Initialize samples from RBM for persistent CD
        self.y_samples = None

    def rbm_sample(self, y=None, z=None, chain_length=10, num_samples=1, start_with_z=True):
        """
        y: (num_samples, M) binary starting point for observable features
        z: (num_samples, K) one hot starting point for communities
        chain_length: Number of steps to take for Gibbs sampling
        num_samples: Number of samples to generate
        start_with_z: Whether to start the Gibbs chain by sampling z
        returns: y, z
            y: (num_samples, M) Sampled y values
            z: (num_samples, K) Sampled z values
        """
        if y is None:
            y = (np.random.random(size=(num_samples, self.M)) <= 0.5).astype(float)

        if z is None:
            idx = np.random.choice(self.K, size=num_samples, replace=True)
            z = np.zeros((num_samples, self.K))
            z[np.arange(num_samples), idx] = 1

        for _ in range(chain_length):
            if start_with_z:
                z = softmax(np.matmul(y, self.W) + self.c.repeat(num_samples, axis=1).T)    # num_samples x K
                idx = [np.random.choice(self.K, size=1, p=z[i, :])[0] for i in range(num_samples)]
                z = np.zeros((num_samples, self.K))
                z[np.arange(num_samples), idx] = 1
                y = sigmoid(np.matmul(z, self.W.T) + self.b.repeat(num_samples, axis=1).T) # num_samples x M
                y = (np.random.random(y.shape) <= y).astype(float)
            else:
                y = sigmoid(np.matmul(z, self.W.T) + self.b.repeat(num_samples, axis=1).T) # num_samples x M
                y = (np.random.random(y.shape) <= y).astype(float)
                z = softmax(np.matmul(y, self.W) + self.c.repeat(num_samples, axis=1).T)    # num_samples x K
                idx = [np.random.choice(self.K, size=1, p=z[i, :])[0] for i in range(num_samples)]
                z = np.zeros((num_samples, self.K))
                z[np.arange(num_samples), idx] = 1
        z = softmax(np.matmul(y, self.W) + self.c.repeat(num_samples, axis=1).T)
        return y, z

    def variational_e_step(self, edges, A, Y, indices=None, lamda=0.5):
        """
        :param edges: List of edges in the network
        :param A: (N, N) binary adjacency matrix
        :param Y: (N, M) binary node feature matrix
        :param lamda: Regularization parameter
        :param indices: List of indices of nodes that are to be updated
        """
        if indices is None:
            indices = range(self.N)

        # Update the posterior on B
        q_prod_edges = np.zeros((self.K, self.K))
        for i, j in edges:
            q_prod_edges += np.matmul(self.q[i, :].reshape((-1, 1)), self.q[j, :].reshape((1, -1)))

        q_sum = self.q.sum(axis=0).reshape((-1, 1))
        residue = np.matmul(self.q.T, self.q)
        q_prod_all = np.matmul(q_sum, q_sum.T) - residue

        self.alphas_post = q_prod_edges + self.alphas
        self.betas_post = q_prod_all - q_prod_edges + self.betas

        # Update the posterior on community memberships
        digamma_alphas = digamma(self.alphas_post)
        digamma_betas = digamma(self.betas_post)
        digamma_sum = digamma(self.alphas_post + self.betas_post)

        q_alpha_prod = np.matmul(self.q, digamma_alphas)
        q_beta_prod = np.matmul(self.q, digamma_betas)
        q_sum_prod = np.matmul(self.q, digamma_sum)
        q_alpha_prod_t = np.matmul(self.q, digamma_alphas.T)
        q_beta_prod_t = np.matmul(self.q, digamma_betas.T)
        q_sum_prod_t = np.matmul(self.q, digamma_sum.T)

        def h(x, l=0.5):
            return (x <= 0.5) * (x**l * 2**(l-1)) + (x > 0.5) * (1 - 2**(l-1) * (1 - x)**l)

        for idx in indices:
            a_rep = np.asarray(A[idx, :].todense().reshape((-1, 1)).repeat(self.K, axis=1))
            a_rep_comp = 1 - a_rep

            temp1 = (a_rep * (q_alpha_prod - q_sum_prod)).sum(axis=0)
            temp2 = (a_rep_comp * (q_beta_prod - q_sum_prod)).sum(axis=0) - q_beta_prod_t[idx, :] + q_sum_prod_t[idx, :]

            a_rep = np.asarray(A[:, idx].todense().reshape((-1, 1)).repeat(self.K, axis=1))
            a_rep_comp = 1 - a_rep

            temp3 = (a_rep * (q_alpha_prod_t - q_sum_prod_t)).sum(axis=0)
            temp4 = (a_rep_comp * (q_beta_prod_t - q_sum_prod_t)).sum(axis=0) - q_beta_prod_t[idx, :] + \
                    q_sum_prod_t[idx, :]

            self.q[idx, :] = temp1 + temp2 + temp3 + temp4 + np.matmul(Y[idx, :].todense(), self.W) + self.c[:, 0]
            self.q[idx, :] = softmax(self.q[idx, :], axis=0)
            self.q[idx, :] = h(self.q[idx, :], lamda)
            self.q[idx, :] = softmax(self.q[idx, :], axis=0, take_exp=False)

    def variational_m_step(self, Y, num_samples=1, chain_length=10, lr=1e-2, momentum=0.0, use_persistence=False):
        """
        :param Y: (N, M) observed binary feature matrix
        :param num_samples: Number of samples to use for approximating expectations
        :param chain_length: Length of Gibbs chain for RBM sampling
        :param lr: Learning rate for parameter updates
        :param momentum: Momentum term for SGD update
        :param use_persistence: Whether to use persistent-CD or not
        """
        # Get samples from RBM
        if not use_persistence or self.y_samples is None:
            self.y_samples = np.asarray(Y[np.random.choice(self.N, size=num_samples), :].todense())
        y, z = self.rbm_sample(chain_length=chain_length, num_samples=num_samples, y=self.y_samples)
        self.y_samples = y

        # Compute gradients for RBM parameters
        grad_b = (-self.N * y.mean(axis=0) + Y.sum(axis=0)).reshape((-1, 1))
        grad_c = (-self.N * z.mean(axis=0) + self.q.sum(axis=0)).reshape((-1, 1))
        grad_W = -self.N * (np.matmul(y.T, z) / num_samples) + np.matmul(Y.T.todense(), self.q)

        # Update the RBM parameters
        self.grad_b = momentum * self.grad_b + (1 - momentum) * grad_b / self.N
        self.b += lr * self.grad_b
        self.grad_c = momentum * self.grad_c + (1 - momentum) * grad_c / self.N
        self.c += lr * self.grad_c
        self.grad_W = momentum * self.grad_W + (1 - momentum) * grad_W / self.N
        self.W += lr * self.grad_W

        self.W = self.W.clip(-5, 5)
        self.b = self.b.clip(-5, 5)
        self.c = self.c.clip(-5, 5)


if __name__ == '__main__':
    model = RBMSBM(N=100, K=5, M=10)
